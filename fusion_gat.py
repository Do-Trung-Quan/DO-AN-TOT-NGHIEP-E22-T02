"""Train a first multimodal fusion baseline with a static timeseries kNN graph.

Expected local files:
  train_vectors.npy, train_meta.csv, timeseries_features.parquet

The image vectors are matched to the parquet rows by a filename-like column.
The graph is built from standardized numeric timeseries features. Missing image
vectors are represented by zeros plus an explicit availability mask.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn


def normalize_name(value: object) -> str:
    name = Path(str(value)).name.lower()
    name = re.sub(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def find_match_column(meta: pd.DataFrame, clinical: pd.DataFrame) -> str:
    meta_keys = {normalize_name(v) for v in meta["fname"]}
    candidates = []
    for column in clinical.columns:
        if not (pd.api.types.is_object_dtype(clinical[column])
                or pd.api.types.is_string_dtype(clinical[column])):
            continue
        values = clinical[column].dropna().head(10000)
        overlap = sum(normalize_name(v) in meta_keys for v in values)
        if overlap:
            candidates.append((overlap, column))
    if not candidates:
        raise ValueError(
            "Khong tim thay cot ID/filename trong timeseries_features.parquet "
            "co the match voi train_meta.csv."
        )
    return max(candidates)[1]


def find_label_column(clinical: pd.DataFrame) -> str:
    preferred = ["label", "target", "diagnosis", "class", "y"]
    for column in preferred:
        if column in clinical.columns:
            return column
    raise ValueError(
        "Khong tim thay cot nhan trong timeseries_features.parquet "
        "(thu ten label, target, diagnosis, class hoac y)."
    )


def load_data(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.load(root / "train_vectors.npy").astype(np.float32)
    meta = pd.read_csv(root / "train_meta.csv")
    clinical = pd.read_parquet(root / "timeseries_features.parquet")
    if len(vectors) != len(meta):
        raise ValueError(f"Vector va metadata lech so dong: {len(vectors)} vs {len(meta)}")

    id_column = find_match_column(meta, clinical)
    clinical = clinical.copy()
    clinical["_key"] = clinical[id_column].map(normalize_name)
    meta = meta.copy()
    meta["_key"] = meta["fname"].map(normalize_name)
    clinical = clinical.drop_duplicates("_key").set_index("_key")
    if len(clinical) < 100:
        raise ValueError(f"Parquet chi co {len(clinical)} mau co ID; dung lai de kiem tra du lieu.")

    numeric = clinical.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in {"label", "idx", "_key"}]
    if len(numeric) < 2:
        raise ValueError("Parquet can it nhat 2 cot timeseries dang so.")
    numeric = numeric[:32]
    all_keys = clinical.index.to_numpy()
    meta_by_key = meta.drop_duplicates("_key").set_index("_key")
    matched_keys = [key for key in all_keys if key in meta_by_key.index]
    if len(matched_keys) < 100:
        raise ValueError(f"Chi match duoc {len(matched_keys)} mau co nhan tu train_meta.csv.")
    keys = np.asarray(matched_keys)
    ts = clinical.loc[keys, numeric].to_numpy(dtype=np.float32)
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    labels = meta_by_key.loc[keys, "label"].to_numpy(dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"Fusion baseline can 2 lop nhan, nhung tim thay {len(np.unique(labels))}.")
    image = np.zeros((len(keys), vectors.shape[1]), dtype=np.float32)
    mask = np.zeros((len(keys), 1), dtype=np.float32)
    vector_index_by_key = meta_by_key["idx"].to_dict()
    for row, key in enumerate(keys):
        if key in vector_index_by_key:
            image[row] = vectors[int(vector_index_by_key[key])]
            mask[row, 0] = 1.0
    print(f"Labeled rows {len(keys)} from train_meta.csv; timeseries={ts.shape[1]}d, "
          f"image coverage={int(mask.sum())}/{len(keys)} using '{id_column}'")
    return ts, image, labels, mask


def make_edges(features: np.ndarray, k: int) -> torch.Tensor:
    scaled = StandardScaler().fit_transform(features)
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(features))).fit(scaled)
    indices = neighbors.kneighbors(return_distance=False)
    source = np.repeat(np.arange(len(features)), indices.shape[1])
    target = indices.reshape(-1)
    return torch.tensor(np.stack([source, target]), dtype=torch.long)


class GATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attention = nn.Linear(output_dim * 2, 1, bias=False)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        source, target = edges
        scores = nn.functional.leaky_relu(
            self.attention(torch.cat([projected[source], projected[target]], dim=1)).squeeze(1),
            negative_slope=0.2,
        )
        weights = torch.zeros_like(scores)
        for node in torch.unique(target):
            selected = target == node
            weights[selected] = torch.softmax(scores[selected], dim=0)
        aggregated = torch.zeros_like(projected)
        aggregated.index_add_(0, target, projected[source] * weights.unsqueeze(1))
        return aggregated


class FusionGAT(nn.Module):
    def __init__(self, ts_dim: int, image_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.ts_projection = nn.Linear(ts_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.gat1 = GATLayer(hidden_dim * 2 + 1, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, ts: torch.Tensor, image: torch.Tensor,
                image_mask: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        ts_embedding = torch.relu(self.ts_projection(ts))
        image_embedding = torch.relu(self.image_projection(image))
        node_features = torch.cat([ts_embedding, image_embedding, image_mask], dim=1)
        hidden = torch.relu(self.gat1(node_features, edges))
        hidden = torch.dropout(self.gat2(hidden, edges), p=0.25, train=self.training)
        return self.classifier(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(__file__).resolve().parent
    ts, image, labels, image_mask = load_data(root)
    scaler = StandardScaler()
    ts = scaler.fit_transform(ts).astype(np.float32)
    image = StandardScaler().fit_transform(image).astype(np.float32)
    edges = make_edges(ts, args.k)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, random_state=args.seed, stratify=labels[train_idx]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [torch.tensor(value, device=device) for value in (ts, image, image_mask)]
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    edges = edges.to(device)
    model = FusionGAT(ts.shape[1], image.shape[1]).to(device)
    counts = np.bincount(labels[train_idx], minlength=2)
    class_weights = torch.tensor(
        len(train_idx) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(*tensors, edges)
        loss = nn.functional.cross_entropy(logits[train_idx], labels_tensor[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                probabilities = logits.softmax(1)[:, 1].cpu().numpy()
            valid_pred = (probabilities[valid_idx] >= 0.5).astype(int)
            print(f"epoch={epoch:03d} loss={loss.item():.4f} valid_f1="
                  f"{f1_score(labels[valid_idx], valid_pred, zero_division=0):.4f}")
    model.eval()
    with torch.no_grad():
        probabilities = model(*tensors, edges).softmax(1)[:, 1].cpu().numpy()
    predictions = (probabilities[test_idx] >= 0.5).astype(int)
    metrics = {
        "model": "static-kNN-GAT-fusion",
        "epochs": args.epochs,
        "k": args.k,
        "seed": args.seed,
        "samples": int(len(labels)),
        "timeseries_dim": int(ts.shape[1]),
        "image_dim": int(image.shape[1]),
        "image_coverage": int(image_mask.sum().item()),
        "test_samples": int(len(test_idx)),
        "test_f1": round(float(f1_score(labels[test_idx], predictions, zero_division=0)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(labels[test_idx], predictions)), 4),
        "test_auc": round(float(roc_auc_score(labels[test_idx], probabilities[test_idx])), 4),
    }
    result_path = root / "fusion_gat_results.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("test_f1=", metrics["test_f1"])
    print("test_balanced_accuracy=", metrics["test_balanced_accuracy"])
    print("test_auc=", metrics["test_auc"])
    print("results_saved=", result_path)


if __name__ == "__main__":
    main()
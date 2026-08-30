"""Pairwise Graph Transformer fusion baseline.

Each sample is a graph/sequence of exactly two modality nodes:
  node 0: timeseries
  node 1: image
Self-attention is therefore performed between the two modalities of the same
sample, never between different patients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from fusion_gat import load_data


class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(nodes)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        nodes = nodes + attended
        return nodes + self.feedforward(self.norm_feedforward(nodes))


class PairwiseFusionGraphTransformer(nn.Module):
    def __init__(self, ts_dim: int, image_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.ts_projection = nn.Sequential(
            nn.Linear(ts_dim, hidden_dim), nn.GELU()
        )
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim), nn.GELU()
        )
        self.node_projection = nn.Linear(hidden_dim + 1, hidden_dim)
        self.blocks = nn.ModuleList(
            [GraphTransformerBlock(hidden_dim) for _ in range(2)]
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        ts: torch.Tensor,
        image: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        ts_embedding = self.ts_projection(ts)
        image_embedding = self.image_projection(image)

        # Each sample has two nodes: timeseries and image.
        nodes = torch.stack([ts_embedding, image_embedding], dim=1)

        # Only the image node receives the availability flag.
        mask = torch.zeros(
            nodes.size(0), 2, 1, device=nodes.device, dtype=nodes.dtype
        )
        mask[:, 1, 0] = image_mask.squeeze(-1)
        nodes = torch.cat([nodes, mask], dim=2)
        nodes = self.node_projection(nodes)

        for block in self.blocks:
            nodes = block(nodes)

        fused = nodes.mean(dim=1)
        return self.classifier(fused)


def choose_threshold(
    probabilities: np.ndarray, labels: np.ndarray, valid_idx: np.ndarray
) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        valid_pred = (probabilities[valid_idx] >= threshold).astype(int)
        score = f1_score(labels[valid_idx], valid_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold co dinh; neu bo trong se toi uu F1 tren validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fusion_graph_transformer_pairwise_results.json"),
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(__file__).resolve().parent
    ts, image, labels, image_mask = load_data(root)

    # Split before fitting scalers: validation and test never affect statistics.
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    train_idx, valid_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels[train_idx],
    )

    ts_scaler = StandardScaler()
    image_scaler = StandardScaler()
    ts_scaler.fit(ts[train_idx])
    image_scaler.fit(image[train_idx])
    ts = ts_scaler.transform(ts).astype(np.float32)
    image = image_scaler.transform(image).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_tensor = torch.tensor(ts, device=device)
    image_tensor = torch.tensor(image, device=device)
    mask_tensor = torch.tensor(image_mask, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    model = PairwiseFusionGraphTransformer(ts.shape[1], image.shape[1]).to(device)
    counts = np.bincount(labels[train_idx], minlength=2)
    class_weights = torch.tensor(
        len(train_idx) / (2 * np.maximum(counts, 1)),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(ts_tensor, image_tensor, mask_tensor)
        loss = nn.functional.cross_entropy(
            logits[train_idx], labels_tensor[train_idx], weight=class_weights
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                probabilities = model(
                    ts_tensor, image_tensor, mask_tensor
                ).softmax(1)[:, 1].cpu().numpy()
            valid_pred = (probabilities[valid_idx] >= 0.5).astype(int)
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"valid_f1@0.5={f1_score(labels[valid_idx], valid_pred, zero_division=0):.4f}"
            )

    model.eval()
    with torch.no_grad():
        probabilities = model(
            ts_tensor, image_tensor, mask_tensor
        ).softmax(1)[:, 1].cpu().numpy()

    if args.threshold is None:
        best_threshold, valid_f1 = choose_threshold(probabilities, labels, valid_idx)
        threshold_selection = "validation_f1"
    else:
        best_threshold = args.threshold
        valid_f1 = float(
            f1_score(
                labels[valid_idx],
                (probabilities[valid_idx] >= best_threshold).astype(int),
                zero_division=0,
            )
        )
        threshold_selection = "fixed"
    predictions = (probabilities[test_idx] >= best_threshold).astype(int)
    test_labels = labels[test_idx]
    test_confusion_matrix = confusion_matrix(test_labels, predictions).tolist()
    test_classification_report = classification_report(
        test_labels,
        predictions,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    print("Positive rate:", round(float(test_labels.mean()), 4))
    print("Predicted positive rate:", round(float(predictions.mean()), 4))
    print("Threshold:", best_threshold)
    print("Confusion matrix:")
    print(np.asarray(test_confusion_matrix))
    print("Classification report:")
    print(
        classification_report(
            test_labels, predictions, digits=4, zero_division=0
        )
    )
    metrics = {
        "model": "pairwise-global-graph-transformer-fusion",
        "epochs": args.epochs,
        "seed": args.seed,
        "threshold_selection": threshold_selection,
        "best_threshold": round(best_threshold, 2),
        "valid_f1_at_best_threshold": round(valid_f1, 4),
        "test_positive_rate": round(float(test_labels.mean()), 4),
        "test_predicted_positive_rate": round(float(predictions.mean()), 4),
        "test_confusion_matrix": test_confusion_matrix,
        "test_classification_report": test_classification_report,
        "attention_scope": "two_nodes_within_each_sample",
        "samples": int(len(labels)),
        "timeseries_dim": int(ts.shape[1]),
        "image_dim": int(image.shape[1]),
        "image_coverage": int(image_mask.sum()),
        "train_samples": int(len(train_idx)),
        "valid_samples": int(len(valid_idx)),
        "test_samples": int(len(test_idx)),
        "test_f1": round(
            float(f1_score(test_labels, predictions, zero_division=0)), 4
        ),
        "test_balanced_accuracy": round(
            float(balanced_accuracy_score(test_labels, predictions)), 4
        ),
        "test_auc": round(
            float(roc_auc_score(labels[test_idx], probabilities[test_idx])), 4
        ),
    }
    result_path = root / args.output
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("best_threshold=", metrics["best_threshold"])
    print("valid_f1_at_best_threshold=", metrics["valid_f1_at_best_threshold"])
    print("test_f1=", metrics["test_f1"])
    print("test_balanced_accuracy=", metrics["test_balanced_accuracy"])
    print("test_auc=", metrics["test_auc"])
    print("results_saved=", result_path)


if __name__ == "__main__":
    main()

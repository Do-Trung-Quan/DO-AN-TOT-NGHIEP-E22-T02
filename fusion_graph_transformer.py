"""Global Graph Transformer baseline for timeseries + image fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
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
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        nodes = nodes + attended
        return nodes + self.feedforward(self.norm_feedforward(nodes))


class FusionGraphTransformer(nn.Module):
    def __init__(self, ts_dim: int, image_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.ts_projection = nn.Sequential(nn.Linear(ts_dim, hidden_dim), nn.GELU())
        self.image_projection = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.GELU())
        self.node_projection = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.blocks = nn.ModuleList([GraphTransformerBlock(hidden_dim) for _ in range(2)])
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, ts: torch.Tensor, image: torch.Tensor,
                image_mask: torch.Tensor) -> torch.Tensor:
        ts_embedding = self.ts_projection(ts)
        image_embedding = self.image_projection(image)
        nodes = torch.cat([ts_embedding, image_embedding, image_mask], dim=1)
        nodes = self.node_projection(nodes).unsqueeze(0)
        for block in self.blocks:
            nodes = block(nodes)
        return self.classifier(nodes.squeeze(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(__file__).resolve().parent
    ts, image, labels, image_mask = load_data(root)
    ts = StandardScaler().fit_transform(ts).astype(np.float32)
    image = StandardScaler().fit_transform(image).astype(np.float32)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, random_state=args.seed, stratify=labels[train_idx]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_tensor = torch.tensor(ts, device=device)
    image_tensor = torch.tensor(image, device=device)
    mask_tensor = torch.tensor(image_mask, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    model = FusionGraphTransformer(ts.shape[1], image.shape[1]).to(device)
    counts = np.bincount(labels[train_idx], minlength=2)
    class_weights = torch.tensor(
        len(train_idx) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device
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
                probabilities = model(ts_tensor, image_tensor, mask_tensor).softmax(1)[:, 1].cpu().numpy()
            valid_pred = (probabilities[valid_idx] >= args.threshold).astype(int)
            print(f"epoch={epoch:03d} loss={loss.item():.4f} valid_f1="
                  f"{f1_score(labels[valid_idx], valid_pred, zero_division=0):.4f}")
    model.eval()
    with torch.no_grad():
        probabilities = model(ts_tensor, image_tensor, mask_tensor).softmax(1)[:, 1].cpu().numpy()
    predictions = (probabilities[test_idx] >= args.threshold).astype(int)
    metrics = {
        "model": "global-graph-transformer-fusion",
        "epochs": args.epochs,
        "seed": args.seed,
        "threshold": args.threshold,
        "attention": "all_nodes",
        "samples": int(len(labels)),
        "timeseries_dim": int(ts.shape[1]),
        "image_dim": int(image.shape[1]),
        "image_coverage": int(image_mask.sum()),
        "test_samples": int(len(test_idx)),
        "test_f1": round(float(f1_score(labels[test_idx], predictions, zero_division=0)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(labels[test_idx], predictions)), 4),
        "test_auc": round(float(roc_auc_score(labels[test_idx], probabilities[test_idx])), 4),
    }
    result_path = root / "fusion_graph_transformer_results.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("test_f1=", metrics["test_f1"])
    print("test_balanced_accuracy=", metrics["test_balanced_accuracy"])
    print("test_auc=", metrics["test_auc"])
    print("results_saved=", result_path)


if __name__ == "__main__":
    main()
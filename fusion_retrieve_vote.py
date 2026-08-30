"""Retrieve-and-vote fusion baseline (no learned classifier).

Instead of a trained fusion head, each test/valid row is classified by
majority vote of its k nearest training neighbors, across 4 retrieval
streams:

  ts_to_ts    within-modal: timeseries query  -> timeseries corpus
  img_to_img  within-modal: image (CLIP) query -> image corpus
  ts_to_img   cross-modal:  timeseries query  -> image corpus
  img_to_ts   cross-modal:  image query       -> timeseries corpus

Timeseries (32d) and image (512d) live in unrelated spaces, so cross-modal
retrieval only makes sense in a space where the two are comparable. That
shared space is built with CCA fit on paired training rows; within-modal
streams keep using their native standardized space since that preserves the
most discriminative signal.

Each stream votes using the fraction of its k neighbors that are label 1;
the 4 stream votes are combined by majority voting, with ties (2-2) broken
by the mean soft score across streams. That mean soft score also doubles as
the probability used for AUC.

Requires faiss (`pip install faiss-cpu`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fusion_gat import load_data

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "faiss is required for this script. Install it with `pip install faiss-cpu`."
    ) from exc


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def knn_vote(index_vectors: np.ndarray, index_labels: np.ndarray,
             query_vectors: np.ndarray, k: int) -> np.ndarray:
    """Fraction of each query's k nearest (cosine, via FAISS inner product on
    L2-normalized vectors) training neighbors that are labeled positive."""
    index = faiss.IndexFlatIP(index_vectors.shape[1])
    index.add(l2_normalize(index_vectors))
    _, neighbor_ids = index.search(l2_normalize(query_vectors), k)
    return index_labels[neighbor_ids].mean(axis=1)


def fuse(scores: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    votes = np.stack([(s >= 0.5).astype(int) for s in scores.values()], axis=1)
    soft = np.stack(list(scores.values()), axis=1).mean(axis=1)
    majority = votes.sum(axis=1)
    n_streams = votes.shape[1]
    tie = majority * 2 == n_streams
    predictions = np.where(tie, (soft >= 0.5).astype(int), (majority > n_streams / 2).astype(int))
    return predictions, soft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--cca-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
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

    cca_dim = min(args.cca_dim, ts.shape[1], image.shape[1])
    cca = CCA(n_components=cca_dim, max_iter=2000)
    cca.fit(ts[train_idx], image[train_idx])
    ts_shared, image_shared = cca.transform(ts, image)
    ts_shared = ts_shared.astype(np.float32)
    image_shared = image_shared.astype(np.float32)

    streams = {
        "ts_to_ts": (ts, ts),
        "img_to_img": (image, image),
        "ts_to_img": (ts_shared, image_shared),
        "img_to_ts": (image_shared, ts_shared),
    }

    def stream_scores(eval_idx: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: knn_vote(index_space[train_idx], labels[train_idx], query_space[eval_idx], args.k)
            for name, (query_space, index_space) in streams.items()
        }

    valid_scores = stream_scores(valid_idx)
    valid_pred, _ = fuse(valid_scores)
    print(f"valid_f1={f1_score(labels[valid_idx], valid_pred, zero_division=0):.4f}")
    for name, s in valid_scores.items():
        stream_pred = (s >= 0.5).astype(int)
        print(f"  stream={name} valid_f1={f1_score(labels[valid_idx], stream_pred, zero_division=0):.4f}")

    test_scores = stream_scores(test_idx)
    test_pred, test_soft = fuse(test_scores)

    per_stream_test_f1 = {
        name: round(float(f1_score(labels[test_idx], (s >= 0.5).astype(int), zero_division=0)), 4)
        for name, s in test_scores.items()
    }

    metrics = {
        "model": "retrieve-and-vote-fusion",
        "k": args.k,
        "cca_dim": cca_dim,
        "seed": args.seed,
        "samples": int(len(labels)),
        "timeseries_dim": int(ts.shape[1]),
        "image_dim": int(image.shape[1]),
        "image_coverage": int(image_mask.sum()),
        "test_samples": int(len(test_idx)),
        "per_stream_test_f1": per_stream_test_f1,
        "test_f1": round(float(f1_score(labels[test_idx], test_pred, zero_division=0)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(labels[test_idx], test_pred)), 4),
        "test_auc": round(float(roc_auc_score(labels[test_idx], test_soft)), 4),
    }
    result_path = root / "fusion_retrieve_vote_results.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("test_f1=", metrics["test_f1"])
    print("test_balanced_accuracy=", metrics["test_balanced_accuracy"])
    print("test_auc=", metrics["test_auc"])
    print("results_saved=", result_path)


if __name__ == "__main__":
    main()

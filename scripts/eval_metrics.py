"""
Ham danh gia dung chung cho nhanh anh (imagefeat/evaluate_labels.py) va fusion
(fusion/baseline_ketqua.py) — mot noi dinh nghia, hai noi dung, khong bao gio lech.

  metrics            : ROC-AUC, PR-AUC, Accuracy, macro-F1, Sensitivity, Specificity
  cv_threshold_preds : nguong chon tren cac fold KHAC (toi da macro-F1), ap cho fold con lai
  patient_bootstrap  : KTC 95% lay mau co hoan lai theo BENH NHAN, khong theo dong
  paired_diff        : chenh lech hai mo hinh tren cung mau bootstrap

macro-F1 = trung binh F1 cua hai lop — dinh nghia F1 bai SiCLIP dung.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)

SEED = 42
N_BOOT = 1000


def metrics(y: np.ndarray, s: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    return {
        "ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
        "Accuracy": accuracy_score(y, p), "macro-F1": f1_score(y, p, average="macro"),
        "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
    }


def cv_threshold_preds(s: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Nhan du doan voi nguong chon tren cac fold khac (toi da macro-F1), ap cho fold con lai."""
    pred = np.zeros(len(y), dtype=int)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        grid = np.quantile(s[tr], np.linspace(0.02, 0.98, 97))
        best = max(grid, key=lambda t: f1_score(y[tr], s[tr] >= t, average="macro"))
        pred[va] = (s[va] >= best).astype(int)
    return pred


def _rows_by_patient(groups: np.ndarray) -> list[np.ndarray]:
    uniq, inv = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(len(uniq))]


def patient_bootstrap(y, s, p, groups, n: int = N_BOOT) -> dict:
    """KTC 95%: lay mau co hoan lai theo BENH NHAN (mot nguoi kham nhieu lan di cung nhau)."""
    rng = np.random.default_rng(SEED)
    rows_of = _rows_by_patient(groups)
    samples = {k: [] for k in metrics(y, s, p)}
    for _ in range(n):
        idx = np.concatenate([rows_of[g] for g in rng.integers(0, len(rows_of), len(rows_of))])
        if y[idx].min() == y[idx].max():
            continue
        for k, v in metrics(y[idx], s[idx], p[idx]).items():
            samples[k].append(v)
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in samples.items()}


def paired_diff(y, s_a, s_b, groups, fn, n: int = N_BOOT):
    """fn(a) - fn(b) tren cung mau bootstrap theo benh nhan -> (hieu, KTC duoi, KTC tren, P(a > b))."""
    rng = np.random.default_rng(SEED)
    rows_of = _rows_by_patient(groups)
    d = []
    for _ in range(n):
        idx = np.concatenate([rows_of[g] for g in rng.integers(0, len(rows_of), len(rows_of))])
        d.append(fn(y[idx], s_a[idx]) - fn(y[idx], s_b[idx]))
    d = np.array(d)
    return fn(y, s_a) - fn(y, s_b), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()

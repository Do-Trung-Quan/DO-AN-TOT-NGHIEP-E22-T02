"""
================================================================================
ĐÁNH GIÁ ĐẶC TRƯNG ẢNH X-QUANG: BỘ CONTROL VÀ BỘ FROZEN TRÊN 2 NHÃN
================================================================================
Mục đích:
  Đo lường năng lực chẩn đoán độc lập của 2 bộ đặc trưng ảnh (256-D):
    1. control : BioViL-T đã fine-tune trên SetA (ngoài luồng)
    2. frozen  : BioViL-T gốc của Microsoft (chưa fine-tune)

  Trên 2 bài toán nhãn:
    - bnn    : Bệnh nghề nghiệp (99 ca dương / 1.835 ảnh, tỷ lệ 5,4%)
    - ketqua : Tổn thương trên phim X-quang (462 ca dương / 1.835 ảnh, tỷ lệ 25,2%)

Phương pháp đánh giá:
  - 5-Fold Cross Validation phân tầng theo nhóm bệnh nhân (StratifiedGroupKFold theo patient_uid)
  - 0% rò rỉ bệnh nhân giữa tập huấn luyện và kiểm thử.
  - Linear Probe: StandardScaler + LogisticRegression (C=0.1, cân bằng lớp).
  - Dự đoán Out-of-fold (OOF) trên toàn bộ 1.835 bệnh nhân.
  - Khoảng tin cậy Bootstrap 95% theo bệnh nhân (1.000 lần lấy mẫu).

Cách chạy:
  python imagefeat/evaluate_labels.py --label bnn       # Đánh giá theo nhãn bnn
  python imagefeat/evaluate_labels.py --label ketqua    # Đánh giá theo nhãn ketqua
  python imagefeat/evaluate_labels.py --label all       # Đánh giá cả 2 nhãn để đối chiếu
================================================================================
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_metrics import cv_threshold_preds, metrics, patient_bootstrap  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "imagefeat" / "output"

# 2 bộ đặc trưng chuẩn (mỗi bộ là 1 mô hình duy nhất, cùng 1 không gian vector 256-D)
SOURCES = [
    ("control", "image_features_control.parquet", "BioViL-T fine-tune tren SetA (bnn)"),
    ("frozen",  "image_features_frozen.parquet",  "BioViL-T goc Microsoft (pretrained)"),
]

METRIC_COLS = ["ROC-AUC", "PR-AUC", "Accuracy", "macro-F1", "Sensitivity", "Specificity"]


def make_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=42)
    )


def evaluate_single_label(label: str, base: pd.DataFrame, feats: dict[str, np.ndarray]) -> list[str]:
    y = base[f"label_{label}"].to_numpy().astype(int)
    g = base["patient_uid"].to_numpy()
    
    # Chia 5-fold phân tầng theo nhãn, nhóm theo bệnh nhân (0% rò rỉ)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=int)
    for fold_k, (_, val_idx) in enumerate(sgkf.split(base, y, groups=g)):
        folds[val_idx] = fold_k

    for k in range(5):
        assert not (set(g[folds == k]) & set(g[folds != k])), f"Lỗi: Bệnh nhân bắc cầu qua fold {k}"

    lines = []
    def log(text=""):
        print(text)
        lines.append(text)

    log(f"================================================================================")
    log(f"BÁO CÁO ĐÁNH GIÁ ĐẶC TRƯNG ẢNH TRÊN NHÃN: [{label.upper()}]")
    log(f"================================================================================")
    log(f"- Tổng số mẫu có ảnh   : {len(y)} lượt khám ({len(set(g))} bệnh nhân)")
    log(f"- Số ca dương ({label:6s}= 1) : {y.sum()} ca")
    log(f"- Tỷ lệ dương tính (Prevalence): {y.mean():.2%} (Mốc ngẫu nhiên PR-AUC = {y.mean():.4f})")
    log(f"- Phương pháp đánh giá  : 5-Fold CV (StratifiedGroupKFold theo patient_uid, 0% rò rỉ)")
    log()

    # Bảng kết quả chính
    log("### BẢNG KẾT QUẢ CHÍNH (Out-of-fold kèm KTC 95% Bootstrap)")
    log()
    header = "| Bộ đặc trưng | Mô tả nguồn gốc | " + " | ".join(METRIC_COLS) + " | PR-AUC vs Ngẫu nhiên |"
    sep = "|---|---|" + "---|" * len(METRIC_COLS) + "---|"
    log(header)
    log(sep)

    for name, _, desc in SOURCES:
        X = feats[name]
        # Dự đoán out-of-fold (học 4 fold, dự đoán fold còn lại)
        s = np.full(len(y), np.nan)
        for k in range(5):
            tr, va = (folds != k), (folds == k)
            s[va] = make_probe().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        
        # Tối ưu ngưỡng cắt trên tập train
        p = cv_threshold_preds(s, y, folds)
        
        # Đo chỉ số và Bootstrap 95% CI
        m = metrics(y, s, p)
        ci = patient_bootstrap(y, s, p, g)
        
        ratio = m["PR-AUC"] / y.mean()
        row_str = f"| **{name}** | {desc} | "
        row_str += " | ".join(f"{m[c]:.3f} [{ci[c][0]:.3f}–{ci[c][1]:.3f}]" for c in METRIC_COLS)
        row_str += f" | **{ratio:.1f}x** |"
        log(row_str)

    log()
    log("### MỐC THAM CHIẾU (BASELINE)")
    log(f"- **Đoán ngẫu nhiên** : ROC-AUC = 0.500 | PR-AUC = {y.mean():.3f} (tỷ lệ nền)")
    
    # Đoán thuần túy theo tuổi
    roc_age = roc_auc_score(y, base["tuoi"])
    pr_age = average_precision_score(y, base["tuoi"])
    log(f"- **Chỉ dùng Tuổi**   : ROC-AUC = {roc_age:.3f} | PR-AUC = {pr_age:.3f} "
        f"(Nếu vector ảnh vượt mốc này tức là ảnh thực sự mang tín hiệu bệnh học)")
    log()

    out_file = OUT / f"eval_{label}.md"
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"-> Đã lưu báo cáo chi tiết vào: {out_file.relative_to(ROOT).as_posix()}\n")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Đánh giá đặc trưng ảnh X-quang trên nhãn bnn và ketqua")
    ap.add_argument("--label", default="all", choices=["bnn", "ketqua", "all"],
                    help="Nhãn mục tiêu cần đánh giá: 'bnn', 'ketqua', hoặc 'all' (mặc định)")
    args = ap.parse_args()

    # Nạp chỉ mục ảnh và tuổi
    index = pd.read_parquet(OUT / "image_index.parquet")
    age = pd.read_parquet(ROOT / "output" / "fusion_node_meta.parquet")[["id", "tuoi"]]
    base = index.merge(age, left_on="img_id", right_on="id", how="left")
    assert len(base) == 1835, f"Lệch số lượng ảnh: {len(base)}/1835"

    # Nạp 2 bộ vector ảnh
    feats = {}
    for name, fname, _ in SOURCES:
        f = pd.read_parquet(OUT / fname)
        cols = [c for c in f.columns if c.startswith("img_feat_")]
        X = base[["img_id"]].merge(f, on="img_id", how="left")[cols].to_numpy(np.float64)
        assert np.isfinite(X).all(), f"{name}: Chứa giá trị NaN hoặc Inf"
        feats[name] = X

    labels_to_run = ["bnn", "ketqua"] if args.label == "all" else [args.label]

    for lbl in labels_to_run:
        evaluate_single_label(lbl, base, feats)


if __name__ == "__main__":
    main()

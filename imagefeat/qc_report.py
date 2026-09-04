"""
================================================================================
P3 — CONG NGHIEM THU (ban mac dinh cho bo FROZEN)
[khong co bao cao nay thi KHONG nhan ban giao]
================================================================================
Sinh output/image_features_frozen_qc.md cho bo dac trung
imagefeat/output/image_features_frozen.parquet. Thoat ma 1 neu bat ky assert
cung nao truot.

5 ASSERT CUNG (truot = tu choi ban giao):
  1. So hang khop chi muc            = 1835/1835
  2. So chieu hang so (KIEM FLOAT64) = 0
  3. Benh nhan nam o >1 fold         = 0
  4. NaN/Inf trong hang co anh       = 0
  5. Ten file / ho ten trong artifact= 0

4 CHI SO BAO CAO (khong dat nguong, nhung phai co mat):
  6. Cosine TB truoc/sau mean-centering
  7. Effective rank + % phuong sai PC1
  8. PR-AUC probe out-of-fold tren nhan bnn
  9. Do chinh xac doan LO CHUP tu embedding

VI SAO PHAI KIEM BANG FLOAT64 (chi so 2):
  Kiem bang float32 chi lo ra 1 chieu chet. Phai dung float64 hoac ptp() moi
  thay du 256. Mot cai QC ho hoi se cho bo vector hong di qua.

VI SAO CO CHI SO 6:
  Nhan xet ban dau cho rang bo 256 chieu chet se chua duoc do thi kNN
  (cosine 0.789 -> 0.011). DO THUC TE: 512 chieu = 0.759, 256 chieu song =
  0.755 — GAN NHU KHONG DOI. Khoi chet chi chiem 1.5% norm^2. Cosine chi tut
  ve ~0.003 khi MEAN-CENTERING. Chi so nay bat buoc phai co de nguoi dung
  fusion biet ho PHAI center truoc khi dung do thi kNN.

CHAY (khong can tham so, da mac dinh vao bo frozen):
  python imagefeat/qc_report.py

Van co the doi sang bo khac neu can:
  python imagefeat/qc_report.py --features output/image_features_control.parquet --output output/image_features_control_qc.md
================================================================================
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXPECTED_ROWS = 1835
NAME_PATTERN = re.compile(r"[A-Za-z]{2,}[ _][A-Za-z]{2,}[ _][A-Za-z]{2,}")


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed: list[str] = []

    def gate(self, number: int, name: str, value, threshold, ok: bool) -> None:
        status = "DAT" if ok else "**TRUOT**"
        self.lines.append(f"| {number} | {name} | `{value}` | {threshold} | {status} |")
        print(f"  [{'OK ' if ok else 'FAIL'}] {number}. {name} = {value} (can {threshold})")
        if not ok:
            self.failed.append(f"{number}. {name} = {value}, can {threshold}")

    def note(self, number: int, name: str, value: str) -> None:
        self.lines.append(f"| {number} | {name} | `{value}` | bao cao | — |")
        print(f"  [    ] {number}. {name} = {value}")


def mean_cosine(matrix: np.ndarray, sample: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(matrix), size=min(sample, len(matrix)), replace=False)
    block = matrix[rows].astype(np.float64)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = block / norms
    similarity = unit @ unit.T
    upper = similarity[np.triu_indices(len(unit), k=1)]
    return float(upper.mean())


def effective_rank(matrix: np.ndarray) -> tuple[float, float]:
    centered = matrix - matrix.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular ** 2
    total = variance.sum()
    if total <= 0:
        return 0.0, 0.0
    proportion = variance / total
    positive = proportion[proportion > 0]
    return float(np.exp(-(positive * np.log(positive)).sum())), float(proportion[0] * 100)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=here / "output" / "image_index.parquet")
    # DA DOI MAC DINH -> tro thang vao bo FROZEN, khong can truyen tay
    parser.add_argument("--features", type=Path,
                        default=here / "output" / "image_features_frozen.parquet")
    parser.add_argument("--output", type=Path,
                        default=here / "output" / "image_features_frozen_qc.md")
    args = parser.parse_args()

    print("=" * 80)
    print("P3 — CONG NGHIEM THU (bo FROZEN)")
    print("=" * 80)

    index = pd.read_parquet(args.index)
    features_frame = pd.read_parquet(args.features)
    merged = index.merge(features_frame, on="img_id", how="inner")
    feature_columns = [c for c in features_frame.columns if c.startswith("img_feat_")]
    report = Report()

    # --- 1. so hang ---
    report.gate(1, "So hang khop chi muc", f"{len(merged)}/{EXPECTED_ROWS}",
                f"= {EXPECTED_ROWS}", len(merged) == EXPECTED_ROWS)

    has_image = merged["has_image"].to_numpy().astype(bool)
    matrix = merged.loc[has_image, feature_columns].to_numpy(dtype=np.float64)

    # --- 2. chieu hang so (FLOAT64) ---
    spread = matrix.max(0) - matrix.min(0)
    dead = int((spread == 0).sum())
    report.gate(2, "Chieu hang so (float64)", f"{dead}/{len(feature_columns)}", "= 0", dead == 0)

    # --- 3. benh nhan nam o >1 fold ---
    folded = merged[merged["fold_id"] >= 0]
    if len(folded):
        per_patient = folded.groupby("patient_group")["fold_id"].nunique()
        straddling = int((per_patient > 1).sum())
    else:
        straddling = 0
    report.gate(3, "Benh nhan nam o >1 fold", straddling, "= 0", straddling == 0)

    # --- 4. NaN/Inf ---
    bad = int((~np.isfinite(matrix)).sum())
    report.gate(4, "NaN/Inf trong hang co anh", bad, "= 0", bad == 0)

    # --- 5. PII ---
    leaked = 0
    for column in features_frame.columns:
        if features_frame[column].dtype == object:
            leaked += int(features_frame[column].astype(str).str.contains(
                NAME_PATTERN, regex=True, na=False).sum())
    for column in index.columns:
        if column in {"file_hash", "patient_group"}:
            continue
        if index[column].dtype == object:
            leaked += int(index[column].astype(str).str.contains(
                NAME_PATTERN, regex=True, na=False).sum())
    report.gate(5, "Ten file / ho ten trong artifact", leaked, "= 0", leaked == 0)

    # --- 6. cosine truoc/sau centering ---
    raw_cosine = mean_cosine(matrix)
    centered_cosine = mean_cosine(matrix - matrix.mean(0, keepdims=True))
    report.note(6, "Cosine TB (raw)", f"{raw_cosine:.4f}")
    report.note(6, "Cosine TB (da mean-center)", f"{centered_cosine:.4f}")

    # --- 7. effective rank + PC1 ---
    rank, pc1 = effective_rank(matrix)
    report.note(7, "Effective rank", f"{rank:.2f}/{len(feature_columns)}")
    report.note(7, "Phuong sai PC1", f"{pc1:.1f}%")

    # --- 8. probe out-of-fold tren nhan bnn ---
    y_bnn = merged.loc[has_image, "label_bnn"].to_numpy()
    groups = merged.loc[has_image, "patient_group"].to_numpy()
    baseline = float(y_bnn.mean())
    probe_line = "khong du du lieu"
    if len(np.unique(y_bnn)) > 1:
        probe = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        predicted = cross_val_predict(probe, matrix, y_bnn, cv=splitter,
                                      groups=groups, method="predict_proba")[:, 1]
        probe_line = f"{average_precision_score(y_bnn, predicted):.4f} (nen {baseline:.4f})"
    report.note(8, "PR-AUC probe tren nhan bnn", probe_line)

    # --- 9. doan lo chup tu embedding ---
    batches = merged.loc[has_image, "batch_date"].to_numpy()
    # StratifiedGroupKFold(5) doi moi lop co it nhat 5 mau. Du lieu that co lo
    # chi 5-6 anh (vd 20190827) -> phai loai truoc, neu khong se nem loi.
    counts = pd.Series(batches).value_counts()
    keep_batches = set(counts[counts >= 5 * 2].index)
    keep_mask = np.array([b in keep_batches for b in batches])
    batch_line = "khong du lo"
    if keep_mask.sum() > 50 and len(keep_batches) > 1:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            predicted = cross_val_predict(clf, matrix[keep_mask], batches[keep_mask],
                                          cv=splitter, groups=groups[keep_mask])
            accuracy = float((predicted == batches[keep_mask]).mean())
            chance = float(pd.Series(batches[keep_mask]).value_counts(normalize=True).max())
            batch_line = (f"{100 * accuracy:.1f}% (nen theo lop lon nhat {100 * chance:.1f}%, "
                          f"{len(keep_batches)} lo, {int(keep_mask.sum())} anh)")
        except ValueError as error:
            batch_line = f"khong tinh duoc: {error}"
    report.note(9, "Do chinh xac doan lo chup", batch_line)

    verdict = "DAT — chap nhan ban giao" if not report.failed else "TRUOT — TU CHOI ban giao"
    document = [
        "# Bao cao kiem dinh dac trung anh (QC) — bo FROZEN",
        "",
        f"- Sinh luc: {datetime.now(timezone.utc).isoformat()}",
        f"- Nguon dac trung: `{args.features.name}`",
        f"- Chi muc: `{args.index.name}`",
        f"- **Ket luan: {verdict}**",
        "",
        "| # | Chi tieu | Gia tri | Nguong | Ket qua |",
        "|---|---|---|---|---|",
        *report.lines,
        "",
    ]
    if report.failed:
        document += ["## Cac muc TRUOT", "", *[f"- {item}" for item in report.failed], ""]
    document += [
        "## Doc chi so 6 — quan trong cho phase fusion",
        "",
        f"Cosine trung binh raw = **{raw_cosine:.4f}**, sau mean-centering = **{centered_cosine:.4f}**.",
        "",
        "Embedding sau co mot thanh phan trung binh chung rat lon (anisotropy).",
        "Bo 256 chieu chet **khong** xu ly duoc dieu nay — do thuc te cho thay",
        "cosine 512 chieu (0.759) va 256 chieu song (0.755) gan nhu bang nhau.",
        "",
        "> **Bat buoc voi ben fusion:** phai mean-center hoac standardize truoc khi",
        "> dung do thi kNN. Neu khong, moi anh se 'giong nhau ~76%' va do thi",
        "> gan nhu vo nghia.",
        "",
        "## Doc chi so 9 — nhieu loan theo lo chup",
        "",
        "Ty le duong theo ngay chup lech 8 lan (20191115 = 5.5% ... 20181223 = 44.8%,",
        "ca biet 20181224 = 58.7%). Embedding ma hoa manh thong tin may chup / dot kham.",
        "Khi bao cao ket qua fusion nen tach metric theo lo.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(document), encoding="utf-8")

    print("-" * 80)
    print(f"Ket luan: {verdict}")
    print(f"Da ghi -> {args.output}")
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

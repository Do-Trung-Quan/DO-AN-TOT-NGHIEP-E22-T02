"""
================================================================================
BASELINE 2 NHANH VOI NHAN `ketqua` + UOC LUONG FUSION, DAT CANH BAI SiCLIP
================================================================================
Nhan `ketqua` = ket qua doc phim X-quang cua chinh dot kham theo ILO (1 = bui phoi
silic). Day la nhan bai SiCLIP (Artif. Intell. Med. 2026) dung — khac `bnn`, von la
TIEN SU da duoc cong nhan benh nghe nghiep.

DAU VAO
  output/ketqua/ts_predictions.parquet   <- chay DO_AN.ipynb voi TARGET = "ketqua"
  imagefeat/output/image_features_control.parquet   (control — mot khong gian vector)
  imagefeat/output/image_features_frozen.parquet     (frozen  — mot khong gian vector)
  imagefeat/output/image_index.parquet               (label_ketqua, patient_uid)

CAC MO HINH DUOC DANH GIA
  TS       : mo hinh Keras cua nhanh timeseries, train tren 8030 nguoi voi nhan ketqua.
             Diem so ngoai mau: dev = OOF 5 fold, test = ensemble 5 fold.
  ANH      : probe tuyen tinh tren vector control, CV 5 fold nhom theo benh nhan —
             CUNG cach chia voi imagefeat/evaluate_labels.py nen hai bao cao trung so.
  FUSION   : late fusion — hoi quy logistic tren [logit(TS), logit(ANH control)], CV cung fold.
             Day la MOC DUOI cua fusion: GNN / CLIP chi dang lam neu vuot duoc no.

GIAO THUC DANH GIA (giong bai bao o nhung diem bai bao cong bo)
  - Chi tren 1835 nguoi co anh (bai bao: 1633 nguoi, deu co anh).
  - Gop du doan ngoai mau cua ca 5 fold roi tinh chi so; KTC 95% bootstrap theo BENH NHAN.
  - F1 = macro-F1 (trung binh F1 hai lop), dung dinh nghia cua bai bao.
  - Nguong phan loai duoc CHON TREN 4 FOLD, AP DUNG CHO FOLD CON LAI — khong chon
    tren chinh du lieu danh gia. Bai bao khong noi ho chon nguong the nao.

CHAY
  python fusion/baseline_ketqua.py
================================================================================
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_metrics import (N_BOOT, cv_threshold_preds, metrics,  # noqa: E402
                                  paired_diff, patient_bootstrap)
from imagefeat.evaluate_labels import make_probe  # noqa: E402  (cung cau hinh probe voi nhanh anh)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
TS_PRED = ROOT / "output" / "ketqua" / "ts_predictions.parquet"
IMG_DIR = ROOT / "imagefeat" / "output"
REPORT = ROOT / "fusion" / "output" / "baseline_ketqua_report.md"

# Bai SiCLIP, Table 5 + Table 6. F1 = macro-F1. Ty le duong 462/1633 = 0,283.
PAPER = [
    ("DenseNet201 (chi anh)",       0.96, 0.85, 0.70, 0.78, 0.87),
    ("ConvNeXtLarge (chi anh)",     0.98, 0.88, 0.72, 0.83, 0.89),
    ("Qwen2-VL (chi text)",         0.94, 0.78, 0.58, 0.71, 0.75),
    ("Qwen2-VL (anh + text)",       0.96, 0.86, 0.70, 0.81, 0.87),
    ("SiCLIP k=3 (anh + text)",     0.98, 0.94, 0.85, 0.91, 0.93),
]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pooled_scores(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Probe anh: hoc tren 4 fold, du doan fold con lai — cung ham probe voi nhanh anh."""
    s = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        s[va] = make_probe().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    assert np.isfinite(s).all()
    return s


def cv_scores(X: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float) -> np.ndarray:
    """Diem so ngoai mau: fold k duoc du doan boi mo hinh fit tren cac fold con lai."""
    out = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        out[va] = mdl.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    assert np.isfinite(out).all()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts-pred", type=Path, default=TS_PRED)
    ts_pred = parser.parse_args().ts_pred
    if not ts_pred.exists():
        raise SystemExit(f"Chua co {ts_pred} — chay DO_AN.ipynb voi TARGET = \"ketqua\" truoc.")

    ts = pd.read_parquet(ts_pred)
    idx = pd.read_parquet(IMG_DIR / "image_index.parquet")[["img_id", "patient_uid", "label_ketqua"]]
    ct = pd.read_parquet(IMG_DIR / "image_features_control.parquet")
    fz = pd.read_parquet(IMG_DIR / "image_features_frozen.parquet")
    feat_cols = [c for c in ct.columns if c.startswith("img_feat_")]

    # ---- Ghep 1835 nguoi co anh, kiem tra lien thong ----
    d = (idx.merge(ct[["img_id"] + feat_cols], on="img_id")
            .merge(ts[["id", "label", "score", "split", "patient_uid", "fold_id"]], left_on="img_id",
                   right_on="id", suffixes=("", "_ts")))
    assert len(d) == 1835, f"Ghep that bai: {len(d)}/1835"
    assert (d["label"] == d["label_ketqua"]).all(), "Nhan ketqua lech giua 2 nhanh"
    assert (d["patient_uid"] == d["patient_uid_ts"]).all(), "patient_uid lech giua 2 nhanh"
    fz = fz.set_index("img_id").loc[d["img_id"], feat_cols].to_numpy(np.float64)

    y = d["label"].to_numpy().astype(int)
    g = d["patient_uid"].to_numpy()
    # Chia 5 fold ngay tai day, nhom theo benh nhan — giong imagefeat/evaluate_labels.py
    folds = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42)
                                .split(d, y, groups=g)):
        folds[va] = k
    for k in range(5):
        assert not (set(g[folds == k]) & set(g[folds != k])), f"Benh nhan bac cau fold {k}"

    # ---- Diem so ngoai mau cua tung mo hinh tren 1835 nguoi ----
    s_ts = d["score"].to_numpy()
    s_img = pooled_scores(d[feat_cols].to_numpy(np.float64), y, folds)
    s_fz = pooled_scores(fz, y, folds)
    s_fus = cv_scores(np.column_stack([logit(s_ts), logit(s_img)]), y, folds, C=1.0)

    models = [("TS — Keras nhanh timeseries", s_ts),
              ("ANH — control (probe)", s_img),
              ("ANH — frozen (probe, doi chung)", s_fz),
              ("FUSION muon — TS + ANH control", s_fus)]

    L = []   # dong markdown
    def out(line=""):
        print(line); L.append(line)

    out(f"# Baseline nhan `ketqua` — hai nhanh va uoc luong fusion")
    out()
    out(f"- Sinh luc: {datetime.now(timezone.utc).isoformat()}")
    out(f"- Tap danh gia: **{len(y)} nguoi co anh**, {y.sum()} ca bui phoi, ty le {y.mean():.3f} "
        f"(bai bao: 1633 nguoi, 462 ca, ty le 0,283)")
    out(f"- Du doan ngoai mau gop tu 5 fold; KTC 95% bootstrap theo benh nhan ({N_BOOT} lan); "
        f"nguong chon tren 4 fold, ap cho fold con lai")
    out(f"- Nhanh TS: `{ts_pred.relative_to(ROOT).as_posix()}` — train tren {len(ts)} nguoi")
    out()

    # ---- Bang 1: do an ----
    out("## 1. Ket qua do an (1835 nguoi co anh)")
    out()
    cols = ["ROC-AUC", "PR-AUC", "Accuracy", "macro-F1", "Sensitivity", "Specificity"]
    out("| Mo hinh | " + " | ".join(cols) + " |")
    out("|---|" + "---|" * len(cols))
    for name, s in models:
        p = cv_threshold_preds(s, y, folds)
        m, ci = metrics(y, s, p), patient_bootstrap(y, s, p, g)
        out(f"| {name} | " + " | ".join(f"{m[c]:.3f} [{ci[c][0]:.3f}–{ci[c][1]:.3f}]" for c in cols) + " |")
    out()

    # ---- Bang 2: bai bao ----
    out("## 2. So bai bao cong bo (Table 5 + 6, khong co PR-AUC)")
    out()
    out("| Mo hinh | ROC-AUC | Accuracy | macro-F1 | Sensitivity | Specificity |")
    out("|---|---|---|---|---|---|")
    for name, auc, acc, f1, se, sp in PAPER:
        out(f"| {name} | {auc:.2f} | {acc:.2f} | {f1:.2f} | {se:.2f} | {sp:.2f} |")
    out()

    # ---- Bang 3: fusion co them duoc gi ----
    out("## 3. Fusion co them duoc gi so voi tung nhanh")
    out()
    out("| So sanh | Chi so | Chenh lech | KTC 95% | P(fusion tot hon) |")
    out("|---|---|---|---|---|")
    for ref_name, ref in [("ANH control", s_img), ("TS", s_ts)]:
        for mname, fn in [("ROC-AUC", roc_auc_score), ("PR-AUC", average_precision_score)]:
            diff, lo, hi, pr = paired_diff(y, s_fus, ref, g, fn)
            out(f"| FUSION − {ref_name} | {mname} | {diff:+.3f} | [{lo:+.3f}; {hi:+.3f}] | {pr:.0%} |")
    out()

    # ---- Bang 4: nhanh TS theo tung tap (chi cac tap co du lieu) ----
    out(f"## 4. Nhanh TS theo tung tap (cohort train: {len(ts)} nguoi)")
    out()
    out("| Tap | n | Ca duong | Ty le | ROC-AUC | PR-AUC |")
    out("|---|---|---|---|---|---|")
    for tag, mask in [("dev (OOF)", ts.split == "dev"), ("test (hold-out)", ts.split == "test"),
                      ("co anh (dev+test)", ts.has_image == 1), ("KHONG anh (dev+test)", ts.has_image == 0)]:
        yy, ss = ts.loc[mask, "label"].to_numpy(), ts.loc[mask, "score"].to_numpy()
        if len(yy) == 0 or yy.min() == yy.max():
            continue
        out(f"| {tag} | {mask.sum()} | {yy.sum()} | {yy.mean():.3f} | "
            f"{roc_auc_score(yy, ss):.3f} | {average_precision_score(yy, ss):.3f} |")
    out()
    out("## Luu y khi doc")
    out()
    out("- Bai bao loai 202 ca am (1373 -> 1171) theo tieu chi o Phu luc bang 2 (chua co); "
        "do an giu nguyen 1835 phim cua 1748 nguoi.")
    out("- ANH la probe tuyen tinh tren vector control. Chi tiet hai bo dac trung anh: "
        "imagefeat/output/eval_ketqua.md")
    out("- FUSION muon chi la moc duoi — cach ket hop don gian nhat co the.")
    out("- Accuracy, macro-F1 phu thuoc ty le duong: bai bao 0,283 vs do an "
        f"{y.mean():.3f}. ROC-AUC khong phu thuoc ty le, la chi so so sanh cong bang nhat.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nDa ghi -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

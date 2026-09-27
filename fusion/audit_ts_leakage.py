"""
================================================================================
KIEM TOAN: dac trung timeseries co "biet nhan" o tap dev khong, va bao nhieu?
================================================================================
BOI CANH
--------
`DO_AN.ipynb` cell 12 trich vector ts 32 chieu bang cach lay TRUNG BINH du doan
cua CA 5 model fold tren TOAN BO 8030 node. Voi mot node thuoc tap dev o fold k,
thi 4/5 model trong ensemble da nhin thay nhan cua chinh no luc huan luyen
-> vector cua node do mang thong tin nhan cua chinh no (label-aware).

Node tap test thi sach: split chia theo nhom benh nhan tu truoc, khong model nao
trong 5 fold duoc nhin thay chung.

=> Neu do CUNG MOT chi so tren dev va tren test, phan chenh lech chinh la
   uoc luong do THOI PHONG do ro ri. Day la mot phep do co doi chung tu nhien,
   khong phai phong doan.

BA PHEP DO
----------
  1. Probe tuyen tinh tren ts, CV 5-fold theo nhom benh nhan, RIENG tung tap.
     Do bao nhieu tin hieu nhan nam san trong 32 chieu ts.
  2. Homophily lop duong tren kNN dung RIENG trong tung tap, da ha mau dev ve
     dung co test. Do bao nhieu phan cua cau truc do thi Phase 2 den tu ro ri.
  3. Ty le khoang cach noi-lop / ngoai-lop (separation ratio).
     Cang nho thi lop duong cang tu thanh cum chat -> cang dang ngo.

LUU Y VE CONG BANG PHEP SO SANH
-------------------------------
dev (n=6882, 180 duong) va test (n=1148, 31 duong) khac co mau. Mau nho lam
phuong sai lon, KHONG lam lech ky vong. De khong so mot uoc luong on dinh voi
mot uoc luong nhieu, phep do 1 kem bootstrap KTC 95% cho ca hai tap; ket luan
chi duoc rut khi hai khoang KHONG chong nhau nhieu.

CHAY:
    python fusion/audit_ts_leakage.py
================================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
N_BOOT = 1000


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """PR-AUC/ROC-AUC out-of-fold cua probe tuyen tinh, nhom theo benh nhan.

    Tra ve diem OOF tren chinh tap dua vao — KHONG dung de du bao, chi dung de
    do luong tin hieu nhan da nam san trong dac trung.
    """
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, va in cv.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
    assert np.isfinite(oof).all()
    return oof


def boot_ci(y: np.ndarray, s: np.ndarray, fn, n: int = N_BOOT):
    """KTC 95% bootstrap phan vi. Bo qua mau khong co du 2 lop."""
    rng = np.random.default_rng(SEED)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].sum() < 2 or y[b].sum() == len(b):
            continue
        vals.append(fn(y[b], s[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def knn_positive_homophily(X: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Dung kNN cosine RIENG trong tap duoc dua vao, roi do P(hang xom duong | duong).

    KHONG dung do thi Phase 2 va loc canh noi bo: lam vay thi do thi cam sinh
    trong test chi con bac ~2,3 trong khi trong dev la ~13 — hai che do mat do
    hoan toan khac nhau, khong the so truc tiep. Dung lai kNN trong tung tap thi
    ca hai deu co dung k hang xom.
    """
    Xs = StandardScaler().fit_transform(X)
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    pos = np.flatnonzero(y == 1)
    return float((y[nn[pos]] == 1).mean())


def separation(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_pair: int = 200_000):
    """Khoang cach cosine TB trong lop duong / giua duong voi am.

    Ty le < 1 nghia la cac ca benh nam gan nhau hon so voi gan ca khong benh.
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    a, b = rng.choice(pos, n_pair), rng.choice(pos, n_pair)
    ok = a != b
    d_in = float(1 - (Xn[a[ok]] * Xn[b[ok]]).sum(1).mean())
    a, b = rng.choice(pos, n_pair), rng.choice(neg, n_pair)
    d_out = float(1 - (Xn[a] * Xn[b]).sum(1).mean())
    return d_in, d_out, d_in / d_out


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=here / "output" / "fusion_dataset_control.npz")
    args = p.parse_args()

    d = np.load(args.dataset, allow_pickle=False)
    X_ts, y = d["X_ts"].astype(np.float64), d["y"].astype(int)
    uid, split = d["patient_uid"], d["split"]

    # `X_ts` giong nhau o ca 3 nguon anh (chi nhanh anh khac) -> chay 1 lan la du.
    masks = {"dev": split == "dev", "test": split == "test"}

    print("=" * 78)
    print("KIEM TOAN RO RI DAC TRUNG TIMESERIES")
    print("=" * 78)
    print(f"Bo du lieu: {args.dataset.name}  (X_ts giong nhau o ca 3 nguon anh)")
    print()

    print("PHEP DO 1 — Probe tuyen tinh tren ts, CV 5-fold nhom benh nhan, RIENG tung tap")
    print("  (dev: dac trung label-aware | test: dac trung sach)")
    res1 = {}
    for name, m in masks.items():
        s = probe(X_ts[m], y[m], uid[m])
        yy = y[m]
        ap, auc = average_precision_score(yy, s), roc_auc_score(yy, s)
        lo, hi = boot_ci(yy, s, average_precision_score)
        base = yy.mean()
        res1[name] = (ap, lo, hi, auc, base)
        print(f"  {name:5s} n={m.sum():5d} duong={yy.sum():4d} | ROC-AUC {auc:.4f} | "
              f"PR-AUC {ap:.4f} [KTC95 {lo:.4f}–{hi:.4f}] (nen {base:.4f}, lift {ap/base:.1f}x)")
    infl = res1["dev"][0] / res1["test"][0] - 1
    print(f"  -> Thoi phong PR-AUC (dev so voi test): {infl:+.1%}")
    print(f"  -> KTC95 dev  [{res1['dev'][1]:.4f}–{res1['dev'][2]:.4f}] "
          f"vs test [{res1['test'][1]:.4f}–{res1['test'][2]:.4f}] "
          f"-> {'CHONG NHAU' if res1['dev'][1] < res1['test'][2] else 'TACH ROI'}")
    print()

    print("PHEP DO 2 — Homophily lop duong tren kNN dung RIENG trong tung tap (k=10)")
    ph_test = knn_positive_homophily(X_ts[masks["test"]], y[masks["test"]])
    n_test = int(masks["test"].sum())
    print(f"  test  n={n_test:5d} duong={int(y[masks['test']].sum()):4d} | "
          f"P(hang xom duong | node duong) = {ph_test:.4f} | lift {ph_test/y[masks['test']].mean():.1f}x")

    # Ha mau dev xuong DUNG co test: tap lon hon thi 10 hang xom gan nhat tu nhien
    # gan hon, day homophily len — khong khu thi dang so tao voi tao va cam.
    # Ha mau theo NHOM BENH NHAN de khong tach doi mot nguoi.
    dev_idx = np.flatnonzero(masks["dev"])
    uniq = np.unique(uid[dev_idx])
    subs = []
    for rep in range(10):
        r = np.random.default_rng(SEED + rep)
        take = set(r.permutation(uniq)[: int(len(uniq) * n_test / len(dev_idx)) + 5].tolist())
        sub = dev_idx[[u in take for u in uid[dev_idx]]]
        if y[sub].sum() < 5:
            continue
        subs.append(knn_positive_homophily(X_ts[sub], y[sub]))
    ph_dev = float(np.mean(subs))
    print(f"  dev   ha mau ve n~{n_test} x {len(subs)} lan | "
          f"P(hang xom duong | node duong) = {ph_dev:.4f} "
          f"(min {min(subs):.4f} max {max(subs):.4f}) | lift {ph_dev/y[masks['dev']].mean():.1f}x")
    print(f"  -> Thoi phong homophily duong (dev so voi test): {ph_dev/ph_test-1:+.1%}")
    print()

    print("PHEP DO 3 — Ty le khoang cach noi-lop / ngoai-lop tren ts")
    rng = np.random.default_rng(SEED)
    for name, m in masks.items():
        d_in, d_out, ratio = separation(X_ts[m], y[m], rng)
        print(f"  {name:5s} noi-lop duong {d_in:.4f} | duong-am {d_out:.4f} | ty le {ratio:.4f}")
    print()
    print("=" * 78)
    print("Chi so cang gan nhau giua dev va test thi ro ri cang it anh huong.")
    print("=" * 78)


if __name__ == "__main__":
    main()

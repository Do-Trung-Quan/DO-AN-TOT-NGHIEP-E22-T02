"""Kiem goi app PG (chay tren may hoc, can du lieu goc). Chi in so lieu gop, KHONG dung nhan test.

  python kiem_tra.py [--n-anh 30] [--bo-anh] [--ten PG|PG-noctx]

 a) lam sang tho (clin_raw, chuan_bi tinh lai bmi / fev1_fvc / cap nghe) + dac trung anh da luu -> diem app
    so voi diem da luu run_PG_nghe10.npz: test_cal (hang test) va oof_cal theo fold (hang dev fold k, chi_fold=k)
 b) goi tung luot bang dict (giong backend gui len) tren mau hang -> so voi (a)
 c) XuLyAnh tren n anh (cung thu tu file nhu v3_image.py) -> so voi regf_biovil_V2 / area trong img_v3.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent
PAG = DIR.parent
sys.path.insert(0, str(DIR)); sys.path.insert(0, str(PAG / "scripts_v3"))
from v3_base import load_base_v3, CACHE3, IMG3  # noqa: E402
from v3_lib import MERG, BASE  # noqa: E402
from du_doan_pg import MoHinhPG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-anh", type=int, default=30)
ap.add_argument("--bo-anh", action="store_true")
ap.add_argument("--n-dict", type=int, default=200)
ap.add_argument("--ten", default="PG", choices=["PG", "PG-noctx"], help="PG-noctx: ban khong dung tinh (mo_hinh_PG_noctx_web)")
a = ap.parse_args()

t0 = time.time()
B = load_base_v3()
hi, fold, dev, te = B["has_img"], B["fold"], B["dev"], B["test"]
OLD = np.load(CACHE3 / f"run_{a.ten}_nghe10.npz")
mh = MoHinhPG(CACHE3 / ("mo_hinh_PG_web" if a.ten == "PG" else "mo_hinh_PG_noctx_web"))
print(f"[kiem] nap xong {time.time() - t0:.0f}s, thiet bi {mh.dev}")
DERIVED = ["bmi", "fev1_fvc", "cviec_lv", "pxuong_lv", "cviec1_lv", "cviec2_lv"]


def lech(x, y, m, ten):
    d = np.abs(x[m] - y[m])
    print(f"  {ten:<34} n={int(m.sum()):5d}  lech TB {d.mean():.2e}  max {d.max():.2e}")
    return d.max()


# --- bien tinh lai
raw = B["raw"]
C = mh.chuan_bi(raw.drop(columns=DERIVED))
print("[kiem] bien tinh lai so voi B['raw']:")
for c in DERIVED:
    x, y = C[c], raw[c]
    if c.endswith("_lv"):
        ok = ((x == y) | (x.isna() & y.isna())).mean()
    else:
        x, y = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        ok = (((x - y).abs() < 1e-9) | (x.isna() & y.isna())).mean()
    print(f"  {c:<10} khop {ok:.4%}")

# --- a) theo lo
print("[kiem] a) theo lo:")
K = mh.diem(raw.drop(columns=DERIVED), B["reg_a"], B["area"], hi)
M = [lech(K["p"], OLD["test_cal"], te & g, f"test_cal ({n})") for g, n in ((hi, "co phim"), (~hi, "khong phim"))]
for k in range(5):
    Kk = mh.diem(raw.drop(columns=DERIVED), B["reg_a"], B["area"], hi, chi_fold=k)
    for g, n in ((hi, "co phim"), (~hi, "khong phim")):
        M.append(lech(Kk["p"], OLD["oof_cal"], dev & (fold == k) & g, f"oof_cal fold {k} ({n})"))

# --- b) tung luot bang dict
print("[kiem] b) tung luot bang dict:")
rng = np.random.default_rng(0)
idx = rng.choice(len(raw), min(a.n_dict, len(raw)), replace=False)
cols = [c for c in raw.columns if c not in DERIVED]
d1 = []
for i in idx:
    rec = {c: (None if pd.isna(v) else v) for c, v in raw.iloc[i][cols].items()}
    r = mh.diem(pd.DataFrame([rec], dtype=object), B["reg_a"][i:i + 1], B["area"][i:i + 1], hi[i:i + 1])
    d1.append(abs(r["p"][0] - K["p"][i]))
print(f"  dict vs lo  n={len(idx)}  (co phim {int(hi[idx].sum())})  lech max {max(d1):.2e}")
M.append(max(d1))
r = mh.du_doan({c: (None if pd.isna(v) else v) for c, v in raw.iloc[int(np.where(~hi)[0][0])][cols].items()})
print(f"  du_doan() tra khoa: {sorted(r)}")
if mh.dung_tinh:
    try:
        mh.du_doan({"tinh": "Ha Noi"}); print("  LOI: tinh la khong bi chan")
    except ValueError:
        print("  tinh ngoai 5 tinh -> ValueError (dung)")
else:
    r1 = mh.du_doan({"gioitinh": "Nam", "namsinh": 1975})
    r2 = mh.du_doan({"gioitinh": "Nam", "namsinh": 1975, "tinh": "Ha Noi"})   # tinh bi bo qua
    d = abs(r1["xac_suat"] - r2["xac_suat"])
    print(f"  ban khong tinh: gui kem tinh la hay khong gui deu khong anh huong: lech {d:.2e} ({'DUNG' if d == 0 else 'LOI'})")
    M.append(d)

# --- c) xu ly anh
if not a.bo_anh:
    from PIL import Image
    from xu_ly_anh import XuLyAnh
    files = []
    goc = MERG if (MERG / "train").exists() else BASE / "Data DGCNN" / "New Merged DATA"   # anh goc nam o ban Data DGCNN
    for root in (goc / "train", goc.parent / "NEW DATA" / "test"):
        for sub in ("normal", "silicosis"):
            for f in sorted(p.name for p in (root / sub).iterdir()):
                if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png"):
                    files.append(root / sub / f)
    assert len(files) == 1835
    Z = np.load(IMG3)
    pick = np.unique(np.linspace(0, len(files) - 1, a.n_anh).round().astype(int))
    xa = XuLyAnh(BASE / "Data DGCNN" / "Pre-train BioViL-T" / "biovil_t_image_model_proj_size_128.pt", thiet_bi=str(mh.dev))
    modes = pd.Series([Image.open(files[i]).mode for i in pick]).value_counts().to_dict()
    dr, da, cs = [], [], []
    t1 = time.time()
    for i in pick:
        o = xa(files[i]); R = Z["regf_biovil_V2"][i].astype(np.float32)
        dr.append(np.abs(o["regf"] - R).max()); da.append(np.abs(o["area"] - Z["area"][i]).max())
        nz = (np.linalg.norm(R, axis=1) > 0) & (np.linalg.norm(o["regf"], axis=1) > 0)
        cs.append(float(np.mean(np.sum(o["regf"][nz] * R[nz], 1) / np.linalg.norm(o["regf"][nz], axis=1) / np.linalg.norm(R[nz], axis=1))) if nz.any() else np.nan)
    dr, da = np.array(dr), np.array(da)
    print(f"[kiem] c) anh n={len(pick)} (che do anh {modes}), {(time.time() - t1) / len(pick):.2f} s/anh")
    print(f"  regf lech max: trung vi {np.median(dr):.2e}  max {dr.max():.2e}; so anh lech > 1e-2: {int((dr > 1e-2).sum())}")
    print(f"  area lech max: trung vi {np.median(da):.2e}  max {da.max():.2e}; cosine TB thap nhat {np.nanmin(cs):.6f}")

print(f"[kiem] lech diem lon nhat (a+b): {max(M):.2e} -> {'DAT' if max(M) < 1e-5 else 'KHONG DAT'} | {time.time() - t0:.0f}s")

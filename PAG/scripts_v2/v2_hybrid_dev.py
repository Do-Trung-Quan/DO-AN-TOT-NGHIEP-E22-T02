"""Huong (b) tren dev OOF: PG cho luot co anh + mo hinh lam sang (L0/X0/X2) cho luot khong anh. Chi in so lieu gop."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pag_lib import *
B = load_base(); y, hi, uid, dev = B["y"], B["has_img"], B["uid"], B["dev"]
BL = np.load(CACHE / "run_baselines.npz")
def pred(n): return BL[f"oof_cal_{n}"] if f"oof_cal_{n}" in BL.files else np.load(CACHE / f"run_{n}.npz")["oof_cal"]
def sens_at(yy, pp, s): t = np.quantile(pp[yy == 0], s); return float((pp[yy == 1] > t).mean())
pg = pred("PG"); rows = []
combos = {"PG (một mô hình)": pg, "X2 (một mô hình)": pred("X2")}
for c in ("L0", "X0", "X2"): combos[f"PG có ảnh + {c} không ảnh"] = np.where(hi, pg, pred(c))
for name, p in combos.items():
    m = dev & np.isfinite(p); yy, pp = y[m], p[m]
    a, lo, h, _ = boot_ci(lambda i: roc_auc_score(yy[i], pp[i]), uid[m])
    r = metrics(yy, pp)
    rows.append(dict(cach=name, auc=round(a, 3), ci=f"[{lo:.3f}-{h:.3f}]", pr_auc=round(r["pr_auc"], 3), brier=round(r["brier"], 4),
                     ece=round(r["ece"], 3), nhay_dh80=round(sens_at(yy, pp, .8), 3), nhay_dh90=round(sens_at(yy, pp, .9), 3)))
print(pd.DataFrame(rows).to_string(index=False))

"""Viec (3) (PGS 27/10): giai thich chenh LOBO 0,742 (PG_update 17/10) so voi 0,764 (P0.1 ban 3, hang "Cu").
Tai lap tren dac trung anh CU (khong dung anh ban 3), LR C=0.1 nhu P0.1, chi dev co anh; test KHONG mo.
Luoi 2 x 2 x 2: dac trung (frozen / control) x tap dev (goc: split/fold/uid file fold; moi: dev2/fold2/uid2)
x dinh nghia "de nguyen lo" (18 lan LOBO that / GroupKFold(5) theo lo) x cach tinh AUC (gop / trong tung lo). Kem Δ(lo + anh so voi lo) de doi chieu +0,078.
Chi ghi so lieu gop.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from v3_lib import *
from v3_lib import CACHE3, run_config

core = load_core()
P5 = np.load(CACHE3 / "p05_folds.npz", allow_pickle=True)
assert (P5["id"] == core.id.to_numpy()).all()
y, batch, hi = core.y.to_numpy(), core.batch.to_numpy(), core.has_img.to_numpy()
FEAT = {"frozen": np.stack(core.X_frozen), "control": np.stack(core.X_control)}
DEV = {"goc": ((core.split == "dev").to_numpy(), core.fold.to_numpy(), core.uid.to_numpy()),
       "moi (dev2/fold2)": (P5["dev2"].astype(bool), P5["fold2"].astype(int), pd.factorize(P5["uid2"])[0])}
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(batch[hi].reshape(-1, 1))
Bx = np.zeros((len(core), len(oh.categories_[0])), np.float32); Bx[hi] = oh.transform(batch[hi].reshape(-1, 1))

res = []
for dn, (dev, fold, uid) in DEV.items():
    m = dev & hi; yi, ui, bi = y[m], uid[m], batch[m]
    for fn, X in FEAT.items():
        pb = oof_cv(Bx, y, fold, m)[m]; pbi = oof_cv(np.column_stack([Bx, X]), y, fold, m)[m]
        d, dlo, dhi, _ = boot_ci(lambda i: roc_auc_score(yi[i], pbi[i]) - roc_auc_score(yi[i], pb[i]), ui)
        p18, p5 = np.full(m.sum(), np.nan), np.full(m.sum(), np.nan)
        for b in np.unique(bi):                                       # 18 lan, moi lan de nguyen 1 lo
            tr, va = bi != b, bi == b
            p18[va] = lr().fit(X[m][tr], yi[tr]).predict_proba(X[m][va])[:, 1]
        for tr, va in GroupKFold(5).split(X[m], yi, bi):              # 5 fold, moi lo nam tron 1 fold (cach v2)
            p5[va] = lr().fit(X[m][tr], yi[tr]).predict_proba(X[m][va])[:, 1]
        a18, l18, h18, _ = boot_ci(lambda i: roc_auc_score(yi[i], p18[i]), ui)
        a5, l5, h5, _ = boot_ci(lambda i: roc_auc_score(yi[i], p5[i]), ui)
        w18 = within_batch_auc(yi, p18, bi); w5 = within_batch_auc(yi, p5, bi)   # AUC trong tung lo (ham v2), tren cung du doan
        res.append(dict(tap_dev=dn, dac_trung=fn, n=int(m.sum()), duong=int(yi.sum()),
                        delta=d, delta_ci=f"[{dlo:+.3f}; {dhi:+.3f}]",
                        lobo_18_lan=a18, lobo_18_ci=f"[{l18:.3f}; {h18:.3f}]",
                        groupkfold5_theo_lo=a5, groupkfold5_ci=f"[{l5:.3f}; {h5:.3f}]",
                        lobo_18_trong_lo=w18, groupkfold5_trong_lo=w5))
        r = res[-1]
        print(f"dev {dn} | {fn} | n {r['n']} | delta {d:+.3f} {r['delta_ci']} | LOBO 18 lan {a18:.3f} {r['lobo_18_ci']} "
              f"| GroupKFold5 theo lo {a5:.3f} {r['groupkfold5_ci']} "
              f"| trong lo: LOBO18 {w18:.4f} GKF5 {w5:.4f}", flush=True)

R = pd.DataFrame(res)
out = RES3 / "tuan1_phase0/viec3_LOBO"; out.mkdir(parents=True, exist_ok=True)
R.round(4).to_csv(out / "lobo_luoi.csv", index=False, encoding="utf-8-sig")
run_config("tuan1_phase0/viec3_LOBO", __file__, data=[CACHE3 / "p05_folds.npz"], mo_hinh="LR C=0.1 (lr() v2_lib)",
           luoi=dict(dac_trung=list(FEAT), tap_dev=list(DEV), dinh_nghia=["LOBO 18 lan", "GroupKFold(5) theo lo"]))

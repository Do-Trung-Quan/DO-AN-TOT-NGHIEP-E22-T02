"""Viec (2) (PGS 27/10): T0 / T2 (FT-Transformer) tren fold moi, giao thuc nhu v2_run.py --search:
5 fold dev2 x 3 seed; moi fold do 30 cau hinh tren validation noi bo (fold (k+1)%5 trong 4 fold train), hieu chinh Platt trong fold.
T0 = chi lam sang; T2 = lam sang + 6 token vung anh (BioViL-T V2, PCA 128 chung cho 6 vung, fit trong fold) — anh ban 3 qua v3_base.
python v3_ftt.py T0 | T2
Dau ra cache_v3/run_<ten>.npz (OOF tho + da hieu chinh, du doan test CHUA cham), log_<ten>.csv, selected_<ten>.json. Chi in so lieu gop.
"""
import sys, json, time
import numpy as np, pandas as pd, torch
from v3_base import *
from v3_base import load_base_v3, CACHE3, IMG3, FOLDS, run_config

SEEDS = [0, 1, 2]
name = sys.argv[1]; assert name in ("T0", "T2")
v = variant(name)
B = load_base_v3()
y, fold, hi = B["y"], B["fold"], B["has_img"]
rows, te_rows = B["dev"], B["test"]
N = len(y); I = lambda m: torch.as_tensor(np.where(m)[0], device=DEV)
SP = search_space()
enc = lambda tr: encode_fold(B, tr, v["image"], v["region"], v["drop"])

K = dict(oof=np.full(N, np.nan), oof_cal=np.full(N, np.nan), test=np.zeros(N), test_cal=np.zeros(N))
log, chosen = [], []
t0 = time.time()
for k in range(5):
    tr, va = rows & (fold != k) & (fold >= 0), rows & (fold == k)
    inner = (k + 1) % 5
    itr, iva = tr & (fold != inner), tr & (fold == inner)
    Ti = to_tensors(enc(itr), B, v)
    res = []
    for ci, cfg in enumerate(SP):
        m_, sc, ep = train(Ti, v, cfg, I(itr), I(iva), seed=0)
        res.append((sc, ep, predict(m_, Ti, I(iva), v)[0]))
        log.append(dict(fold=k, cau_hinh=ci, **cfg, diem_val_noi_bo=sc, epoch_tot=ep))
    j = int(np.nanargmax([r[0] for r in res])); cfg = SP[j]; _, ep, p_in = res[j]
    chosen.append(dict(fold=k, cau_hinh=j, **cfg, epoch=ep))
    del Ti; torch.cuda.empty_cache()

    T = to_tensors(enc(tr), B, v)
    pv, pt = [], []
    for seed in SEEDS:
        model, _, _ = train(T, v, cfg, I(tr), None, epochs=ep, seed=seed)
        pv.append(predict(model, T, I(va), v)[0]); pt.append(predict(model, T, I(te_rows), v)[0])
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    K["oof"][va] = pv; K["test"][te_rows] += pt / 5
    K["oof_cal"][va] = calibrate_split(p_in, y[iva], hi[iva], pv, hi[va])
    K["test_cal"][te_rows] += calibrate_split(p_in, y[iva], hi[iva], pt, hi[te_rows]) / 5
    del T; torch.cuda.empty_cache()
    print(f"[{name}] fold {k}: cau hinh {j} epoch {ep} | {time.time()-t0:.0f}s", flush=True)

(CACHE3 / f"selected_{name}.json").write_text(json.dumps(chosen, indent=1))
pd.DataFrame(log).to_csv(CACHE3 / f"log_{name}.csv", index=False)
for t in ("test", "test_cal"): K[t] = np.where(te_rows, K[t], np.nan)
np.savez_compressed(CACHE3 / f"run_{name}.npz", **K, chosen=json.dumps(chosen))
run_config(f"tuan1_phase0/viec2_baseline_fold_moi/{name}", __file__, data=[FOLDS, IMG3], mo_hinh=name, SEEDS=SEEDS,
           khong_gian_do=SPACE, so_cau_hinh=len(SP), cau_hinh_chon=chosen)
msg = []
for g, mm in (("co anh", rows & hi), ("khong anh", rows & ~hi)):
    msg.append(f"{g} AUC {roc_auc_score(y[mm], K['oof'][mm]):.3f}")
msg.append(f"trong lo {within_batch_auc(y[rows & hi], K['oof'][rows & hi], B['batch'][rows & hi]):.3f}")
print(f"[{name}] XONG {time.time()-t0:.0f}s | " + " | ".join(msg))

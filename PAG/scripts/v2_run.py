"""Chay mot bien the PG / T0 / T2 ban 2: 5 fold dev2 (theo nguoi da gop) x 3 seed, hieu chinh Platt trong fold.

python v2_run.py PG --search      # do 30 cau hinh; validation noi bo = fold (k+1)%5 trong 4 fold train
python v2_run.py PG-shuffle       # ablation: dung cau hinh PG da chon o tung fold
python v2_run.py T2 --search
Dau ra: PAG/cache/run_<ten>.npz (OOF tho + da hieu chinh, du doan test CHUA cham diem, attention) + log_<ten>.csv.
Chi in so lieu gop.
"""
import sys, json, time
import numpy as np, pandas as pd, torch
from pag_lib import *

SEEDS = [0, 1, 2]
name = sys.argv[1]; SEARCH = "--search" in sys.argv
v = variant(name)
B = load_base()
y, fold, hi = B["y"], B["fold"], B["has_img"]
rows = B["dev"] & (hi if v["rows"] == "imaged" else True)
te_rows = B["test"] & (hi if v["rows"] == "imaged" else True)
N = len(y); I = lambda m: torch.as_tensor(np.where(m)[0], device=DEV)
SP = search_space()
sel_file = CACHE / (f"selected_{name}.json" if SEARCH else "selected_PG.json")
SEL = None if SEARCH else json.loads(sel_file.read_text())
enc = lambda tr: encode_fold(B, tr, v["image"], v["region"], v["drop"])

K = dict(oof=np.full(N, np.nan), oof_cal=np.full(N, np.nan), oof_drop=np.full(N, np.nan), oof_drop_cal=np.full(N, np.nan),
         test=np.zeros(N), test_cal=np.zeros(N), test_drop=np.zeros(N), test_drop_cal=np.zeros(N), att=np.full((N, 6), np.nan))
log, chosen = [], []
t0 = time.time()
for k in range(5):
    tr, va = rows & (fold != k) & (fold >= 0), rows & (fold == k)
    inner = (k + 1) % 5
    itr, iva = tr & (fold != inner), tr & (fold == inner)
    Ti = to_tensors(enc(itr), B, v)
    cands = list(enumerate(SP)) if SEARCH else [(SEL[k]["cau_hinh"], {kk: SEL[k][kk] for kk in SPACE})]
    res = []
    for ci, cfg in cands:
        m_, sc, ep = train(Ti, v, cfg, I(itr), I(iva), seed=0)
        res.append((sc, ep, predict(m_, Ti, I(iva), v)[0], predict(m_, Ti, I(iva), v, drop_all_img=True)[0]))
        log.append(dict(fold=k, cau_hinh=ci, **cfg, diem_val_noi_bo=sc, epoch_tot=ep))
    j = int(np.nanargmax([r[0] for r in res])); ci, cfg = cands[j]; _, ep, p_in, p_in_drop = res[j]
    chosen.append(dict(fold=k, cau_hinh=ci, **cfg, epoch=ep))
    del Ti; torch.cuda.empty_cache()
    y_in, hi_in = y[iva], hi[iva]

    T = to_tensors(enc(tr), B, v)
    iva_, ite = I(va), I(te_rows)
    pv, pdv, av, pt, pdt = [], [], [], [], []
    want_drop = v["image"] is not None and v["arch"] != "ftt" and v["only"] is None
    for seed in SEEDS:
        model, _, _ = train(T, v, cfg, I(tr), None, epochs=ep, seed=seed)
        p, _, a = predict(model, T, iva_, v); pv.append(p); pt.append(predict(model, T, ite, v)[0])
        if a is not None: av.append(a)
        if want_drop:
            pdv.append(predict(model, T, iva_, v, drop_all_img=True)[0]); pdt.append(predict(model, T, ite, v, drop_all_img=True)[0])
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    K["oof"][va] = pv; K["test"][te_rows] += pt / 5
    K["oof_cal"][va] = calibrate_split(p_in, y_in, hi_in, pv, hi[va])
    K["test_cal"][te_rows] += calibrate_split(p_in, y_in, hi_in, pt, hi[te_rows]) / 5
    if av: K["att"][va] = np.mean(av, 0)
    if want_drop:
        pdv, pdt = np.mean(pdv, 0), np.mean(pdt, 0)
        K["oof_drop"][va] = pdv; K["test_drop"][te_rows] += pdt / 5
        # bo anh -> giong nguoi khong phim: hieu chinh bang du doan bo anh tren validation noi bo
        f = np.zeros(iva.sum(), bool)
        K["oof_drop_cal"][va] = calibrate_split(p_in_drop, y_in, f, pdv, np.zeros(va.sum(), bool))
        K["test_drop_cal"][te_rows] += calibrate_split(p_in_drop, y_in, f, pdt, np.zeros(te_rows.sum(), bool)) / 5
    del T; torch.cuda.empty_cache()
    print(f"[{name}] fold {k}: cau hinh {ci} epoch {ep} | {time.time()-t0:.0f}s", flush=True)

if SEARCH: sel_file.write_text(json.dumps(chosen, indent=1))
pd.DataFrame(log).to_csv(CACHE / f"log_{name}.csv", index=False)
for t in ("test", "test_cal", "test_drop", "test_drop_cal"): K[t] = np.where(te_rows, K[t], np.nan)
np.savez_compressed(CACHE / f"run_{name}.npz", **K, chosen=json.dumps(chosen))
msg = []
for g, mm in (("co anh", rows & hi), ("khong anh", rows & ~hi)):
    if mm.sum() and 0 < y[mm].sum() < mm.sum():
        msg.append(f"{g} AUC {roc_auc_score(y[mm], K['oof'][mm]):.3f}")
if (rows & hi).sum(): msg.append(f"trong lo {within_batch_auc(y[rows & hi], K['oof'][rows & hi], B['batch'][rows & hi]):.3f}")
print(f"[{name}] XONG {time.time()-t0:.0f}s | " + " | ".join(msg))

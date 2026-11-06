"""Buoc 4 (tuan 3, PG_update muc 6): huan luyen PG ban 3 tren fold moi, giao thuc nhu v3_ftt.py / v2_run.py.

  python v3_run_pg.py PG-global --search   # 1 nut anh = C (lam truoc, thay chot 01/11)
  python v3_run_pg.py PG --search          # PG 6 vung (PG chinh cua de cuong)
  python v3_run_pg.py PG-shuffle           # H2: xao tron canh E1/E2, DUNG LAI cau hinh PG da chon o tung fold
  them --smoke de chay thu nhanh (1 fold, 2 cau hinh, 2 epoch, 1 seed; ket qua mang duoi _smoke, khong dung bao cao)

Moi fold bao cao k (0..4):
  1. do (--search) toi da 30 cau hinh: hoc tren 3 fold, cham tren fold validation noi bo (k+1)%5 — ma hoa "noi" fit tren 3 fold
  2. huan luyen lai cau hinh tot nhat tren 4 fold (ma hoa "ngoai"), so epoch = epoch tot nhat o buoc 1, 3 seed, lay trung binh
  3. du doan fold k (OOF) + test (CHUA cham), hieu chinh Platt trong fold rieng nhom co / khong anh
Dau vao: load_base_v3() (10 cap nghe, fold p05_folds.npz). Ma hoa giong het v3_build_graphs.py — kiem tra voi file
cache_v3/graphs/pg_*_fold0_ngoai.pt neu co. Dau ra cache_v3/run_<ten>_nghe10.npz, log_*.csv, selected_*.json (du lieu that,
KHONG commit) + results_v3/tuan3_PG_nghe10/<ten>/run_config.json. Chi in so lieu gop.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_base import *  # noqa: F401,F403,E402
from v3_base import load_base_v3, CACHE3, IMG3, FOLDS, run_config, TAG, NGHE10
from pag_lib import DEV, SPACE, calibrate_split, encode_fold, predict, search_space, to_tensors, train, variant
from v2_lib import within_batch_auc
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("name", choices=["PG-global", "PG", "PG-shuffle"])
ap.add_argument("--search", action="store_true", help="do cau hinh (bat buoc voi PG-global va PG)")
ap.add_argument("--smoke", action="store_true", help="chay thu nhanh tren CPU")
args = ap.parse_args()
assert NGHE10, "PG ban 3 dung 10 cap nghe: khong dat PG_NGHE10=0"
name = args.name
if name in ("PG-global", "PG") and not args.search:
    raise SystemExit(f"{name} can --search (do cau hinh tren validation noi bo)")
if name == "PG-shuffle" and args.search:
    raise SystemExit("PG-shuffle dung lai cau hinh cua PG (khai bao truoc), khong do lai")

SMOKE = "_smoke" if args.smoke else ""
SEEDS = [0] if args.smoke else [0, 1, 2]
FOLDS_RUN = [0] if args.smoke else range(5)
EPOCHS = 2 if args.smoke else 80
SP = search_space()[: 2 if args.smoke else 30]

v = variant(name)
B = load_base_v3()
y, fold, hi = B["y"], B["fold"], B["has_img"]
rows, te_rows = B["dev"], B["test"]
N = len(y)
I = lambda m: torch.as_tensor(np.where(m)[0], device=DEV)
enc = lambda tr: encode_fold(B, tr, v["image"], v["region"], v["drop"])
if name == "PG-shuffle":
    SEL = json.loads((CACHE3 / f"selected_PG{TAG}{SMOKE}.json").read_text())
    SEL = {s["fold"]: s for s in SEL}


def check_graph_file(E, tr):
    """Ma hoa trong luc huan luyen phai trung voi file do thi da nop (v3_build_graphs.py)."""
    f = CACHE3 / "graphs" / f"pg_{'global' if name == 'PG-global' else 'vung'}_fold0_ngoai.pt"
    if not f.exists():
        print(f"  (khong co {f.name}: bo qua doi chieu file do thi)")
        return
    G = torch.load(f, weights_only=False)
    h = G["hang"].numpy()
    ok = np.allclose(G["x_img"].numpy(), E["img"][h]) and all(np.allclose(G["x_clin"][g].numpy(), E["clin"][g][h]) for g in G["x_clin"])
    print(f"  doi chieu {f.name}: {'KHOP' if ok else 'LECH'}")
    assert ok, "ma hoa lech voi file do thi da dung — dung lai kiem tra"


K = dict(oof=np.full(N, np.nan), oof_cal=np.full(N, np.nan), oof_drop=np.full(N, np.nan), oof_drop_cal=np.full(N, np.nan),
         test=np.zeros(N), test_cal=np.zeros(N), test_drop=np.zeros(N), test_drop_cal=np.zeros(N), att=np.full((N, 6), np.nan))
log, chosen = [], []
t0 = time.time()
print(f"[{name}{SMOKE}] thiet bi {DEV} | {len(SP)} cau hinh | seeds {SEEDS} | dev {int(rows.sum())} | test {int(te_rows.sum())} (khong cham)")
for k in FOLDS_RUN:
    tr, va = rows & (fold != k) & (fold >= 0), rows & (fold == k)
    inner = (k + 1) % 5
    itr, iva = tr & (fold != inner), tr & (fold == inner)

    # --- 1. chon cau hinh tren validation noi bo
    Ti = to_tensors(enc(itr), B, v)
    if name == "PG-shuffle":
        s = SEL[k]; cands = [(s["cau_hinh"], {kk: s[kk] for kk in SPACE})]
    else:
        cands = list(enumerate(SP))
    res = []
    for ci, cfg in cands:
        m_, sc, ep = train(Ti, v, cfg, I(itr), I(iva), epochs=EPOCHS, seed=0)
        res.append((sc, ep, predict(m_, Ti, I(iva), v)[0], predict(m_, Ti, I(iva), v, drop_all_img=True)[0]))
        log.append(dict(fold=k, cau_hinh=ci, **cfg, diem_val_noi_bo=sc, epoch_tot=ep))
    j = int(np.nanargmax([r[0] for r in res]))
    ci, cfg = cands[j]
    _, ep, p_in, p_in_drop = res[j]
    chosen.append(dict(fold=k, cau_hinh=ci, **cfg, epoch=ep))
    del Ti
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. huan luyen lai tren 4 fold, 3 seed
    E = enc(tr)
    if k == 0:
        check_graph_file(E, tr)
    T = to_tensors(E, B, v)
    pv, pdv, av, pt, pdt = [], [], [], [], []
    for seed in SEEDS:
        model, _, _ = train(T, v, cfg, I(tr), None, epochs=ep, seed=seed)
        p, _, a = predict(model, T, I(va), v)
        pv.append(p); pt.append(predict(model, T, I(te_rows), v)[0])
        if a is not None:
            av.append(a)
        pdv.append(predict(model, T, I(va), v, drop_all_img=True)[0])
        pdt.append(predict(model, T, I(te_rows), v, drop_all_img=True)[0])

    # --- 3. OOF + test, hieu chinh Platt trong fold
    pv, pt, pdv, pdt = np.mean(pv, 0), np.mean(pt, 0), np.mean(pdv, 0), np.mean(pdt, 0)
    y_in, hi_in = y[iva], hi[iva]
    K["oof"][va] = pv; K["test"][te_rows] += pt / 5
    K["oof_cal"][va] = calibrate_split(p_in, y_in, hi_in, pv, hi[va])
    K["test_cal"][te_rows] += calibrate_split(p_in, y_in, hi_in, pt, hi[te_rows]) / 5
    if av:
        K["att"][va] = np.mean(av, 0)
    # PG-drop (bo moi nut anh khi danh gia): hieu chinh nhu nguoi khong phim
    f0 = np.zeros(iva.sum(), bool)
    K["oof_drop"][va] = pdv; K["test_drop"][te_rows] += pdt / 5
    K["oof_drop_cal"][va] = calibrate_split(p_in_drop, y_in, f0, pdv, np.zeros(va.sum(), bool))
    K["test_drop_cal"][te_rows] += calibrate_split(p_in_drop, y_in, f0, pdt, np.zeros(te_rows.sum(), bool)) / 5
    del T
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{name}{SMOKE}] fold {k}: cau hinh {ci} epoch {ep} | {time.time() - t0:.0f}s", flush=True)

stem = f"{name}{TAG}{SMOKE}"
if name != "PG-shuffle":
    (CACHE3 / f"selected_{stem}.json").write_text(json.dumps(chosen, indent=1))
pd.DataFrame(log).to_csv(CACHE3 / f"log_{stem}.csv", index=False)
for t in ("test", "test_cal", "test_drop", "test_drop_cal"):
    K[t] = np.where(te_rows, K[t], np.nan)
np.savez_compressed(CACHE3 / f"run_{stem}.npz", **K, chosen=json.dumps(chosen))
if not args.smoke:
    run_config(f"tuan3_PG{TAG}/{name}", __file__, data=[FOLDS, IMG3, MERG / "cache_pg" / "clin_raw.pkl"], mo_hinh=name,
               SEEDS=SEEDS, khong_gian_do=SPACE, so_cau_hinh=len(SP) if name != "PG-shuffle" else "dung lai cau hinh PG",
               cau_hinh_chon=chosen, thiet_bi=str(DEV), nghe="10 cap (v3_base.add_job_levels)")

msg = []
done = rows & np.isfinite(K["oof"])
for g, mm in (("co anh", done & hi), ("khong anh", done & ~hi)):
    if mm.sum() and 0 < y[mm].sum() < mm.sum():
        msg.append(f"{g} AUC {roc_auc_score(y[mm], K['oof'][mm]):.3f}")
mm = done & hi
if mm.sum():
    msg.append(f"trong lo {within_batch_auc(y[mm], K['oof'][mm], B['batch'][mm]):.3f}")
print(f"[{stem}] XONG {time.time() - t0:.0f}s | " + " | ".join(msg) + f" | -> cache_v3/run_{stem}.npz")

"""Xuat mo hinh PG DA KHOA (07/11) de dua len web — hoc lai dung cau hinh, LUU trong so, kiem tai tao.

  python v3_xuat_pg.py            # 5 fold: mo hinh noi bo (Platt) + 3 seed tren 4 fold, luu, so voi diem da luu
  python v3_xuat_pg.py --smoke    # chay thu fold 0, khong ghi mo hinh chinh

Tai sao phai hoc lai: v3_run_pg.py chi luu diem du doan (cache_v3/run_PG_nghe10.npz), khong luu trong so.
Script nay lap lai DUNG cac buoc cua v3_run_pg.py cho cau hinh da chon o tung fold (cache_v3/selected_PG_nghe10.json),
KHONG do lai cau hinh, KHONG sua v3_run_pg.py / v3_test_mot_lan.py:
  1. mo hinh noi bo: hoc tren 3 fold, dung som tren fold (k+1)%5, seed 0 -> diem validation noi bo -> tham so Platt
     (rieng nhom co / khong phim, nhu calibrate_split)
  2. 3 seed tren 4 fold, so epoch = epoch da chon (selected_PG_nghe10.json)
  3. luu: trong so, ma hoa lam sang (spec), chuan hoa + PCA anh (mang numpy), tham so Platt
Kiem tai tao: so diem moi voi diem da luu — OOF dev (oof_cal) va test (test_cal). Phan test CHI so hai day diem voi nhau,
KHONG dung nhan test, KHONG tinh chi so test -> khong phai cham lai test.
Dau ra: cache_v3/mo_hinh_PG_web/ (rieng tu, .gitignore PAG/cache*/) + results_v3/tuan7_xuat_web/ (BAO_CAO.md, run_config.json,
tai_tao.csv). Chi in so lieu gop.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_base import *  # noqa: F401,F403,E402
from v3_base import load_base_v3, CACHE3, IMG3, FOLDS, run_config, TAG, NGHE10, JOB_COLS  # noqa: E402
from v3_lib import RES3  # noqa: E402
import pag_lib  # noqa: E402
from pag_lib import (DEV, GNAMES, CTX, ClinEncoder, ImgEncoder, calibrate_split, encode_fold, predict,  # noqa: E402
                     to_tensors, train, variant)
from v2_lib import within_batch_auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
assert NGHE10 and TAG == "_nghe10", "PG khoa dung 10 cap nghe"

NAME = "PG"
SEEDS = [0, 1, 2]
FOLDS_RUN = [0] if args.smoke else range(5)
OUT = CACHE3 / ("mo_hinh_PG_web_smoke" if args.smoke else "mo_hinh_PG_web")
REP = "tuan7_xuat_web" + ("/smoke" if args.smoke else "")
OUT.mkdir(parents=True, exist_ok=True)

v = variant(NAME)
B = load_base_v3()
y, fold, hi = B["y"], B["fold"], B["has_img"]
rows, te_rows = B["dev"], B["test"]
N = len(y)
I = lambda m: torch.as_tensor(np.where(m)[0], device=DEV)  # noqa: E731

SEL = {s["fold"]: s for s in json.loads((CACHE3 / f"selected_{NAME}{TAG}.json").read_text())}
OLD = np.load(CACHE3 / f"run_{NAME}{TAG}.npz", allow_pickle=False)
CFG_KEYS = ["dim", "so_lop", "dropout", "lr", "weight_decay", "p_modality_dropout"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p_in, y_in, hi_in):
    """Giong calibrate_split + v2_lib.calibrate (LR C=1e6 tren logit): tra tham so (a, b) moi nhom, None = giu nguyen."""
    out = {}
    for g, key in ((True, "co_phim"), (False, "khong_phim")):
        mi = hi_in == g
        if mi.sum() < 20 or y_in[mi].min() == y_in[mi].max():
            out[key] = None
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_in[mi])[:, None], y_in[mi])
        out[key] = dict(a=float(m.coef_[0, 0]), b=float(m.intercept_[0]))
    return out


def apply_platt(P, p, hi_):
    out = np.full(len(p), np.nan)
    for g, key in ((True, "co_phim"), (False, "khong_phim")):
        m = hi_ == g
        out[m] = p[m] if P[key] is None else 1 / (1 + np.exp(-(P[key]["a"] * logit(p[m]) + P[key]["b"])))
    return out


def encoders(tr):
    """Nhu encode_fold(B, tr, 'a', True, drop) nhung giu lai bo ma hoa de luu; kiem trung voi encode_fold."""
    ce = ClinEncoder(v["drop"]).fit(B["raw"], tr)
    ie = ImgEncoder(128).fit(B["reg_a"], tr & hi)
    E = encode_fold(B, tr, v["image"], v["region"], v["drop"])
    clin, _, _ = ce.transform(B["raw"])
    assert all(np.array_equal(clin[g], E["clin"][g]) for g in E["clin"]), "ma hoa lam sang lech encode_fold"
    img = ie.transform(B["reg_a"]) * hi[:, None, None]
    assert np.array_equal(img, E["img"]), "ma hoa anh lech encode_fold"
    # PCA whiten luu duoi dang mang -> app khong phu thuoc phien ban sklearn; kiem tai lap
    pca = ie.pca
    Xr = B["reg_a"][hi].reshape(-1, B["reg_a"].shape[-1]).astype(np.float64)
    Z = ((Xr - ie.mu) / ie.sd - pca.mean_) @ pca.components_.T / np.sqrt(pca.explained_variance_)
    Z = np.clip(Z, -10, 10).astype(np.float32).reshape(-1, 6, Z.shape[-1])
    assert np.abs(Z - E["img"][hi]).max() < 1e-4, "PCA dang mang lech sklearn"
    img_np = dict(mu=ie.mu, sd=ie.sd, pca_mean=pca.mean_, pca_comp=pca.components_, pca_var=pca.explained_variance_)
    clin_spec = {c: list(s) for c, s in ce.spec.items()}
    return E, img_np, clin_spec


new = dict(oof_cal=np.full(N, np.nan), test_cal=np.zeros(N), oof=np.full(N, np.nan), test=np.zeros(N))
log = []
t0 = time.time()
print(f"[xuat PG] thiet bi {DEV} | fold {list(FOLDS_RUN)} | seeds {SEEDS} | dev {int(rows.sum())} | test {int(te_rows.sum())} (khong cham)")
for k in FOLDS_RUN:
    s = SEL[k]; cfg = {kk: s[kk] for kk in CFG_KEYS}; ep = int(s["epoch"])
    tr, va = rows & (fold != k) & (fold >= 0), rows & (fold == k)
    inner = (k + 1) % 5
    itr, iva = tr & (fold != inner), tr & (fold == inner)
    d = OUT / f"fold{k}"; d.mkdir(exist_ok=True)

    # --- 1. mo hinh noi bo (chi de lay diem validation noi bo cho Platt), giong buoc do cau hinh: seed 0, toi da 80 epoch
    Ti = to_tensors(encode_fold(B, itr, v["image"], v["region"], v["drop"]), B, v)
    m_in, sc_in, ep_in = train(Ti, v, cfg, I(itr), I(iva), epochs=80, seed=0)
    p_in = predict(m_in, Ti, I(iva), v)[0]
    del Ti, m_in
    P = fit_platt(p_in, y[iva], hi[iva])
    chk = np.nanmax(np.abs(apply_platt(P, p_in, hi[iva]) - calibrate_split(p_in, y[iva], hi[iva], p_in, hi[iva])))
    assert chk < 1e-6, f"tham so Platt luu lech calibrate_split ({chk})"

    # --- 2. 3 seed tren 4 fold, so epoch da chon
    E, img_np, clin_spec = encoders(tr)
    T = to_tensors(E, B, v)
    pv, pt = [], []
    for seed in SEEDS:
        model, _, _ = train(T, v, cfg, I(tr), None, epochs=ep, seed=seed)
        pv.append(predict(model, T, I(va), v)[0]); pt.append(predict(model, T, I(te_rows), v)[0])
        torch.save({k_: x.detach().cpu() for k_, x in model.state_dict().items()}, d / f"seed{seed}.pt")
    kien_truc = dict(d_img=int(T["d_img"]), n_img=int(T["n_img"]), clin_dims=[int(c) for c in T["clin_dims"]],
                     dim=cfg["dim"], so_lop=cfg["so_lop"], dropout=cfg["dropout"], arch=v["arch"])
    del T
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    new["oof"][va] = pv; new["test"][te_rows] += pt / 5
    new["oof_cal"][va] = apply_platt(P, pv, hi[va])
    new["test_cal"][te_rows] += apply_platt(P, pt, hi[te_rows]) / 5

    np.savez(d / "ma_hoa_anh.npz", **img_np)
    (d / "ma_hoa_lam_sang.json").write_text(json.dumps(clin_spec, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (d / "platt.json").write_text(json.dumps(P, indent=1))
    (d / "cau_hinh.json").write_text(json.dumps(dict(fold=k, cau_hinh=s["cau_hinh"], cfg=cfg, epoch=ep, kien_truc=kien_truc,
                                                     noi_bo=dict(epoch_tot=int(ep_in), epoch_da_chon=ep, diem_val=float(sc_in))),
                                                ensure_ascii=False, indent=1))
    log.append(dict(fold=k, cau_hinh=s["cau_hinh"], epoch=ep, epoch_noi_bo_moi=int(ep_in), epoch_noi_bo_khop=int(ep_in) == ep,
                    lech_oof_max=float(np.nanmax(np.abs(new["oof_cal"][va] - OLD["oof_cal"][va]))),
                    giay=round(time.time() - t0)))
    print(f"[xuat PG] fold {k}: cau hinh {s['cau_hinh']} epoch {ep} | noi bo epoch moi {ep_in} "
          f"({'khop' if ep_in == ep else 'LECH'}) | lech OOF max {log[-1]['lech_oof_max']:.4f} | {time.time() - t0:.0f}s", flush=True)

# --- 3. kiem tai tao (dev OOF: co nhan dev; test: chi so diem voi diem, khong nhan)
def so(a, b, m):
    a, b = a[m], b[m]
    return dict(n=int(m.sum()), lech_tb=float(np.mean(np.abs(a - b))), lech_max=float(np.max(np.abs(a - b))),
                pearson=float(pearsonr(a, b)[0]), spearman=float(spearmanr(a, b)[0]))


done = rows & np.isfinite(new["oof_cal"])
R = []
for g, mm in (("có phim", hi), ("không phim", ~hi)):
    R.append(dict(tap="dev OOF (oof_cal)", nhom=g, **so(new["oof_cal"], OLD["oof_cal"], done & mm),
                  auc_moi=float(roc_auc_score(y[done & mm], new["oof_cal"][done & mm])),
                  auc_cu=float(roc_auc_score(y[done & mm], OLD["oof_cal"][done & mm]))))
    if not args.smoke:
        R.append(dict(tap="test (test_cal), KHONG dung nhan", nhom=g, **so(new["test_cal"], OLD["test_cal"], te_rows & mm)))
mm = done & hi
wb_new, wb_old = within_batch_auc(y[mm], new["oof_cal"][mm], B["batch"][mm]), within_batch_auc(y[mm], OLD["oof_cal"][mm], B["batch"][mm])
R = pd.DataFrame(R)
rd = RES3 / REP; rd.mkdir(parents=True, exist_ok=True)
R.to_csv(rd / "tai_tao.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(log).to_csv(rd / "theo_fold.csv", index=False, encoding="utf-8-sig")

# --- 4. cau hinh chung cho app
tinh = sorted(B["raw"]["tinh"][rows].dropna().astype(str).unique().tolist())
files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "cau_hinh_chung.json")
chung = dict(
    mo_hinh="PG (khoa 07/11/2026, cham test 1 lan)", bien_the=v, nghe="10 cap (v3_base.add_job_levels, cot " + ", ".join(JOB_COLS) + ")",
    nhom_bien={g: list(pag_lib.GROUPS[g]) for g in GNAMES}, boi_canh=list(CTX), tinh_hop_le=tinh,
    anh="BioViL-T dong bang, V2, 6 vung x 512 (v3_image.py) -> chuan hoa + PCA whiten 128 theo fold; area*10",
    du_doan="trung binh 5 fold; moi fold: trung binh sigmoid 3 seed -> Platt rieng nhom co/khong phim -> trung binh 5 fold",
    canh_bao="Dong Nai va Phu Yen gan nhu khong co ca co phim trong du lieu hoc",
    sha256={str(p.relative_to(OUT)).replace("\\", "/"): sha(p) for p in files},
    du_lieu={p.name: sha(p) for p in (FOLDS, IMG3, MERG / "cache_pg" / "clin_raw.pkl")})
(OUT / "cau_hinh_chung.json").write_text(json.dumps(chung, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
run_config(REP, __file__, data=[FOLDS, IMG3, MERG / "cache_pg" / "clin_raw.pkl", CACHE3 / f"selected_{NAME}{TAG}.json",
                                CACHE3 / f"run_{NAME}{TAG}.npz"],
           mo_hinh=NAME, SEEDS=SEEDS, folds=list(FOLDS_RUN), thiet_bi=str(DEV), torch=torch.__version__, thu_muc_mo_hinh=str(OUT))

# --- 5. bao cao
def md(df):
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)])


L = ["# Tuần 7 — Xuất mô hình PG đã khoá cho web (học lại, lưu trọng số, kiểm tái tạo)", "",
     f"- Script `scripts_v3/v3_xuat_pg.py`; cấu hình từng fold lấy từ `selected_PG_nghe10.json` (không dò lại).",
     f"- Mô hình lưu ở `cache_v3/{OUT.name}/` (riêng tư). Thiết bị {DEV}. Thời gian {time.time() - t0:.0f} s.",
     "- Phần test chỉ so điểm mới với điểm đã lưu ngày 07/11, **không dùng nhãn test, không tính chỉ số test**.", "",
     "## Theo fold", "", md(pd.DataFrame(log)), "",
     "## So điểm mới với điểm đã lưu", "", md(R), "",
     f"AUC trong lô dev (có phim): mới {wb_new:.4f}, cũ {wb_old:.4f}.", ""]
(rd / "BAO_CAO.md").write_text("\n".join(L), encoding="utf-8")
print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"AUC trong lo dev co phim: moi {wb_new:.4f} | cu {wb_old:.4f}")
print(f"[xuat PG] XONG {time.time() - t0:.0f}s -> cache_v3/{OUT.name} | results_v3/{REP}/BAO_CAO.md")

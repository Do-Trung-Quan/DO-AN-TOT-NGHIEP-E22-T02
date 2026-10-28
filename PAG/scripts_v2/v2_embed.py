"""PAG_V2 §7 — E1: embedding node benh nhan (CLS) cua PG da huan luyen, tach theo fold.

Fold k: huan luyen PG (cau hinh + so epoch da chon o fold k trong selected_PG.json, seed 0) tren dev2 & fold != k,
roi lay vector CLS (sau LayerNorm) cho MOI hang. Embedding cua fold k chi den tu mo hinh khong thay fold k.
Dau ra: PAG/cache/emb_PG.npz (emb_0..emb_4, p_0..p_4; theo thu tu hang cua core -> PII, KHONG dua len GitHub).
Chi in so lieu gop.
"""
import json, time
import numpy as np, torch
from pag_lib import *

B = load_base(); v = variant("PG")
y, fold, hi = B["y"], B["fold"], B["has_img"]
N = len(y); I = lambda m: torch.as_tensor(np.where(m)[0], device=DEV)
SEL = json.loads((CACHE / "selected_PG.json").read_text())
out, t0 = {}, time.time()
for k in range(5):
    tr = B["dev"] & (fold != k) & (fold >= 0)
    T = to_tensors(encode_fold(B, tr, v["image"], v["region"], v["drop"]), B, v)
    cfg = {kk: SEL[k][kk] for kk in SPACE}
    model, _, _ = train(T, v, cfg, I(tr), None, epochs=SEL[k]["epoch"], seed=0)
    p, h, _ = predict(model, T, torch.arange(N, device=DEV), v)
    out[f"emb_{k}"], out[f"p_{k}"] = h.astype(np.float32), p.astype(np.float32)
    va = B["dev"] & (fold == k)
    print(f"fold {k}: dim {h.shape[1]} | AUC dau PG tren fold {roc_auc_score(y[va], p[va]):.3f} | {time.time()-t0:.0f}s", flush=True)
    del T, model; torch.cuda.empty_cache()
np.savez_compressed(CACHE / "emb_PG.npz", **out)
print("EMB XONG")

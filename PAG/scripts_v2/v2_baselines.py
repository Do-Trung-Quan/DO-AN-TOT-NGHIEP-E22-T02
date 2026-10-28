"""Baseline ban 2 (PAG_V2 tuan 2): cung fold2/dev2, 3 seed, encoder fit trong fold, hieu chinh Platt trong fold.

L0      LR lam sang (5 nhom + kham + tinh) + co phim
L1      LR anh toan cuc BioViL-T dong bang (512, sau che chu + cat), chi nguoi co phim
Z1      LR 6 vung BioViL-T (PCA 128/vung trong fold) + dien tich, chi nguoi co phim
L1-kham LR chi tren dau hieu kham thuc the (moi: kham co bao nhieu thong tin)
L2      LR lam sang + anh toan cuc (0 neu khong phim) + co phim
X0/X2   XGBoost lam sang / lam sang + anh toan cuc
L3b     LR nhu L2 + diem nguy co trung binh cua hang xom co phim tren do thi kNN (k chon tren validation noi bo)
B3b     MLP cung dau vao L3b
GAT     GATv2 2 lop tren do thi kNN benh nhan (patient-as-node), k chon tren validation noi bo
Do thi kNN: cosine tren lam sang da ma hoa (fit trong fold) + canh cau noi moi nguoi khong phim -> nguoi co phim gan nhat.
Dau ra PAG/cache/run_baselines.npz. Chi in so lieu gop.
"""
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from xgboost import XGBClassifier
from pag_lib import *

B = load_base(); y, fold, hi, dev, test = B["y"], B["fold"], B["has_img"], B["dev"], B["test"]
N = len(y); SEEDS = [0, 1, 2]; KS = [5, 10, 20]
xgb = lambda s, spw: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=2, scale_pos_weight=spw, tree_method="hist", random_state=s, n_jobs=8, verbosity=0)


def feats(trm):
    E = encode_fold(B, trm, "a", True)
    g = ImgEncoder(128).fit(B["glob_a"], trm & hi).transform(B["glob_a"]) * hi[:, None]
    clin = np.column_stack([E["flat_clin"], hi])
    return dict(clin=clin, glob=B["glob_a"], reg=np.column_stack([E["img"].reshape(N, -1), E["area"]]),
                kham=E["clin"]["Khám thực thể"], clin_glob=np.column_stack([clin, g]), g=g)


# ---------------------------------------------------------------- do thi kNN + diem hang xom --
def knn_graph(X, k):
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6); Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9)
    Xt = torch.as_tensor(Xn, dtype=torch.float32, device=DEV); src, dst = [], []
    for i in range(0, N, 2048):
        s = Xt[i:i + 2048] @ Xt.T; s[torch.arange(s.shape[0]), torch.arange(i, i + s.shape[0])] = -2
        nb = s.topk(k, 1).indices.cpu().numpy(); src.append(np.repeat(np.arange(i, i + len(nb)), k)); dst.append(nb.ravel())
    src, dst = np.concatenate(src), np.concatenate(dst)
    im, pl = np.where(hi)[0], np.where(~hi)[0]                    # cau noi
    near = np.concatenate([im[(Xt[pl[i:i + 2048]] @ Xt[im].T).argmax(1).cpu().numpy()] for i in range(0, len(pl), 2048)])
    src, dst = np.concatenate([src, pl]), np.concatenate([dst, near])
    e = np.unique(np.sort(np.stack([src, dst]), 0), axis=1)
    return np.concatenate([e, e[::-1]], 1)


def nb_score(e, s):
    keep = hi[e[1]]; a, b = e[0][keep], e[1][keep]
    tot, cnt = np.zeros(N), np.zeros(N); np.add.at(tot, a, s[b]); np.add.at(cnt, a, 1)
    return np.column_stack([np.where(cnt > 0, tot / np.maximum(cnt, 1), 0), np.log1p(cnt)])


def probe(trm, Fe):
    """Diem anh: LR tren anh toan cuc hoc rieng tren hang TRAIN co phim -> cham moi nguoi co phim."""
    m = lr().fit(Fe["glob"][trm & hi], y[trm & hi]); s = np.zeros(N); s[hi] = m.predict_proba(Fe["glob"][hi])[:, 1]; return s


# ---------------------------------------------------------------- GAT / MLP toan do thi --
class GATv2(nn.Module):
    def __init__(s, i, o, h, drop):
        super().__init__(); s.h, s.o = h, o
        s.ls, s.ld = nn.Linear(i, h * o), nn.Linear(i, h * o); s.att = nn.Parameter(torch.randn(1, h, o) * 0.1); s.drop = drop
    def forward(s, x, e):
        xs, xd = s.ls(x).view(-1, s.h, s.o), s.ld(x).view(-1, s.h, s.o)
        a = (F.leaky_relu(xs[e[0]] + xd[e[1]], 0.2) * s.att).sum(-1)
        a = a - torch.zeros(x.shape[0], s.h, device=x.device).scatter_reduce(0, e[1][:, None].expand(-1, s.h), a, "amax", include_self=False)[e[1]]
        a = a.exp(); den = torch.zeros(x.shape[0], s.h, device=x.device).index_add_(0, e[1], a)
        a = F.dropout(a / den[e[1]].clamp_min(1e-9), s.drop, s.training)
        return torch.zeros_like(xs).index_add_(0, e[1], xs[e[0]] * a[..., None]).reshape(x.shape[0], -1)


class GNet(nn.Module):
    def __init__(s, d_in, hid=64, heads=4, drop=0.3, graph=True):
        super().__init__(); s.graph = graph
        s.inp = nn.Sequential(nn.Linear(d_in, hid), nn.LayerNorm(hid), nn.ReLU(), nn.Dropout(drop))
        if graph: s.g1, s.g2 = GATv2(hid, hid // heads, heads, drop), GATv2(hid, hid, 1, drop)
        s.out = nn.Sequential(nn.Linear(hid * 2 if graph else hid, hid // 2), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid // 2, 1))
    def forward(s, x, e):
        h = s.inp(x)
        if not s.graph: return s.out(h)[:, 0]
        g = F.elu(s.g1(h, e)); g = s.g2(g, e)
        return s.out(torch.cat([h, g], 1))[:, 0]


def fit_gnet(X, e, trm, vam, seed, graph, epochs=300, patience=30):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV); et = torch.as_tensor(e, device=DEV)
    sl = torch.arange(N, device=DEV); et = torch.cat([et, torch.stack([sl, sl])], 1)
    yt = torch.as_tensor(y, dtype=torch.float32, device=DEV); tri = torch.as_tensor(np.where(trm)[0], device=DEV)
    m = GNet(X.shape[1], graph=graph).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
    pw = (1 - y[trm].mean()) / y[trm].mean()
    best, state, bad, best_ep = -1, None, 0, epochs
    for ep in range(epochs):
        m.train(); lo = m(Xt, et)
        loss = F.binary_cross_entropy_with_logits(lo[tri], yt[tri], pos_weight=torch.tensor(pw, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if vam is not None:
            m.eval()
            with torch.no_grad(): p = torch.sigmoid(m(Xt, et)).cpu().numpy()
            sc = sel_score(y[vam], p[vam], hi[vam])
            if sc > best + 1e-4: best, bad, best_ep, state = sc, 0, ep + 1, {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    with torch.no_grad(): p = torch.sigmoid(m(Xt, et)).cpu().numpy()
    return p, best, best_ep


# ---------------------------------------------------------------- chay --
SK = {"L0": ("lr", "clin", False), "L1": ("lr", "glob", True), "Z1": ("lr", "reg", True), "L1-khám": ("lr", "kham", False),
      "L2": ("lr", "clin_glob", False), "X0": ("xgb", "clin", False), "X2": ("xgb", "clin_glob", False)}
ALL = list(SK) + ["L3b", "B3b", "GAT"]
OUT = {f"{t}_{m}": np.full(N, np.nan) for m in ALL for t in ("oof", "oof_cal", "test", "test_cal")}
for m in ALL:
    for t in ("test", "test_cal"): OUT[f"{t}_{m}"] = np.zeros(N)
KSEL = []
t0 = time.time()


def sk_fit_predict(kind, X, trm, apply_rows):
    ps = []
    for s in SEEDS:
        mod = lr(seed=s) if kind == "lr" else xgb(s, (y[trm] == 0).sum() / y[trm].sum())
        mod.fit(X[trm], y[trm]); ps.append(mod.predict_proba(X[apply_rows])[:, 1])
    return np.mean(ps, 0)


for k in range(5):
    tr, va = dev & (fold != k) & (fold >= 0), dev & (fold == k)
    inner = (k + 1) % 5; itr, iva = tr & (fold != inner), tr & (fold == inner)
    Fi, Fo = feats(itr), feats(tr)
    for m, (kind, fs, io) in SK.items():
        sub = hi if io else np.ones(N, bool)
        p_in = sk_fit_predict(kind, Fi[fs], itr & sub, iva & sub)
        pv = sk_fit_predict(kind, Fo[fs], tr & sub, va & sub); pt = sk_fit_predict(kind, Fo[fs], tr & sub, test & sub)
        OUT[f"oof_{m}"][va & sub] = pv; OUT[f"test_{m}"][test & sub] += pt / 5
        OUT[f"oof_cal_{m}"][va & sub] = calibrate_split(p_in, y[iva & sub], hi[iva & sub], pv, hi[va & sub])
        OUT[f"test_cal_{m}"][test & sub] += calibrate_split(p_in, y[iva & sub], hi[iva & sub], pt, hi[test & sub]) / 5
    # --- do thi kNN: chon k tren validation noi bo (dung do thi dung tu dac trung fit tren itr)
    gi = {kk: knn_graph(Fi["clin"][:, :-1], kk) for kk in KS}; si = probe(itr, Fi)
    Xi = {kk: np.column_stack([Fi["clin_glob"], nb_score(gi[kk], si)]) for kk in KS}
    sc = {}
    for kk in KS:
        sc[("L3b", kk)] = sel_score(y[iva], sk_fit_predict("lr", Xi[kk], itr, iva), hi[iva])
        sc[("B3b", kk)] = fit_gnet(Xi[kk], gi[kk], itr, iva, 0, False)[1]
        sc[("GAT", kk)] = fit_gnet(Fi["clin_glob"], gi[kk], itr, iva, 0, True)[1]
    go, so = None, probe(tr, Fo)
    for m in ("L3b", "B3b", "GAT"):
        kk = max(KS, key=lambda q: sc[(m, q)]); KSEL.append(dict(fold=k, mo_hinh=m, k_chon=kk, **{f"val_noi_bo_k{q}": sc[(m, q)] for q in KS}))
        Xin = np.column_stack([Fi["clin_glob"], nb_score(gi[kk], si)]) if m != "GAT" else Fi["clin_glob"]
        go = knn_graph(Fo["clin"][:, :-1], kk)
        Xo = np.column_stack([Fo["clin_glob"], nb_score(go, so)]) if m != "GAT" else Fo["clin_glob"]
        if m == "L3b":
            p_in = sk_fit_predict("lr", Xin, itr, iva); pv = sk_fit_predict("lr", Xo, tr, va); pt = sk_fit_predict("lr", Xo, tr, test)
        else:
            pin_s, pall_s = [], []
            for s in SEEDS:
                pin, _, ep = fit_gnet(Xin, gi[kk], itr, iva, s, m == "GAT"); pin_s.append(pin)
                pall_s.append(fit_gnet(Xo, go, tr, None, s, m == "GAT", epochs=ep)[0])
            p_in = np.mean(pin_s, 0)[iva]; pall = np.mean(pall_s, 0); pv, pt = pall[va], pall[test]
        OUT[f"oof_{m}"][va] = pv; OUT[f"test_{m}"][test] += pt / 5
        OUT[f"oof_cal_{m}"][va] = calibrate_split(p_in, y[iva], hi[iva], pv, hi[va])
        OUT[f"test_cal_{m}"][test] += calibrate_split(p_in, y[iva], hi[iva], pt, hi[test]) / 5
    torch.cuda.empty_cache()
    print(f"fold {k} xong {time.time()-t0:.0f}s", flush=True)

for m, spec in list(SK.items()) + [(m, (None, None, False)) for m in ("L3b", "B3b", "GAT")]:
    sub = hi if spec[2] else np.ones(N, bool)
    for t in ("test", "test_cal"): OUT[f"{t}_{m}"] = np.where(test & sub, OUT[f"{t}_{m}"], np.nan)
np.savez_compressed(CACHE / "run_baselines.npz", **OUT)
pd.DataFrame(KSEL).to_csv(CACHE / "k_chon_baselines.csv", index=False)
for m in ALL:
    o = OUT[f"oof_{m}"]; out = []
    for g, mm in (("co anh", dev & hi), ("khong anh", dev & ~hi)):
        mm = mm & np.isfinite(o)
        if mm.sum(): out.append(f"{g} {roc_auc_score(y[mm], o[mm]):.3f}")
    mm = dev & hi & np.isfinite(o)
    if mm.sum(): out.append(f"trong lo {within_batch_auc(y[mm], o[mm], B['batch'][mm]):.3f}")
    print(m, " | ".join(out))
print(pd.DataFrame(KSEL).groupby("mo_hinh").k_chon.value_counts().to_dict())

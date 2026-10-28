"""Viec (2) (PGS 27/10): chay lai baseline tren fold moi (dev2/fold2 trong cache_v3/p05_folds.npz, chuan duy nhat).
Anh = C (trung binh 6 vector vung BioViL-T tren anh V2) — chot o viec (1), user chon dung C cho ca L2/X2.
L0      LR lam sang (5 nhom + kham + tinh) + co phim
C       LR chi anh C, chi luot co phim — thay L1
L2      LR lam sang + C (0 neu khong phim) + co phim
        -> C cua LR chinh bang inner CV nhu viec (1): StratifiedGroupKFold(5) theo uid2 tren phan train fold ngoai,
           tieu chi ROC-AUC gop, luoi 1e-6..10. Mo hinh dung de hieu chinh (hoc tren fold noi bo) dung lai C da chon.
X0/X2   XGBoost lam sang / lam sang + C (sieu tham so co dinh nhu ban 2, 3 seed)
B3b     MLP tren [L2 + diem nguy co trung binh cua hang xom co phim tren do thi kNN]
GAT     GATv2 2 lop tren do thi kNN benh nhan (patient-as-node)
        -> k trong {5, 10, 20} chon tren validation noi bo (fold (k+1)%5 trong 4 fold train), chi dev.
Do thi kNN: cosin tren lam sang da ma hoa (fit trong fold) + canh cau noi moi nguoi khong phim -> nguoi co phim gan nhat.
Hieu chinh Platt trong fold (rieng nhom co / khong anh), nhu ban 2.
Test: chi luu du doan, KHONG cham diem. Dau ra cache_v3/run_baselines_v3.npz + chon_baselines_v3.csv. Chi in so lieu gop.
"""
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from xgboost import XGBClassifier
from v3_base import *
from v3_base import load_base_v3, CACHE3, IMG3, FOLDS, run_config

B = load_base_v3(); y, fold, hi, dev, test, uid = B["y"], B["fold"], B["has_img"], B["dev"], B["test"], B["uid"]
N = len(y); SEEDS = [0, 1, 2]; KS = [5, 10, 20]; GRID = np.logspace(-6, 1, 15)
xgb = lambda s, spw: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=2, scale_pos_weight=spw, tree_method="hist", random_state=s, n_jobs=8, verbosity=0)


def feats(trm):
    E = encode_fold(B, trm, None)
    clin = np.column_stack([E["flat_clin"], hi])
    G = B["glob_a"]; mu, sd = G[trm & hi].mean(0), G[trm & hi].std(0) + 1e-6
    g = ((G - mu) / sd * hi[:, None]).astype(np.float32)
    return dict(clin=clin, img=g, clin_img=np.column_stack([clin, g]))


# ---------------------------------------------------------------- LR co C chinh bang inner CV --
def lr_c(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=0))


def tune_C(X, rows):
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X[rows], y[rows], uid[rows]))
    gs = GridSearchCV(lr_c(0.1), {"logisticregression__C": GRID}, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X[rows], y[rows])
    return float(gs.best_params_["logisticregression__C"])


def lr_pred(X, trm, rows, C): return lr_c(C).fit(X[trm], y[trm]).predict_proba(X[rows])[:, 1]


def xgb_pred(X, trm, rows):
    return np.mean([xgb(s, (y[trm] == 0).sum() / y[trm].sum()).fit(X[trm], y[trm]).predict_proba(X[rows])[:, 1] for s in SEEDS], 0)


# ---------------------------------------------------------------- do thi kNN + diem hang xom (nhu ban 2) --
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


def probe(trm, Fe, C):
    """Diem anh: LR tren C (cung C cua mo hinh C fold nay) hoc tren hang TRAIN co phim -> cham moi nguoi co phim."""
    s = np.zeros(N); s[hi] = lr_pred(Fe["img"], trm & hi, hi, C); return s


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
SK = {"L0": ("lr", "clin", False), "C": ("lr", "img", True), "L2": ("lr", "clin_img", False),
      "X0": ("xgb", "clin", False), "X2": ("xgb", "clin_img", False)}
ALL = list(SK) + ["B3b", "GAT"]
OUT = {f"{t}_{m}": np.full(N, np.nan) for m in ALL for t in ("oof", "oof_cal")}
OUT.update({f"{t}_{m}": np.zeros(N) for m in ALL for t in ("test", "test_cal")})
SEL = []
t0 = time.time()


def store(m, va, sub, p_in, iva, pv, pt):
    OUT[f"oof_{m}"][va & sub] = pv; OUT[f"test_{m}"][test & sub] += pt / 5
    OUT[f"oof_cal_{m}"][va & sub] = calibrate_split(p_in, y[iva & sub], hi[iva & sub], pv, hi[va & sub])
    OUT[f"test_cal_{m}"][test & sub] += calibrate_split(p_in, y[iva & sub], hi[iva & sub], pt, hi[test & sub]) / 5


for k in range(5):
    tr, va = dev & (fold != k) & (fold >= 0), dev & (fold == k)
    inner = (k + 1) % 5; itr, iva = tr & (fold != inner), tr & (fold == inner)
    Fi, Fo = feats(itr), feats(tr)
    Cs = {}
    for m, (kind, fs, io) in SK.items():
        sub = hi if io else np.ones(N, bool)
        if kind == "lr":
            Cs[m] = C = tune_C(Fo[fs], tr & sub); SEL.append(dict(fold=k, mo_hinh=m, tham_so="C", gia_tri=C))
            p_in = lr_pred(Fi[fs], itr & sub, iva & sub, C)
            pv, pt = lr_pred(Fo[fs], tr & sub, va & sub, C), lr_pred(Fo[fs], tr & sub, test & sub, C)
        else:
            p_in, pv, pt = xgb_pred(Fi[fs], itr & sub, iva & sub), xgb_pred(Fo[fs], tr & sub, va & sub), xgb_pred(Fo[fs], tr & sub, test & sub)
        store(m, va, sub, p_in, iva, pv, pt)
    # --- do thi kNN: chon k tren validation noi bo (do thi dung tu dac trung fit tren itr)
    gi = {kk: knn_graph(Fi["clin"][:, :-1], kk) for kk in KS}; si = probe(itr, Fi, Cs["C"])
    sc = {}
    for kk in KS:
        sc[("B3b", kk)] = fit_gnet(np.column_stack([Fi["clin_img"], nb_score(gi[kk], si)]), gi[kk], itr, iva, 0, False)[1]
        sc[("GAT", kk)] = fit_gnet(Fi["clin_img"], gi[kk], itr, iva, 0, True)[1]
    so = probe(tr, Fo, Cs["C"]); allm = np.ones(N, bool)
    for m in ("B3b", "GAT"):
        kk = max(KS, key=lambda q: sc[(m, q)])
        SEL.append(dict(fold=k, mo_hinh=m, tham_so="k", gia_tri=kk, **{f"val_noi_bo_k{q}": sc[(m, q)] for q in KS}))
        go = knn_graph(Fo["clin"][:, :-1], kk)
        Xin = np.column_stack([Fi["clin_img"], nb_score(gi[kk], si)]) if m == "B3b" else Fi["clin_img"]
        Xo = np.column_stack([Fo["clin_img"], nb_score(go, so)]) if m == "B3b" else Fo["clin_img"]
        pin_s, pall_s = [], []
        for s in SEEDS:
            pin, _, ep = fit_gnet(Xin, gi[kk], itr, iva, s, m == "GAT"); pin_s.append(pin)
            pall_s.append(fit_gnet(Xo, go, tr, None, s, m == "GAT", epochs=ep)[0])
        pall = np.mean(pall_s, 0)
        store(m, va, allm, np.mean(pin_s, 0)[iva], iva, pall[va], pall[test])
    torch.cuda.empty_cache()
    print(f"fold {k} xong {time.time()-t0:.0f}s", flush=True)

for m, (_, _, io) in list(SK.items()) + [(m, (None, None, False)) for m in ("B3b", "GAT")]:
    sub = hi if io else np.ones(N, bool)
    for t in ("test", "test_cal"): OUT[f"{t}_{m}"] = np.where(test & sub, OUT[f"{t}_{m}"], np.nan)
np.savez_compressed(CACHE3 / "run_baselines_v3.npz", **OUT)
S = pd.DataFrame(SEL); S.to_csv(CACHE3 / "chon_baselines_v3.csv", index=False)
run_config("tuan1_phase0/viec2_baseline_fold_moi", __file__, data=[FOLDS, IMG3], phan="v3_baselines", SEEDS=SEEDS, KS=KS,
           luoi_C=GRID.tolist(), inner_cv="StratifiedGroupKFold(5, shuffle=True, random_state=SEED) theo uid2, ROC-AUC",
           xgb=dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=2),
           gnet=dict(hid=64, heads=4, drop=0.3, lr=3e-3, weight_decay=1e-3, epochs=300, patience=30))
for m in ALL:
    o = OUT[f"oof_{m}"]; out = []
    for g, mm in (("co anh", dev & hi), ("khong anh", dev & ~hi)):
        mm = mm & np.isfinite(o)
        if mm.sum(): out.append(f"{g} {roc_auc_score(y[mm], o[mm]):.3f}")
    mm = dev & hi & np.isfinite(o)
    out.append(f"trong lo {within_batch_auc(y[mm], o[mm], B['batch'][mm]):.3f}")
    print(m, " | ".join(out))
print(S.groupby(["mo_hinh", "tham_so"]).gia_tri.apply(list).to_dict())

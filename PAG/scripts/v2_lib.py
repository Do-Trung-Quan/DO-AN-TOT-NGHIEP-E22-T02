"""Thu vien chung cho PG ban 2 (PAG_V2, 04/10).

Khac ban 1:
- Nut boi canh chi con `tinh`. Cot `nam` trong Excel la nam CONG NHAN BNN (chi co o 211 ca bnn=Co),
  nen xep cung nhom voi bnn va bo khi chay PG-nobnn.
- Moi chi so anh bao them AUC trong tung lo chup (within-batch AUC).
- Moi mo hinh duoc hieu chinh xac suat (Platt / isotonic) trong fold truoc khi tinh Brier/ECE.
Cache chua id -> PII, KHONG dua len GitHub. Chi in/ghi so lieu gop.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings("ignore")

PAG = Path(__file__).resolve().parents[1]          # New Merged DATA/PAG
MERG = PAG.parent
BASE = MERG.parents[1]
FUS = BASE / "Fusion" / "DO-AN-TOT-NGHIEP-E22-T02" / "fusion" / "output"
CXR = BASE / "DO-AN-TOT-NGHIEP-E22-T02" / "imagefeat" / "output"
XLSX = MERG / "Main_data_Merged_1835_Images.xlsx"
CACHE = PAG / "cache"; CACHE.mkdir(exist_ok=True)          # chua id -> PII, KHONG dua len GitHub
RES = PAG / "results"; RES.mkdir(exist_ok=True)
SEED, NBOOT = 42, 1000
EXAM_SIGNS = {"rungt": "Rung thanh tăng", "rungg": "Rung thanh giảm", "goduc": "Gõ đục", "vtam": "Ran ẩm",
              "vtno": "Ran nổ", "vtrit": "Ran rít", "vtngay": "Ran ngáy"}
REG = ["RU", "RM", "RL", "LU", "LM", "LL"]


def load_core():
    """Bang cot loi (8030 hang, thu tu = file fold fusion): id, y, uid, fold, split, has_img, batch, bnn, tinh + 2 vector anh cu."""
    pk = CACHE / "core.pkl"
    if pk.exists():
        return pd.read_pickle(pk)
    D = np.load(FUS / "fusion_dataset_ketqua_frozen.npz", allow_pickle=False)
    Dc = np.load(FUS / "fusion_dataset_ketqua_control.npz", allow_pickle=False)
    xl = pd.read_excel(XLSX, usecols=["id", "tinh", "bnn", "nam", "ketqua", "matdotonthuong", "kichthuoctt", "chatluongphim"])
    xl = xl.set_index("id").loc[D["id"]].reset_index()
    core = pd.DataFrame(dict(id=D["id"], y=D["y"].astype(int), uid=D["patient_uid"], fold=D["fold_id"].astype(int),
                             split=D["split"], has_img=D["has_image"].astype(bool),
                             batch=np.where(D["has_image"].astype(bool), D["batch_date"], ""),
                             bnn=(xl["bnn"].astype(str) == "Co").astype(int).to_numpy(), nam_bnn=xl["nam"].to_numpy(),
                             tinh=xl["tinh"].astype(str).to_numpy(), matdo=xl["matdotonthuong"].astype(str).to_numpy(),
                             kichthuoc=xl["kichthuoctt"].astype(str).to_numpy(), chatluong=xl["chatluongphim"].to_numpy()))
    assert (core.y == (xl["ketqua"] == 1).astype(int)).all(), "nhan lech"
    core = add_person_merge(core)
    core["X_frozen"] = list(np.nan_to_num(D["X_img"].astype(np.float32)))
    core["X_control"] = list(np.nan_to_num(Dc["X_img"].astype(np.float32)))
    core.to_pickle(pk)
    return core


def _norm_name(s):
    import re, unicodedata
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d")); s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z ]", "", s).strip()


def add_person_merge(core):
    """P0.5: gop patient_uid bi tach (khoa long ten + nam sinh + gioi + tinh, union-find) -> uid2.
    Test giu nguyen; luot dev cua nguoi da co trong test bi loai (dev2=False); chia lai 5 fold dev theo uid2 -> fold2.
    Khoa ten chi dung cuc bo, khong luu."""
    from sklearn.model_selection import StratifiedGroupKFold
    xl = pd.read_excel(XLSX, usecols=["id", "hoten", "namsinh", "gioitinh", "tinh"]).set_index("id").loc[core.id]
    key = (xl.hoten.map(_norm_name) + "|" + xl.namsinh.astype(str) + "|" + xl.gioitinh.astype(str) + "|" + xl.tinh.astype(str)).to_numpy()
    par = {}
    def f(a):
        while par.setdefault(a, a) != a: par[a] = par[par[a]]; a = par[a]
        return a
    for u, k in zip(core.uid, key): par[f(("u", u))] = f(("k", k))
    roots = pd.factorize(pd.Series([f(("u", u)) for u in core.uid]).astype(str))[0]
    core["uid2"] = roots
    test_p = set(core.uid2[core.split == "test"])
    core["dev2"] = (core.split == "dev") & ~core.uid2.isin(test_p)
    core["fold2"] = -1
    d = core[core.dev2]
    strat = d.y * 2 + d.has_img.astype(int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(d, strat, d.uid2)):
        core.loc[d.index[va], "fold2"] = k
    return core


def lr(C=0.1, seed=0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed))


def oof_cv(X, y, folds, rows, make=lr):
    """OOF tren `rows` voi fold cho truoc (fold >= 0). Tra ve mang N (nan ngoai rows)."""
    p = np.full(len(y), np.nan)
    for k in sorted(set(folds[rows]) - {-1}):
        tr, va = rows & (folds != k) & (folds >= 0), rows & (folds == k)
        if y[tr].min() == y[tr].max() or va.sum() == 0: continue
        m = make().fit(X[tr], y[tr]); p[va] = m.predict_proba(X[va])[:, 1]
    return p


def within_batch_auc(y, p, batch):
    """AUC tinh rieng trong tung lo, trung binh co trong so co mau; bo lo chi co mot nhan."""
    df = pd.DataFrame(dict(y=y, p=p, b=batch))
    rows = []
    for b, g in df.groupby("b"):
        if g.y.nunique() < 2: continue
        rows.append((len(g), roc_auc_score(g.y, g.p)))
    if not rows: return np.nan
    w, a = np.array(rows).T
    return float((w * a).sum() / w.sum())


def boot_idx(uid, rng):
    """Bootstrap theo cum nguoi: rut lai nguoi, lay moi luot kham cua nguoi do."""
    u, inv = np.unique(uid, return_inverse=True)
    pick = rng.integers(0, len(u), len(u))
    cnt = np.bincount(pick, minlength=len(u))
    return np.repeat(np.arange(len(uid)), cnt[inv])


def boot_ci(fn, uid, n=NBOOT, seed=SEED):
    """fn(idx) -> so; tra (diem, lo, hi, mang bootstrap)."""
    rng = np.random.default_rng(seed); base = fn(np.arange(len(uid)))
    vals = []
    for _ in range(n):
        i = boot_idx(uid, rng)
        try: vals.append(fn(i))
        except ValueError: pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


# --------------------------------------------------------------- hieu chinh + chi so --
def calibrate(p_tr, y_tr, p_ap, method="platt"):
    """Fit tren du doan validation noi bo cua fold, ap len du doan can hieu chinh."""
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p_tr, y_tr).predict(p_ap)
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z(p_tr)[:, None], y_tr)
    return m.predict_proba(z(p_ap)[:, None])[:, 1]


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


def cal_slope_intercept(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def metrics(y, p, batch=None):
    r = dict(n=len(y), duong=int(y.sum()), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
             brier=brier_score_loss(y, p), brier_nen=float(y.mean() * (1 - y.mean())), ece=ece(y, p))
    r["do_doc_hc"], r["chan_hc"] = cal_slope_intercept(y, p)
    if batch is not None: r["auc_trong_lo"] = within_batch_auc(y, p, batch)
    return r


def md_table(df, nd=3, index=False):
    """Bang markdown khong can tabulate."""
    d = df.reset_index() if index else df
    fmt = lambda v: (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)).replace("|", "/")
    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(out)


class Report:
    """Ghi CSV + BAO_CAO.md vao PAG/results/<thu muc>."""
    def __init__(self, folder, title):
        self.d = RES / folder; self.d.mkdir(parents=True, exist_ok=True); self.lines = [f"# {title}\n"]
    def text(self, s): self.lines.append(s + "\n")
    def table(self, df, name, caption, note=None, nd=3, index=False):
        df.to_csv(self.d / f"{name}.csv", index=index, encoding="utf-8-sig")
        self.lines.append(f"### {caption}\n\n{md_table(df, nd, index)}\n\n_Tệp: `{name}.csv`_\n")
        if note: self.lines.append(note + "\n")
    def fig(self, fig, name, caption):
        fig.savefig(self.d / f"{name}.png", dpi=130, bbox_inches="tight"); self.lines.append(f"### {caption}\n\n![]({name}.png)\n")
    def save(self): (self.d / "BAO_CAO.md").write_text("\n".join(self.lines), encoding="utf-8")

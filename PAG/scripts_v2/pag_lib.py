"""Thu vien mo hinh PG ban 2 (PAG_V2 04/10).

Do thi moi benh nhan (dang day, 14 khe):
    0-5   node anh 6 vung ILO RU RM RL LU LM LL (chi luot co phim; dac trung SAU che chu + cat theo phoi)
    6-10  5 node lam sang: Nhan khau & the trang | Phoi nhiem/nghe | Hut thuoc | Trieu chung & tien su | Ho hap ky
    11    node kham thuc the — DANG DU PHONG (P0.6: moi o vung x dau hieu duong < 1% -> 1 node kham, khong canh E6)
    12    node boi canh: CHI `tinh` (cot `nam` la nam cong nhan BNN -> ro ri, khong dung)
    13    node benh nhan ao (CLS)
Canh: E1 doc trong mot phoi, E2 ngang cung tang, E3 giua cac node lam sang + kham, E4 lam sang/kham <-> vung, E5 CLS <-> moi node.
Fold = fold2 (5 fold dev chia lai theo nguoi da gop uid2), dev = dev2. Moi buoc hoc tu du lieu fit TRONG fold.
Cache chua id -> PII, KHONG dua len GitHub.
"""
import json, math, itertools
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from v2_lib import *

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_r = lambda p, n: [f"{p}{i}" for i in range(1, n + 1)]
GROUPS = {
    "Nhân khẩu & thể trạng": ["gioitinh", "namsinh", "cao", "can", "bmi", "A6", "hatd", "hatt", "mach", "theluc"],
    "Phơi nhiễm/nghề": ["cviec", "pxuong", "tuoinghe", "nampx", "cv5nam", "cviec1", "tgian1", "cviec2", "tgian2",
                        "A7", "A9a", "A9b", "A10", "A11", "A12", "F4mu", "F4ung", "F4KhauTrang", "F4gang", "F4BHLD", "F4Kinh"],
    "Hút thuốc": ["hutthuoc", "slthuoc"],
    "Triệu chứng & tiền sử": ["tiensuhh", "benhhh", "tsnoi", "tsngoai", "ho", "tdho", "tsho", "n1", "khacdom", "loaidom",
                              "tdkhacdo", "khotho", "mdkhotho", "tdkhotho", "daunguc", "vitridau", "daulan", "tcdau", "n2",
                              "tgdau", "ytodau", "n3", "ytotang", "chaymui", "khan", "khokhe", "dhkhac", "metmoi", "sutcan",
                              "socansut", "tgsut", "bnn"],
    "Hô hấp ký": ["fvctt", "fev1tt", "fev1_fvc"],
    "Khám thực thể": ["longnguc", "khoangls", "rungthan", *_r("rungt", 6), *_r("rungg", 6), "go", *_r("goduc", 6), "riraopn",
                      "ran", "ranam", "ranno", "ranrit", "ranngay", *_r("vtam", 6), *_r("vtno", 6), *_r("vtrit", 6), *_r("vtngay", 6)],
}
CTX = ["tinh"]
GNAMES = list(GROUPS)
NUMERIC_FORCE = {"bnn"}
NOT_USED = {
    "nam": "năm công nhận BNN — chỉ có ở ca bnn=Có → rò rỉ bnn; không dùng (bnn đã có trong nhóm Triệu chứng & tiền sử)",
    "matdotonthuong, kichthuoctt, chatluongphim, vungtonthuong, tonthuongkhac, ILO, ketqua": "cấm — đọc phim/nhãn",
    "fvclt, fev1lt, fvc, fev1, % dự đoán, gaenler": "cấm — chỉ dùng fvctt, fev1tt và FEV1/FVC tính lại",
    "mã lô, ngày chụp, tên tệp, thư mục": "cấm — dấu vết lô chụp",
    "B*, C*, D*, E*, F3, F6-F13": "không có từ điển mã, không có trong bảng ts",
    "hoten, sdt, sobh, bnncuthe": "PII",
}
MIN_FREQ, MAX_CAT = 20, 15
ENC = {"a": "biovil_frozen", "b": "xrv", "c": "biovil_control", "d": "chexfound"}


def load_base():
    core = load_core()
    raw = pd.read_pickle(MERG / "cache_pg" / "clin_raw.pkl").drop(columns=["nam"])
    h, w = pd.to_numeric(raw["cao"], errors="coerce"), pd.to_numeric(raw["can"], errors="coerce")
    h = h / 100                                                    # cao ghi bang cm
    raw["bmi"] = (w / h ** 2).where((h > 1.2) & (h < 2.1) & (w > 25) & (w < 150))
    raw["bnn"] = core["bnn"].to_numpy()
    Z = np.load(CACHE / "img_v2.npz", allow_pickle=False)
    pos = pd.Index(Z["img_id"]).get_indexer(core.id)
    has_img = core.has_img.to_numpy()
    assert ((pos >= 0) == has_img).all(), "anh/has_img lech"
    r = pos >= 0; N = len(core)
    def put(a):
        out = np.zeros((N,) + a.shape[1:], np.float32); out[r] = a[pos[r]].astype(np.float32); return out
    B = dict(y=core.y.to_numpy(), uid=core.uid2.to_numpy(), has_img=has_img, fold=core.fold2.to_numpy(),
             dev=core.dev2.to_numpy(), test=(core.split == "test").to_numpy(), batch=core.batch.to_numpy(),
             tinh=core.tinh.to_numpy(), raw=raw, area=put(Z["area"]))
    for k, e in ENC.items():
        B[f"reg_{k}"] = put(Z[f"regf_{e}"]); B[f"glob_{k}"] = put(Z[f"glob_{e}"])
    return B


# --------------------------------------------------------- ma hoa lam sang trong fold --
class ClinEncoder:
    """Fit tren hang `tr`: so -> dien trung vi + chi bao thieu + z-score; danh muc -> one-hot (tan suat >= 20, toi da 15 muc, NA la mot muc)."""
    def __init__(self, drop=()): self.drop = set(drop)
    def cols(self, g): return [c for c in (GROUPS[g] if g in GROUPS else CTX) if c not in self.drop]
    def fit(self, raw, tr):
        self.spec = {}
        for c in [c for g in GNAMES + ["Bối cảnh"] for c in self.cols(g)]:
            s = raw[c]; v = pd.to_numeric(s, errors="coerce")
            if c != "tinh" and (c in NUMERIC_FORCE or (v[tr].notna().sum() >= 0.95 * s[tr].notna().sum() and s[tr].notna().any())):
                vt = v[tr]; med = float(np.nanmedian(vt)) if vt.notna().any() else 0.0
                f = vt.fillna(med); mu, sd = float(f.mean()), float(f.std() or 1.0)
                self.spec[c] = ("num", med, mu, sd, bool(vt.isna().any()))
            else:
                st = s[tr].astype(str).where(s[tr].notna(), "__NA__")
                vc = st.value_counts(); self.spec[c] = ("cat", [k for k in vc.index if vc[k] >= MIN_FREQ][:MAX_CAT])
        return self

    def _col(self, raw, c):
        sp = self.spec[c]; s = raw[c]
        if sp[0] == "num":
            v = pd.to_numeric(s, errors="coerce"); m = v.isna().to_numpy()
            z = ((v.fillna(sp[1]) - sp[2]) / sp[3]).to_numpy(np.float32)
            return (np.column_stack([z, m]) if sp[4] else z[:, None]).astype(np.float32), [c] + ([f"{c}__thieu"] if sp[4] else [])
        st = s.astype(str).where(s.notna(), "__NA__").to_numpy()
        return (np.stack([st == k for k in sp[1]], 1).astype(np.float32) if sp[1] else np.zeros((len(s), 0), np.float32)), [f"{c}={k}" for k in sp[1]]

    def transform(self, raw):
        out, names, present = {}, {}, {}
        for g in GNAMES + ["Bối cảnh"]:
            cols = self.cols(g); parts = [self._col(raw, c) for c in cols]
            out[g] = np.concatenate([p[0] for p in parts], 1) if parts else np.zeros((len(raw), 1), np.float32)
            names[g] = sum([p[1] for p in parts], [])
            present[g] = raw[cols].notna().any(axis=1).to_numpy() if cols else np.zeros(len(raw), bool)
        return out, names, present


class ImgEncoder:
    """Chuan hoa + PCA whiten trong fold (fit tren hang co anh cua tap train, gop 6 vung)."""
    def __init__(self, k=128): self.k = k
    def fit(self, X, tr_img):
        Xt = X[tr_img].reshape(-1, X.shape[-1]).astype(np.float64)
        self.mu, self.sd = Xt.mean(0), Xt.std(0) + 1e-6
        self.pca = PCA(min(self.k, Xt.shape[1]), whiten=True, random_state=0, svd_solver="randomized").fit((Xt - self.mu) / self.sd)
        return self
    def transform(self, X):
        sh = X.shape; Z = self.pca.transform((X.reshape(-1, sh[-1]) - self.mu) / self.sd)
        return np.clip(Z, -10, 10).reshape(*sh[:-1], -1).astype(np.float32)


def encode_fold(B, tr, image="a", region=True, drop=()):
    """image: None | 'a' (BioViL-T dong bang) | 'b' (xrv) | 'c' (BioViL-T control) | 'd' (CheXFound); region=False -> 1 node anh toan cuc."""
    ce = ClinEncoder(drop).fit(B["raw"], tr)
    clin, names, present = ce.transform(B["raw"])
    hi = B["has_img"]; tri = tr & hi
    E = dict(clin=clin, names=names, present=present)
    if image is not None:
        if region:
            E["img"] = ImgEncoder(128).fit(B[f"reg_{image}"], tri).transform(B[f"reg_{image}"]) * hi[:, None, None]
            E["area"] = B["area"] * 10
        else:
            E["img"] = (ImgEncoder(128).fit(B[f"glob_{image}"], tri).transform(B[f"glob_{image}"]) * hi[:, None])[:, None, :]
            E["area"] = np.zeros((len(hi), 1), np.float32)
    E["flat_clin"] = np.concatenate([clin[g] for g in GNAMES + ["Bối cảnh"]], 1)
    return E


# ------------------------------------------------------------------- cau truc canh --
NONE, SELF, E1, E2, E3, E4, E5 = range(7)
N_ET = 7
E1_PAIRS = [(0, 1), (1, 2), (3, 4), (4, 5)]
E2_PAIRS = [(0, 3), (1, 4), (2, 5)]
REGION_PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]
CLIN_SLOTS = range(6, 12)          # 5 nhom + node kham (du phong)


def base_types(edges="anat", n_img=6):
    T = np.zeros((14, 14), np.int64)
    def s(i, j, t): T[i, j] = T[j, i] = t
    if edges == "complete":
        T[:] = SELF; return T
    for i in range(14): T[i, i] = SELF
    for i in range(13): s(13, i, E5)
    if edges == "noedge": return T
    if n_img == 6:
        for a, b in E1_PAIRS: s(a, b, E1)
        for a, b in E2_PAIRS: s(a, b, E2)
    for a in CLIN_SLOTS:
        for b in CLIN_SLOTS:
            if a != b: s(a, b, E3)
        for b in range(n_img): s(a, b, E4)
    return T


def shuffled_types(N, seed=0):
    """PG-shuffle: moi do thi 7 cap vung ngau nhien (4 nhan E1, 3 nhan E2) thay canh giai phau; giu so canh, cac lop khac giu nguyen."""
    rng = np.random.default_rng(seed)
    T = np.repeat(base_types("anat")[None], N, 0)
    for a, b in E1_PAIRS + E2_PAIRS: T[:, a, b] = T[:, b, a] = NONE
    for n in range(N):
        for t, p in zip([E1] * 4 + [E2] * 3, rng.choice(len(REGION_PAIRS), 7, replace=False)):
            a, b = REGION_PAIRS[p]; T[n, a, b] = T[n, b, a] = t
    return T


# ------------------------------------------------------------------------- mo hinh --
class GTLayer(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__(); s.h = h
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.eb = nn.Embedding(N_ET, h); nn.init.zeros_(s.eb.weight)
        s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        s.drop = nn.Dropout(drop)

    def forward(s, x, et, allow):
        B, n, d = x.shape; dh = d // s.h
        q, k, v = s.qkv(s.ln1(x)).view(B, n, 3, s.h, dh).permute(2, 0, 3, 1, 4)
        if et is None:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[:, None], dropout_p=s.drop.p if s.training else 0.0)
            x = x + s.drop(s.o(o.transpose(1, 2).reshape(B, n, d)))
            return x + s.drop(s.ff(s.ln2(x))), None
        a = q @ k.transpose(-1, -2) / math.sqrt(dh) + s.eb(et).permute(0, 3, 1, 2)
        a = a.masked_fill(~allow[:, None], float("-inf")).softmax(-1)
        x = x + s.drop(s.o((s.drop(a) @ v).transpose(1, 2).reshape(B, n, d)))
        return x + s.drop(s.ff(s.ln2(x))), a


class RGCNLayer(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__(); s.W = nn.Parameter(torch.randn(N_ET, d, d) / math.sqrt(d)); s.ln = nn.LayerNorm(d); s.drop = nn.Dropout(drop)
    def forward(s, x, et, allow):
        h = s.ln(x); out = 0
        for r in range(1, N_ET):
            A = ((et == r) & allow).float(); A = A / A.sum(-1, keepdim=True).clamp_min(1)
            out = out + A @ (h @ s.W[r])
        return x + s.drop(F.gelu(out)), None


class PGNet(nn.Module):
    def __init__(s, d_img, n_img, clin_dims, d=64, layers=2, drop=0.1, heads=4, arch="gt"):
        super().__init__(); s.n_img = n_img
        s.img = nn.Sequential(nn.Dropout(drop), nn.Linear(d_img + 1, d)); s.pos = nn.Embedding(7, d)
        s.clin = nn.ModuleList([nn.Sequential(nn.Linear(k, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d)) for k in clin_dims[:6]])
        s.ctx = nn.Linear(clin_dims[6], d)
        s.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(s.cls, std=0.02)
        L = GTLayer if arch == "gt" else RGCNLayer
        s.layers = nn.ModuleList([L(d, heads, drop) for _ in range(layers)])
        s.ln = nn.LayerNorm(d); s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(s, img, area, clin, present, et):
        B = img.shape[0]; d = s.cls.shape[-1]
        x = img.new_zeros(B, 14, d)
        pos_ids = torch.arange(s.n_img, device=img.device) if s.n_img == 6 else torch.tensor([6], device=img.device)
        x[:, :s.n_img] = s.img(torch.cat([img, area[..., None]], -1)) + s.pos(pos_ids)
        for g in range(6): x[:, 6 + g] = s.clin[g](clin[g])
        x[:, 12] = s.ctx(clin[6]); x[:, 13] = s.cls[:, 0]
        allow = (present[:, None, :] & (et != NONE)) | torch.eye(14, dtype=torch.bool, device=img.device)[None]
        att = None
        for L in s.layers: x, att = L(x, et, allow)
        h = s.ln(x[:, 13])
        return s.head(h)[:, 0], h, att


class FTT(nn.Module):
    """FT-Transformer: moi cot lam sang (sau ma hoa) = 1 token; T2 them 6 token vung anh; attention day du."""
    def __init__(s, n_feat, d_img, n_img, d=64, layers=2, drop=0.1, heads=4):
        super().__init__(); s.n_img = n_img
        s.w = nn.Parameter(torch.randn(n_feat, d) * 0.1); s.b = nn.Parameter(torch.zeros(n_feat, d))
        s.img = nn.Sequential(nn.Dropout(drop), nn.Linear(d_img + 1, d)); s.pos = nn.Embedding(6, d)
        s.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(s.cls, std=0.02)
        s.layers = nn.ModuleList([GTLayer(d, heads, drop) for _ in range(layers)])
        s.ln = nn.LayerNorm(d); s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(s, flat, img, area, img_present):
        B = flat.shape[0]
        toks = [flat[..., None] * s.w + s.b]; mask = [torch.ones(B, flat.shape[1], dtype=torch.bool, device=flat.device)]
        if s.n_img:
            toks.append(s.img(torch.cat([img, area[..., None]], -1)) + s.pos.weight[None]); mask.append(img_present[:, None].expand(B, 6))
        toks.append(s.cls.expand(B, 1, -1)); mask.append(torch.ones(B, 1, dtype=torch.bool, device=flat.device))
        x = torch.cat(toks, 1); m = torch.cat(mask, 1); n = x.shape[1]
        allow = m[:, None, :].expand(B, n, n) | torch.eye(n, dtype=torch.bool, device=x.device)[None]
        for L in s.layers: x, _ = L(x, None, allow)
        h = s.ln(x[:, -1])
        return s.head(h)[:, 0], h, None


# ----------------------------------------------------------------- bien the --
# PAG_V2 §8: PG-nobnn bo bnn VA cac bien tien su benh (tien su ho hap co/khong, benh ho hap, tien su noi/ngoai khoa —
# o dang chu tu do, co ghi chan doan bui phoi/BNN o mot so ca). PG-nobnn1 = chi bo bnn (ban dau).
HIST_VARS = ("bnn", "tiensuhh", "benhhh", "tsnoi", "tsngoai")
VARIANTS = {
    "PG":           dict(),
    "PG-noedge":    dict(edges="noedge"),
    "PG-complete":  dict(edges="complete"),
    "PG-shuffle":   dict(edges="shuffle"),
    "PG-global":    dict(region=False),
    "PG-noMD":      dict(md=False),
    "PG-noclin":    dict(only=("Khám thực thể",), use_ctx=False, rows="imaged"),
    "PG-noctx":     dict(use_ctx=False),
    "PG-RGCN":      dict(arch="rgcn"),
    "PG-nobnn1":    dict(drop=("bnn",)),
    "PG-nobnn":     dict(drop=HIST_VARS),
    "PG-enc-b":     dict(image="b"),
    "PG-enc-c":     dict(image="c"),
    "PG-enc-d":     dict(image="d"),
    "T0":           dict(image=None, arch="ftt"),
    "T2":           dict(arch="ftt"),
}


def variant(name):
    v = dict(image="a", region=True, edges="anat", use_ctx=True, md=True, arch="gt", rows="all", drop=(), only=None)
    v.update(VARIANTS[name]); return v


def to_tensors(E, B, v):
    N = len(B["y"]); t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=DEV)
    img, area = (E["img"], E["area"]) if v["image"] is not None else (np.zeros((N, 6, 1), np.float32), np.zeros((N, 6), np.float32))
    n_img = img.shape[1]
    pres = np.zeros((N, 14), bool)
    pres[:, :n_img] = B["has_img"][:, None] & (v["image"] is not None)
    for g, nm in enumerate(GNAMES): pres[:, 6 + g] = E["present"][nm] & (v["only"] is None or nm in v["only"])
    pres[:, 12] = v["use_ctx"]; pres[:, 13] = True
    if v["arch"] != "ftt":
        et = shuffled_types(N, 0) if v["edges"] == "shuffle" else np.broadcast_to(base_types(v["edges"], n_img), (N, 14, 14))
    else:
        et = np.zeros((1, 1, 1))
    clin = [E["clin"][g] for g in GNAMES + ["Bối cảnh"]]
    return dict(img=t(img), area=t(area), clin=[t(c) for c in clin], present=t(pres, torch.bool),
                et=t(np.ascontiguousarray(et), torch.long), flat=t(E["flat_clin"]), has_img=t(B["has_img"], torch.bool),
                y=t(B["y"]), n_img=n_img, d_img=img.shape[2], clin_dims=[c.shape[1] for c in clin])


def build_model(T, v, cfg):
    if v["arch"] == "ftt":
        return FTT(T["flat"].shape[1], T["d_img"], 0 if v["image"] is None else 6, cfg["dim"], cfg["so_lop"], cfg["dropout"]).to(DEV)
    return PGNet(T["d_img"], T["n_img"], T["clin_dims"], cfg["dim"], cfg["so_lop"], cfg["dropout"], arch=v["arch"]).to(DEV)


def forward(model, T, idx, v, drop_img=None):
    pres = T["present"][idx]
    if drop_img is not None: pres = pres.clone(); pres[:, :6] &= ~drop_img
    if v["arch"] == "ftt":
        return model(T["flat"][idx], T["img"][idx], T["area"][idx], pres[:, 0])
    et = T["et"][idx] if T["et"].shape[0] > 1 else T["et"][:1].expand(len(idx), 14, 14)
    return model(T["img"][idx], T["area"][idx], [c[idx] for c in T["clin"]], pres, et)


@torch.no_grad()
def predict(model, T, idx, v, drop_all_img=False, drop_region=None, bs=1024):
    model.eval(); P, H, A = [], [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]; d = None
        if drop_all_img: d = torch.ones(len(b), 6, dtype=torch.bool, device=DEV)
        elif drop_region is not None: d = drop_region[i:i + bs]
        lo, h, a = forward(model, T, b, v, d)
        P.append(torch.sigmoid(lo)); H.append(h)
        if a is not None: A.append(a.mean(1)[:, 13, :6])
    return torch.cat(P).cpu().numpy(), torch.cat(H).cpu().numpy(), (torch.cat(A).cpu().numpy() if A else None)


def sel_score(y, p, im):
    out = [roc_auc_score(y[m], p[m]) for m in (im, ~im) if m.sum() > 10 and 0 < y[m].sum() < m.sum()]
    return float(np.mean(out)) if out else float("nan")


def train(T, v, cfg, tr_idx, va_idx=None, epochs=80, seed=0, patience=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(T, v, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    y = T["y"][tr_idx]; pw = (len(y) - y.sum()) / y.sum().clamp_min(1)
    p_md = cfg["p_modality_dropout"] if v["md"] else 0.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_ep, best_state, bad = -1, epochs, None, 0
    if va_idx is not None: yv, hv = T["y"][va_idx].cpu().numpy(), T["has_img"][va_idx].cpu().numpy()
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g).to(DEV)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            d = (torch.rand(len(b), generator=g) < p_md).to(DEV)[:, None].expand(len(b), 6) if p_md > 0 else None
            lo, _, _ = forward(model, T, b, v, d)
            loss = F.binary_cross_entropy_with_logits(lo, T["y"][b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if va_idx is not None:
            sc = sel_score(yv, predict(model, T, va_idx, v)[0], hv)
            if sc > best + 1e-4: best, best_ep, bad = sc, ep, 0; best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, best, best_ep


SPACE = dict(dim=[64, 128], so_lop=[2, 3], dropout=[0.1, 0.3], lr=[3e-4, 1e-3], weight_decay=[1e-4, 1e-2], p_modality_dropout=[0.2, 0.3, 0.5])


def search_space(n=30, seed=42):
    grid = [dict(zip(SPACE, c)) for c in itertools.product(*SPACE.values())]
    rng = np.random.default_rng(seed)
    return [grid[i] for i in sorted(rng.choice(len(grid), n, replace=False))]


def calibrate_split(p_inner, y_inner, hi_inner, p_apply, hi_apply, method="platt"):
    """Hieu chinh trong fold, RIENG cho nhom co anh va khong anh (hai nhom co ti le duong khac nhau)."""
    out = np.full(len(p_apply), np.nan)
    for g in (True, False):
        mi, ma = hi_inner == g, hi_apply == g
        if ma.sum() == 0: continue
        if mi.sum() < 20 or y_inner[mi].min() == y_inner[mi].max(): out[ma] = p_apply[ma]; continue
        out[ma] = calibrate(p_inner[mi], y_inner[mi], p_apply[ma], method)
    return out

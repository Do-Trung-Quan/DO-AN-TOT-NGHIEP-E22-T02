"""Ham dung chung cho Phase 0 (P0.1, P0.4, P0.5, P0.6).

Quy uoc (PAG_V2 muc 3, 9, 11):
  - chi dung dev; test (fold_id == -1) khong bao gio vao phan tich
  - bootstrap theo patient_uid, seed co dinh, bao delta + KTC 95% + P(delta>0)
  - khong in ho ten, so dien thoai, ten file
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "phase0"
OUT = PHASE0 / "output"
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ------------------------------------------------------------------ config
def load_config() -> dict:
    path = PHASE0 / "config.json"
    raw = path.read_bytes()
    cfg = json.loads(raw.decode("utf-8"))
    cfg["_hash"] = hashlib.sha256(raw).hexdigest()[:16]
    return cfg


def lib_versions() -> dict:
    import sklearn
    v = {"python": sys.version.split()[0], "numpy": np.__version__,
         "pandas": pd.__version__, "sklearn": sklearn.__version__}
    try:
        import torch
        v["torch"] = torch.__version__
    except Exception:
        pass
    return v


# ------------------------------------------------------------------ du lieu
class Nodes:
    """Mot nguon anh (control|frozen) da can theo thu tu 8030 node."""

    def __init__(self, source: str, dataset_dir: Path | None = None,
                 features: Path | None = None, folds: Path | None = None,
                 fallback: Path | None = None, zone_missing: Path | None = None):
        ds_dir = dataset_dir or (ROOT / "fusion/output")
        d = np.load(ds_dir / f"fusion_dataset_ketqua_{source}.npz", allow_pickle=False)
        self.source = source
        self.id = d["id"].astype(np.int64)
        self.y = d["y"].astype(np.int64)
        self.uid = d["patient_uid"].astype(str)
        self.fold = d["fold_id"].astype(np.int64)
        self.split = d["split"].astype(str)
        self.folds_file = "fusion_dataset (file fold hien co)"
        if folds is not None:
            # parquet (id, split, fold_id[, group]) — giai doan 2 (P0.5 chia lai fold)
            fr = pd.read_parquet(folds).set_index("id").reindex(self.id)
            assert not fr["fold_id"].isna().any(), "file fold khong phu het 8030 id"
            self.fold = fr["fold_id"].to_numpy(np.int64)
            self.split = fr["split"].astype(str).to_numpy()
            if "group" in fr:
                self.uid = fr["group"].astype(str).to_numpy()
            self.folds_file = str(folds)
        self.has = d["has_image"].astype(bool)
        self.lot = d["batch_date"].astype(str)
        self.dev = self.split == "dev"
        # fold_id = -1 <=> test; hai cach tinh phai nhat quan
        assert ((self.fold == -1) == (self.split == "test")).all(), "split/fold_id lech nhau"
        X = d["X_img"].astype(np.float32)
        self.zone_missing = np.zeros(len(self.id), bool)
        if features is not None:
            X = align_features(features, self.id)
            if zone_missing is not None:
                # anh cham "hong" o QA: thay bang vector toan anh (V0), gan co, KHONG loai
                bad = pd.read_csv(zone_missing)["img_id"].to_numpy()
                assert fallback is not None, "--zone-missing can --fallback (embedding V0)"
                XF = align_features(fallback, self.id)
                self.zone_missing = np.isin(self.id, bad)
                X[self.zone_missing] = XF[self.zone_missing]
        assert not np.isnan(X[self.has]).any(), "co anh nhung thieu embedding"
        self.X_img = np.nan_to_num(X)
        self.ids_has = self.has

    @property
    def m(self) -> np.ndarray:
        """Mask phan tich P0.1: dev va co anh."""
        return self.dev & self.has


def align_features(path: Path, ids: np.ndarray) -> np.ndarray:
    """Doc parquet embedding (img_id + img_feat_*) -> mang (8030, D), NaN cho node khong anh."""
    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c.startswith("img_feat_")]
    cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    pos = pd.Series(np.arange(len(ids)), index=ids)
    out = np.full((len(ids), len(cols)), np.nan, np.float32)
    idx = pos.reindex(df["img_id"].to_numpy()).to_numpy()
    assert not np.isnan(idx).any(), "img_id khong co trong dataset"
    out[idx.astype(int)] = df[cols].to_numpy(np.float32)
    return out


def assert_no_test(mask: np.ndarray, nodes: Nodes) -> None:
    assert not (mask & ~nodes.dev).any(), "PHAN TICH DANG CHAM VAO TEST"


# ------------------------------------------------------------------ AUC trong lo
def lot_table(y, s, lot, min_pos=1, min_neg=1):
    """AUC rieng tung lo co du hai nhan. Tra ve (bang, auc co trong so theo co lo, so lo bi loai)."""
    rows, kept_w, kept_a = [], [], []
    dropped = []
    for L in sorted(np.unique(lot)):
        mm = lot == L
        npos, nneg = int(y[mm].sum()), int((1 - y[mm]).sum())
        ok = npos >= min_pos and nneg >= min_neg
        auc = float(roc_auc_score(y[mm], s[mm])) if ok else np.nan
        rows.append({"lo": L, "n": int(mm.sum()), "duong": npos, "am": nneg, "auc": auc})
        if ok:
            kept_w.append(mm.sum())
            kept_a.append(auc)
        else:
            dropped.append(L)
    w = np.array(kept_w, float)
    agg = float((w * np.array(kept_a)).sum() / w.sum()) if len(w) else np.nan
    return pd.DataFrame(rows), agg, dropped


def weighted_lot_auc(y, s, lot, min_pos=1, min_neg=1) -> float:
    num = den = 0.0
    for L in np.unique(lot):
        mm = lot == L
        p = y[mm].sum()
        n = mm.sum() - p
        if p >= min_pos and n >= min_neg:
            num += mm.sum() * roc_auc_score(y[mm], s[mm])
            den += mm.sum()
    return num / den if den else np.nan


def cluster_boot_indices(groups, n_boot, seed):
    """Sinh cac chi so lay mau lai THEO BENH NHAN (cung seed cho moi mo hinh -> ghep cap)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])


def boot_lot_auc(y, scores: dict, lot, groups, n_boot, seed, min_pos=1, min_neg=1,
                 pairs=()):
    """KTC 95% bootstrap theo benh nhan cho AUC trong lo (co trong so) cua tung mo hinh
    va cho cac chenh lech ghep cap. Moi lan lay mau tinh lai ca trung binh co trong so;
    lo chi con mot nhan trong mau lai bi bo cua lan do."""
    vals = {k: [] for k in scores}
    diffs = {p: [] for p in pairs}
    for idx in cluster_boot_indices(groups, n_boot, seed):
        cur = {}
        for k, s in scores.items():
            cur[k] = weighted_lot_auc(y[idx], s[idx], lot[idx], min_pos, min_neg)
            vals[k].append(cur[k])
        for a, b in pairs:
            diffs[(a, b)].append(cur[a] - cur[b])
    res = {}
    for k, v in vals.items():
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        res[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    dres = {}
    for p, v in diffs.items():
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        dres[p] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
                   float((v > 0).mean()))
    return res, dres


def decision_v3(auc_in_lot: float, delta_ci_low: float) -> str:
    """Bang quyet dinh P0.1 ban 3 (PG_update muc 2). Chi doi chieu; quyet dinh thuoc ve PGS."""
    if auc_in_lot < 0.60:
        return "dừng nhánh ảnh"
    if auc_in_lot >= 0.70 and delta_ci_low > 0:
        return "tiếp tục"
    if auc_in_lot >= 0.70:
        return "báo cáo PGS (AUC ≥ 0,70 nhưng KTC của Δ không nằm trọn trên 0)"
    return "báo cáo PGS trước khi làm tiếp"


def fmt_ci(v, ci, nd=3) -> str:
    return f"{v:.{nd}f} [{ci[0]:.{nd}f}; {ci[1]:.{nd}f}]"


def read_merged_xlsx(**kw) -> pd.DataFrame:
    """Doc Main_data_Merged_1835_Images.xlsx (8030 dong). Thu lai vai lan vi Windows/antivirus
    doi luc khoa file tam thoi (PermissionError)."""
    import time
    last = None
    for _ in range(5):
        try:
            return pd.read_excel(ROOT / "data/Main_data_Merged_1835_Images.xlsx", **kw)
        except PermissionError as e:
            last = e
            time.sleep(2)
    raise last


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv),
                    encoding="utf-8")

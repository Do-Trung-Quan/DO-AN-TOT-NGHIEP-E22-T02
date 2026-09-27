"""
================================================================================
PHASE 3 — BON BASELINE B0..B3 TRONG MOT DUONG ONG
================================================================================
Muc dich: dat 4 moc ve CUNG mot cach chia, cung kien truc, cung cach do — de con so
cua GNN o Phase 4 so duoc cong bang.

  B0  MLP tren dac trung lam sang               — bang tho 139 cot + diem nhanh ts
  B1  MLP tren anh (256 chieu), CHI 1835 node co anh
  B2  MLP tren lam sang + anh                   — hop nhat, KHONG dung do thi
  B3  MLP tren lam sang + anh + VECTOR ANH TRUNG BINH CUA HANG XOM (256 chieu)
  B3b MLP tren lam sang + anh + DIEM NGUY CO TRUNG BINH CUA HANG XOM (1 chieu)
      -> ca hai deu la phep "muon anh tu hang xom", khac o cho muon gi:
         B3 muon ca vector, B3b chi muon ket luan. B3b la phep kiem chung gia thuyet
         "bo control chay duoc vi no gan nhu mot truc diem nguy cro, con frozen 15 chieu
         da dang nen trung binh lai thanh mo".
         Ca hai KHONG phai GNN: hang xom duoc gop san, khong hoc trong so.

  L0/L1/L2  Hoi quy logistic tren cung dac trung — de biet mo hinh sau co dang dong
      hay khong. Da do: MLP chi anh 0,749 con LR chi anh 0,760. Neu mo hinh sau khong
      hon duoc mo hinh tuyen tinh thi phai noi thang trong bao cao.

GIA THUYET TRUNG TAM CUA DO AN nam o B3 so voi B0 tren nhom KHONG co anh:
neu thong tin anh truyen qua hang xom co ich thi B3 phai hon B0 o nhom do.

DAC TRUNG LAM SANG = 139 cot bang da ma hoa + 1 cot diem du doan cua nhanh ts.
KHONG dung vector 32 chieu `fusion_dense`: lay trung binh 5 model thi ro ri nhan o tap
dev (MLP dat ROC 0,752 so voi 0,641 that cua nhanh ts), con lay out-of-fold thi 5 fold
nam o 5 khong gian khac nhau (probe chi con ROC 0,487). Xem fusion/export_tabular.py.

NODE KHONG CO ANH
  Vector anh cua ho la NaN. Mo hinh thay bang mot "token thieu" HOC DUOC
  (nn.Parameter), kem co `has_image`. Khong dien 0 — dien 0 se dinh 6195 node
  thanh mot cum gia o giua khong gian dac trung.

GIAO THUC
  - Chi dung tap dev (6882 node). Tap test 1148 node KHONG dung o Phase 3,
    de danh cho Phase 6 cham dung mot lan.
  - 5 fold co san trong `fold_id` (nhom theo benh nhan) x 3 seed.
    Diem OOF = trung binh du doan cua 3 seed.
  - Chuan hoa fit tren phan train cua tung fold; anh fit tren rieng hang co anh.
  - KTC 95% bootstrap theo BENH NHAN.
  - Bao cao tach rieng: toan bo dev / nhom co anh / nhom khong anh.

CHAY
    python fusion/baselines.py                       # mac dinh nguon control
    python fusion/baselines.py --source frozen
================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOT = 1000


# ==============================================================================
# Chuan bi dac trung
# ==============================================================================
def neighbour_image_mean(edge_index: np.ndarray, X_img: np.ndarray,
                         has_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trung binh vector anh cua cac hang xom CO ANH. Khong dung nhan -> khong ro ri.

    Tra ve (dac trung hang xom, so hang xom co anh). Node khong cham duoc hang xom
    co anh nao se co vector 0 va so dem 0 -> mo hinh nhan ra qua token thieu.
    """
    # Do thi luu MOI CANH MOT CHIEU. Phai doi xung hoa truoc khi gop, neu khong moi node
    # chi thay duoc mot nua so hang xom (do duoc: 1,1 thay vi 3,23 hang xom co anh).
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    keep = has_image[dst]                       # chi gop tu hang xom CO anh
    src, dst = src[keep], dst[keep]
    total = np.zeros_like(X_img)
    count = np.zeros(len(X_img), dtype=np.float32)
    np.add.at(total, src, np.nan_to_num(X_img[dst]))
    np.add.at(count, src, 1.0)
    out = np.zeros_like(X_img)
    nz = count > 0
    out[nz] = total[nz] / count[nz, None]
    return out, count


def neighbour_score_mean(edge_index: np.ndarray, scores: np.ndarray,
                         has_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trung binh DIEM NGUY CO cua cac hang xom co anh (1 chieu) + so hang xom."""
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    keep = has_image[dst]
    src, dst = src[keep], dst[keep]
    total = np.zeros(len(scores), dtype=np.float64)
    count = np.zeros(len(scores), dtype=np.float64)
    np.add.at(total, src, scores[dst])
    np.add.at(count, src, 1.0)
    out = np.zeros(len(scores))
    nz = count > 0
    out[nz] = total[nz] / count[nz]
    return out, count


def image_probe_scores(train_imaged: np.ndarray, X_img: np.ndarray, y: np.ndarray,
                       has_image: np.ndarray) -> np.ndarray:
    """Diem nguy co tu anh, hoc tren rieng hang TRAIN co anh roi cham cho MOI node co anh.

    Node dang duoc danh gia khong bao gio nam trong tap train nay -> nhan cua no khong
    tham gia. Node train duoc cham "trong mau" nhung diem do chi dung lam DAC TRUNG cho
    hang xom, khong dung de cham diem chinh no.
    """
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    probe.fit(X_img[train_imaged], y[train_imaged])
    out = np.zeros(len(y))
    out[has_image] = probe.predict_proba(X_img[has_image])[:, 1]
    return out


def sk_oof(keys, maskable, imaged_only, rows, fold, RAW, VALID, y) -> np.ndarray:
    """Hoi quy logistic tren cung khoi dac trung — khoi vang mat dien 0 + them cot co/khong."""
    oof = np.zeros(len(y))
    for k in range(5):
        va, tr = rows & (fold == k), rows & (fold != k) & (fold >= 0)
        parts = []
        for key, can_miss in zip(keys, maskable):
            Z = standardise(tr, RAW[key], VALID[key])
            parts.append(Z)
            if can_miss:
                parts.append(VALID[key].astype(np.float32)[:, None])
        X = np.hstack(parts)
        mdl = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
        oof[va] = mdl.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof


def paired_gap(y, a, b, groups, n=400):
    """Chenh lech ROC-AUC giua hai mo hinh tren CUNG benh nhan + KTC 95% bootstrap."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    diffs = np.array(diffs)
    return (roc_auc_score(y, a) - roc_auc_score(y, b),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            float((diffs > 0).mean()))


def standardise(train_rows: np.ndarray, X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Chuan hoa bang trung binh/do lech chuan cua rieng hang TRAIN HOP LE."""
    sub = X[train_rows & valid]
    mu, sd = sub.mean(0), sub.std(0)
    sd[sd < 1e-8] = 1.0
    out = np.zeros_like(X)
    out[valid] = (X[valid] - mu) / sd
    return out


# ==============================================================================
# Mo hinh
# ==============================================================================
class Net(nn.Module):
    """MLP dung chung cho ca 4 baseline.

    Moi khoi dac trung co the vang mat (anh, hang xom) -> moi khoi vang duoc thay
    bang mot token HOC DUOC thay vi dien 0.
    """

    def __init__(self, dims: list[int], maskable: list[bool], hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.dims, self.maskable = dims, maskable
        self.tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) if m else nn.Parameter(torch.zeros(0), requires_grad=False)
             for d, m in zip(dims, maskable)])
        total = sum(dims) + sum(maskable)        # + co bao "khoi nay co mat khong"
        self.body = nn.Sequential(
            nn.Linear(total, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, blocks: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        parts = []
        for x, m, tok, can_miss in zip(blocks, masks, self.tokens, self.maskable):
            if can_miss:
                x = torch.where(m.unsqueeze(1), x, tok.expand_as(x))
                parts += [x, m.float().unsqueeze(1)]
            else:
                parts.append(x)
        return self.body(torch.cat(parts, dim=1)).squeeze(1)


def train_predict(blocks_tr, masks_tr, y_tr, blocks_va, masks_va, maskable,
                  seed: int, epochs: int, hidden: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net([b.shape[1] for b in blocks_tr], maskable, hidden=hidden).to(DEVICE)
    pos = float(y_tr.sum())
    pos_weight = torch.tensor([(len(y_tr) - pos) / max(pos, 1.0)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bt = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in blocks_tr]
    mt = [torch.tensor(m, device=DEVICE) for m in masks_tr]
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    n = len(y_tr)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for a in range(0, n, 256):
            idx = torch.tensor(order[a:a + 256], device=DEVICE)
            opt.zero_grad()
            loss = loss_fn(model([b[idx] for b in bt], [m[idx] for m in mt]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        bv = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in blocks_va]
        mv = [torch.tensor(m, device=DEVICE) for m in masks_va]
        return torch.sigmoid(model(bv, mv)).cpu().numpy()


# ==============================================================================
# Do luong
# ==============================================================================
def boot_ci(y, s, groups, fn, n=N_BOOT):
    rng = np.random.default_rng(42)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score_block(y, s, groups) -> dict:
    return {
        "n": int(len(y)), "duong": int(y.sum()), "ty_le": float(y.mean()),
        "roc": float(roc_auc_score(y, s)), "roc_ci": boot_ci(y, s, groups, roc_auc_score),
        "pr": float(average_precision_score(y, s)),
        "pr_ci": boot_ci(y, s, groups, average_precision_score),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="control", choices=["control", "frozen"])
    p.add_argument("--graph", type=Path, default=ROOT / "fusion/output/graph_ketqua_k10_bridge.npz")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--out-dir", type=Path, default=ROOT / "fusion/output",
                   help="Noi ghi ket qua. Dung thu muc tam khi chay thu de khong de len ket qua that")
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()

    ds_path = ROOT / f"fusion/output/fusion_dataset_ketqua_{args.source}.npz"
    d = np.load(ds_path, allow_pickle=False)
    g = np.load(args.graph, allow_pickle=False)
    tab = np.load(ROOT / "fusion/output/tabular_ketqua.npz", allow_pickle=False)
    assert (tab["id"] == d["id"]).all(), "tabular_ketqua.npz lech thu tu hang"
    ts_pred_all = pd.read_parquet(ROOT / "output/ketqua/ts_predictions.parquet")
    assert (ts_pred_all["id"].to_numpy() == d["id"]).all()
    # Lam sang = bang tho (sach tuyet doi) + diem nhanh ts (1 chieu, cung nghia moi fold)
    X_ls = np.hstack([tab["X_tab"],
                      ts_pred_all["score"].to_numpy(np.float32)[:, None]]).astype(np.float32)
    X_img_raw = d["X_img"].astype(np.float32)
    has_img = d["has_image"].astype(bool)
    y = d["y"].astype(np.float32)
    uid = d["patient_uid"]
    dev = d["split"] == "dev"
    fold = d["fold_id"]

    print("=" * 78)
    print(f"PHASE 3 — BASELINE B0..B3   [nguon anh: {args.source} | {DEVICE}]")
    print("=" * 78)
    print(f"dev {int(dev.sum())} node ({int(y[dev].sum())} duong) | test {int((~dev).sum())} node — "
          f"KHONG dung o phase nay")

    X_nb, n_nb = neighbour_image_mean(g["edge_index"], X_img_raw, has_img)
    has_nb = n_nb > 0
    print(f"Hang xom co anh: trung binh {n_nb[dev].mean():.2f}/node | "
          f"{100 * has_nb[dev].mean():.1f}% node cham duoc it nhat 1 hang xom co anh")

    # (ten, cac khoi dac trung, khoi nao co the vang mat, chi chay tren node co anh)
    CONFIGS = {
        "B0 — chi lam sang":                    (["ls"], [False], False, "mlp"),
        "B1 — chi anh (node co anh)":           (["img"], [False], True, "mlp"),
        "B2 — lam sang + anh":                  (["ls", "img"], [False, True], False, "mlp"),
        "B3 — + vector anh hang xom":           (["ls", "img", "nb"], [False, True, True], False, "mlp"),
        "B3b — + diem nguy co hang xom":        (["ls", "img", "nbs"], [False, True, True], False, "mlp"),
        "L1 — hoi quy logistic, chi anh":       (["img"], [False], True, "lr"),
        "L2 — hoi quy logistic, lam sang+anh":  (["ls", "img"], [False, True], False, "lr"),
        "L3b — hoi quy logistic, + diem hang xom": (["ls", "img", "nbs"], [False, True, True], False, "lr"),
    }
    RAW = {"ls": X_ls, "img": np.nan_to_num(X_img_raw), "nb": X_nb}
    VALID = {"ls": np.ones(len(y), bool), "img": has_img, "nb": has_nb}

    X_img_filled = np.nan_to_num(X_img_raw)
    results, oof_store = {}, {}
    for name, (keys, maskable, imaged_only, kind) in CONFIGS.items():
        rows = dev & has_img if imaged_only else dev
        oof = np.zeros(len(y))
        for k in range(5):
            va = rows & (fold == k)
            tr = rows & (fold != k) & (fold >= 0)

            # Dac trung "diem hang xom" phu thuoc fold -> tinh lai trong tung fold.
            # Probe chi hoc tren hang TRAIN co anh, nen nhan cua node dang danh gia
            # khong bao gio tham gia.
            if "nbs" in keys:
                s_img = image_probe_scores(tr & has_img, X_img_filled, y, has_img)
                nbs, cnt = neighbour_score_mean(g["edge_index"], s_img, has_img)
                RAW["nbs"] = np.column_stack([nbs, np.log1p(cnt)]).astype(np.float32)
                VALID["nbs"] = cnt > 0

            if kind == "lr":
                continue   # LR chay mot lan cho ca 5 fold ben duoi

            blocks_tr, blocks_va, masks_tr, masks_va = [], [], [], []
            for key in keys:
                Z = standardise(tr, RAW[key], VALID[key])
                blocks_tr.append(Z[tr]); blocks_va.append(Z[va])
                masks_tr.append(VALID[key][tr]); masks_va.append(VALID[key][va])
            preds = [train_predict(blocks_tr, masks_tr, y[tr], blocks_va, masks_va,
                                   maskable, seed, args.epochs, args.hidden)
                     for seed in range(args.seeds)]
            oof[va] = np.mean(preds, axis=0)

        if kind == "lr":
            oof = sk_oof(keys, maskable, imaged_only, rows, fold, RAW, VALID, y)
        oof_store[name] = (oof, rows)

        res = {"toan_bo_dev": score_block(y[rows], oof[rows], uid[rows])}
        if not imaged_only:
            for tag, m in (("co_anh", rows & has_img), ("khong_anh", rows & ~has_img)):
                res[tag] = score_block(y[m], oof[m], uid[m])
        results[name] = res
        print(f"  [xong] {name}")

    # Tham chieu: diem cua nhanh ts (khong phai MLP o day)
    s_ref = ts_pred_all["score"].to_numpy()
    ref = {"toan_bo_dev": score_block(y[dev], s_ref[dev], uid[dev]),
           "co_anh": score_block(y[dev & has_img], s_ref[dev & has_img], uid[dev & has_img]),
           "khong_anh": score_block(y[dev & ~has_img], s_ref[dev & ~has_img], uid[dev & ~has_img])}
    results["(tham chieu) nhanh ts goc"] = ref

    # ---------------- bao cao ----------------
    L = []
    def out(t=""):
        print(t); L.append(t)

    out(f"# Phase 3 — baseline B0..B3  (nguon anh `{args.source}`)")
    out()
    out(f"- Sinh luc: {datetime.now(timezone.utc).isoformat()} | thiet bi: {DEVICE}")
    out(f"- Dac trung lam sang: bang tho {tab['X_tab'].shape[1]} cot + diem nhanh ts")
    out(f"- Tap dev {int(dev.sum())} node, {int(y[dev].sum())} ca duong. Test 1148 node chua dung.")
    out(f"- 5 fold co san x {args.seeds} seed, {args.epochs} epoch; KTC 95% bootstrap theo benh nhan")
    out()
    for tag, title in (("toan_bo_dev", "Toan bo dev"), ("co_anh", "Nhom CO anh"),
                       ("khong_anh", "Nhom KHONG anh")):
        out(f"## {title}")
        out()
        out("| Mo hinh | n | duong | ROC-AUC | PR-AUC |")
        out("|---|---|---|---|---|")
        for name, res in results.items():
            if tag not in res:
                continue
            r = res[tag]
            out(f"| {name} | {r['n']} | {r['duong']} | {r['roc']:.3f} "
                f"[{r['roc_ci'][0]:.3f}–{r['roc_ci'][1]:.3f}] | {r['pr']:.3f} "
                f"[{r['pr_ci'][0]:.3f}–{r['pr_ci'][1]:.3f}] |")
        out()

    # ---------------- cong ----------------
    # Cong dua tren KHOANG TIN CAY chu khong so hai so tran: chenh 0,005 giua hai mo hinh
    # tren 393 ca duong la nhieu, khong phai khac biet.
    mi, mu = dev & has_img, dev & ~has_img
    O = {k: v[0] for k, v in oof_store.items()}
    out("## Cong Phase 3 — xet bang khoang tin cay, khong so so tran")
    out()
    out("| Cong | Chenh lech ROC | KTC 95% | P | Ket qua |")
    out("|---|---|---|---|---|")
    checks = [
        ("Anh + lam sang co kem hon chi anh khong (nhom co anh)",
         O["B2 — lam sang + anh"], O["B1 — chi anh (node co anh)"], mi, "khong_kem"),
        ("Hang xom co giup nhom KHONG anh khong (B3)",
         O["B3 — + vector anh hang xom"], O["B0 — chi lam sang"], mu, "phai_hon"),
        ("Hang xom co giup nhom KHONG anh khong (B3b)",
         O["B3b — + diem nguy co hang xom"], O["B0 — chi lam sang"], mu, "phai_hon"),
        ("Mo hinh sau co hon hoi quy logistic khong (nhom co anh)",
         O["B2 — lam sang + anh"], O["L2 — hoi quy logistic, lam sang+anh"], mi, "khong_kem"),
    ]
    gates = {}
    for title, a, b, mask, rule in checks:
        gap, lo, hi, p = paired_gap(y[mask], a[mask], b[mask], uid[mask])
        ok = (lo > 0) if rule == "phai_hon" else (hi > 0)
        gates[title] = ok
        out(f"| {title} | {gap:+.4f} | [{lo:+.4f}; {hi:+.4f}] | {p:.0%} | "
            f"{'DAT' if ok else '**TRUOT**'} |")
    out()
    out("*`phai_hon` = KTC phai nam tron ben duong. `khong_kem` = chi truot khi KTC "
        "nam tron ben am, tuc thua ro rang.*")
    out()
    best_u = max(("B3 — + vector anh hang xom", "B3b — + diem nguy co hang xom",
                  "L3b — hoi quy logistic, + diem hang xom"),
                 key=lambda n: results[n]["khong_anh"]["roc"])
    best_i = max(("B2 — lam sang + anh", "B1 — chi anh (node co anh)",
                  "L2 — hoi quy logistic, lam sang+anh"),
                 key=lambda n: results[n].get("co_anh", results[n]["toan_bo_dev"])["roc"])
    ru = results[best_u]["khong_anh"]["roc"]
    ri = results[best_i].get("co_anh", results[best_i]["toan_bo_dev"])["roc"]
    out(f"**Moc cho Phase 4 (GNN phai vuot):** nhom KHONG anh **{ru:.3f}** ({best_u}) | "
        f"nhom CO anh **{ri:.3f}** ({best_i})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Ten file gan lien voi nguon anh: chay `frozen` khong bao gio de len ket qua `control`
    (args.out_dir / f"phase3_baselines_{args.source}.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")
    (args.out_dir / f"phase3_baselines_{args.source}.json").write_text(
        json.dumps({"source": args.source, "epochs": args.epochs, "seeds": args.seeds,
                    "ket_qua": results}, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    np.savez_compressed(args.out_dir / f"phase3_oof_{args.source}.npz",
                        **{k.split(" ")[0]: v[0] for k, v in oof_store.items()})
    print(f"\nDa ghi -> {args.out_dir}/phase3_baselines_{args.source}.{{md,json}}"
          f" + phase3_oof_{args.source}.npz")


if __name__ == "__main__":
    main()

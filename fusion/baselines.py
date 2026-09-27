"""
================================================================================
PHASE 3 — BON BASELINE B0..B3 TRONG MOT DUONG ONG
================================================================================
Muc dich: dat 4 moc ve CUNG mot cach chia, cung kien truc, cung cach do — de con so
cua GNN o Phase 4 so duoc cong bang.

  B0  MLP tren dac trung lam sang               — bang tho 139 cot + diem nhanh ts
  B1  MLP tren anh (256 chieu), CHI 1835 node co anh
  B2  MLP tren lam sang + anh                   — hop nhat, KHONG dung do thi
  B3  MLP tren lam sang + anh + DAC TRUNG ANH TRUNG BINH CUA HANG XOM
      -> chinh la phep "muon anh tu hang xom" nhung dua vao duong ong chuan.
         B3 KHONG phai GNN: hang xom duoc gop san mot lan, khong hoc trong so.

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
from sklearn.metrics import average_precision_score, roc_auc_score

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
    src, dst = edge_index
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
        "B0 — chi lam sang":                (["ls"], [False], False),
        "B1 — chi anh (node co anh)":       (["img"], [False], True),
        "B2 — lam sang + anh":              (["ls", "img"], [False, True], False),
        "B3 — lam sang + anh + hang xom":   (["ls", "img", "nb"], [False, True, True], False),
    }
    RAW = {"ls": X_ls, "img": np.nan_to_num(X_img_raw), "nb": X_nb}
    VALID = {"ls": np.ones(len(y), bool), "img": has_img, "nb": has_nb}

    results, oof_store = {}, {}
    for name, (keys, maskable, imaged_only) in CONFIGS.items():
        rows = dev & has_img if imaged_only else dev
        oof = np.zeros(len(y))
        for k in range(5):
            va = rows & (fold == k)
            tr = rows & (fold != k) & (fold >= 0)
            blocks_tr, blocks_va, masks_tr, masks_va = [], [], [], []
            for key in keys:
                Z = standardise(tr, RAW[key], VALID[key])
                blocks_tr.append(Z[tr]); blocks_va.append(Z[va])
                masks_tr.append(VALID[key][tr]); masks_va.append(VALID[key][va])
            preds = [train_predict(blocks_tr, masks_tr, y[tr], blocks_va, masks_va,
                                   maskable, seed, args.epochs, args.hidden)
                     for seed in range(args.seeds)]
            oof[va] = np.mean(preds, axis=0)
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
    b1 = results["B1 — chi anh (node co anh)"]["toan_bo_dev"]["roc"]
    b2i = results["B2 — lam sang + anh"]["co_anh"]["roc"]
    b0u = results["B0 — chi lam sang"]["khong_anh"]["roc"]
    b3u = results["B3 — lam sang + anh + hang xom"]["khong_anh"]["roc"]
    gate1, gate2 = b2i >= b1 - 0.005, b3u > b0u
    out("## Cong Phase 3")
    out()
    out(f"| Cong | Do duoc | Ket qua |")
    out("|---|---|---|")
    out(f"| B2 >= B1 tren nhom co anh | {b2i:.3f} vs {b1:.3f} | {'DAT' if gate1 else '**TRUOT**'} |")
    out(f"| B3 > B0 tren nhom khong anh | {b3u:.3f} vs {b0u:.3f} | {'DAT' if gate2 else '**TRUOT**'} |")
    out()
    out(f"**Moc cho Phase 4:** GNN phai vuot B3 o nhom khong anh (**{b3u:.3f}**) "
        f"va vuot B2 o nhom co anh (**{b2i:.3f}**).")

    (ROOT / "fusion/output/phase3_baselines.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "fusion/output/phase3_baselines.json").write_text(
        json.dumps({"source": args.source, "epochs": args.epochs, "seeds": args.seeds,
                    "ket_qua": results}, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    np.savez_compressed(ROOT / f"fusion/output/phase3_oof_{args.source}.npz",
                        **{k.split(" ")[0]: v[0] for k, v in oof_store.items()})
    print("\nDa ghi -> fusion/output/phase3_baselines.{md,json} + phase3_oof_*.npz")


if __name__ == "__main__":
    main()

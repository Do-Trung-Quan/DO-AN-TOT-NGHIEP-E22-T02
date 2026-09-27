"""
================================================================================
PHASE 4 — GAT (Graph Attention Network) TREN DO THI BENH NHAN
================================================================================
Y TUONG
  Phase 3 da chung minh: cho nguoi KHONG co phim, muon thong tin anh cua hang xom
  thi co ich. Nhung B3/B3b gop hang xom bang TRUNG BINH CONG — ai cung nghe hang xom
  nhu nhau, ke ca nguoi da co phim cua chinh minh (va do la ly do B3 kem B2 o nhom
  co anh).

  GAT hoc TRONG SO CHU Y cho tung canh: hang xom nao dang tin thi nghe nhieu, khong
  dang tin thi nghe it. Ket hop them duong tat (residual) tu dac trung goc, mo hinh
  co the TU BO QUA do thi voi nhung node da du thong tin.

  => GAT chi dang dung neu vuot duoc B3/B3b o nhom khong anh VA khong thua B2 o nhom
     co anh. Neu khong, ket luan trung thuc la "co che chu y khong giup gi o day".

DAC TRUNG DAU VAO — giong B3b cua Phase 3
  lam sang (bang tho + diem nhanh ts) | anh (256 chieu, token thieu neu vang)
  | DIEM NGUY CO TRUNG BINH CUA HANG XOM (1 chieu + so hang xom).
  Khoi thu ba la thu DA THANG o Phase 3 (B3b: +0,022 ROC o nhom khong anh, P=99%).
  Dua no vao thang lam dau vao thay vi bat GAT tu hoc lai tu vector tho.
  Tat bang --no-nb-score de do rieng dong gop cua no.

KHONG DUNG torch_geometric
  Lop GATv2 duoc cai bang torch thuan (~40 dong) de tranh rui ro cai dat va de chay
  duoc ca o may khong co GPU. Cong thuc theo Brody et al., ICLR 2022 (GATv2):
  phi tuyen dat TRUOC tich vo huong voi vector chu y, khac GAT goc.

GIAO THUC — GIONG HET PHASE 3 de so sanh cong bang
  - Chi tap dev (6882 node); test 1148 node de danh cho Phase 6.
  - 5 fold co san x nhieu seed; diem OOF = trung binh cac seed.
  - Truyen tin tren TOAN BO do thi (transductive) nhung loss chi tinh tren node train
    cua fold do. Nhan cua node validation khong bao gio tham gia huan luyen.
  - LayerNorm chu khong BatchNorm: o che do transductive, moi luot truyen tien chay
    tren CA 8030 node nen BatchNorm se tinh thong ke tren ca node validation.
  - Early stopping: tach 15% benh nhan trong tap TRAIN lam tap theo doi, dung khi
    PR-AUC tren do khong cai thien sau `--patience` epoch. Node validation cua fold
    KHONG bao gio duoc dung de dung som.
  - KTC 95% bootstrap theo benh nhan; bao cao tach nhom co anh / khong anh.
  - Tu doc fusion/output/phase3_baselines_<nguon>.json de doi chieu thang voi Phase 3.

CHAY
    python fusion/gnn.py --source control                 # mac dinh
    python fusion/gnn.py --source control --no-graph      # ablation: tat truyen tin
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.baselines import (image_probe_scores, neighbour_score_mean,  # noqa: E402
                              paired_gap, score_block, standardise)
from sklearn.metrics import average_precision_score  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Lop GATv2 tu cai
# ==============================================================================
class GATv2Layer(nn.Module):
    """Mot lop chu y tren do thi. edge_index phai da doi xung + co self-loop."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.h, self.c = heads, out_dim
        self.lin_src = nn.Linear(in_dim, heads * out_dim)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim)
        self.att = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.att)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        xs = self.lin_src(x).view(n, self.h, self.c)
        xd = self.lin_dst(x).view(n, self.h, self.c)

        # GATv2: LeakyReLU TRUOC roi moi nhan voi vector chu y
        e = F.leaky_relu(xs[src] + xd[dst], 0.2)
        alpha = (e * self.att).sum(-1)                        # (E, heads)

        # softmax theo tung node dich, tru max de khong tran so
        amax = torch.full((n, self.h), float("-inf"), device=x.device)
        amax.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.h), alpha,
                             reduce="amax", include_self=True)
        alpha = (alpha - amax[dst]).exp()
        denom = torch.zeros((n, self.h), device=x.device).index_add_(0, dst, alpha)
        alpha = alpha / (denom[dst] + 1e-16)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((n, self.h, self.c), device=x.device)
        out.index_add_(0, dst, xs[src] * alpha.unsqueeze(-1))
        return out.reshape(n, self.h * self.c)


class GATNet(nn.Module):
    """Bo ma hoa tung phuong thuc -> 2 lop GAT -> dau ra, co duong tat bo qua do thi."""

    def __init__(self, dims: list[int], maskable: list[bool], hidden: int = 64,
                 heads: int = 4, dropout: float = 0.3, use_graph: bool = True):
        super().__init__()
        self.maskable, self.use_graph = maskable, use_graph
        self.tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) if m else nn.Parameter(torch.zeros(0), requires_grad=False)
             for d, m in zip(dims, maskable)])
        # LayerNorm chu khong BatchNorm: xem giai thich o docstring
        self.enc = nn.Sequential(
            nn.Linear(sum(dims) + sum(maskable), hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout))
        self.g1 = GATv2Layer(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.g2 = GATv2Layer(hidden, hidden, heads=1, dropout=dropout)
        # Dau ra nhan CA dac trung goc lan dac trung sau truyen tin -> mo hinh tu quyet
        # dinh nghe do thi bao nhieu; voi node da co phim no co the bo qua hang xom.
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 if use_graph else hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, blocks, masks, edge_index) -> torch.Tensor:
        parts = []
        for x, m, tok, can_miss in zip(blocks, masks, self.tokens, self.maskable):
            if can_miss:
                parts += [torch.where(m.unsqueeze(1), x, tok.expand_as(x)), m.float().unsqueeze(1)]
            else:
                parts.append(x)
        h = self.enc(torch.cat(parts, dim=1))
        if not self.use_graph:
            return self.head(h).squeeze(1)
        z = F.elu(self.g1(h, edge_index))
        z = F.elu(self.g2(z, edge_index))
        return self.head(torch.cat([h, z], dim=1)).squeeze(1)


# ==============================================================================
def symmetric_with_self_loops(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Do thi luu moi canh mot chieu -> phai doi xung hoa, roi them self-loop."""
    src = np.concatenate([edge_index[0], edge_index[1], np.arange(n)])
    dst = np.concatenate([edge_index[1], edge_index[0], np.arange(n)])
    return np.stack([src, dst])


def split_inner(train_idx: np.ndarray, groups: np.ndarray, seed: int, frac: float = 0.15):
    """Tach mot phan tap train lam tap theo doi de dung som — tach THEO BENH NHAN."""
    rng = np.random.default_rng(1000 + seed)
    uniq = rng.permutation(np.unique(groups[train_idx]))
    watch = set(uniq[: max(1, int(len(uniq) * frac))].tolist())
    is_watch = np.array([g in watch for g in groups[train_idx]])
    return train_idx[~is_watch], train_idx[is_watch]


def train_fold(blocks, masks, y, edge_index, train_idx, val_idx, seed, args,
               groups: np.ndarray) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GATNet([b.shape[1] for b in blocks], args.maskable, hidden=args.hidden,
                   heads=args.heads, dropout=args.dropout,
                   use_graph=not args.no_graph).to(DEVICE)
    bt = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in blocks]
    mt = [torch.tensor(m, device=DEVICE) for m in masks]
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    ei = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
    tr = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    va = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)

    fit_idx, watch_idx = split_inner(train_idx, groups, seed)
    fit = torch.tensor(fit_idx, dtype=torch.long, device=DEVICE)
    pos = float(y[fit_idx].sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(fit_idx) - pos) / max(pos, 1.0)], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(bt, mt, ei)[fit], yt[fit]).backward()
        opt.step()

        if (epoch + 1) % 5 == 0:          # do tap theo doi 5 epoch mot lan cho nhanh
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(bt, mt, ei)[watch_idx]).cpu().numpy()
            score = average_precision_score(y[watch_idx], p)
            if score > best_score:
                best_score, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(bt, mt, ei)[va]).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="control", choices=["control", "frozen"])
    p.add_argument("--graph", type=Path, default=ROOT / "fusion/output/graph_ketqua_k10_bridge.npz")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--out-dir", type=Path, default=ROOT / "fusion/output",
                   help="Noi ghi ket qua. Dung thu muc tam khi chay thu de khong de len ket qua that")
    p.add_argument("--patience", type=int, default=6,
                   help="So lan do lien tiep khong cai thien thi dung (moi lan cach 5 epoch)")
    p.add_argument("--nb-score", action=argparse.BooleanOptionalAction, default=True,
                   help="Dung khoi 'diem nguy co hang xom' (mac dinh bat). --no-nb-score de tat")
    p.add_argument("--no-graph", action="store_true",
                   help="Ablation: tat truyen tin, chi con MLP — de biet do thi dong gop bao nhieu")
    args = p.parse_args()

    d = np.load(ROOT / f"fusion/output/fusion_dataset_ketqua_{args.source}.npz", allow_pickle=False)
    g = np.load(args.graph, allow_pickle=False)
    tab = np.load(ROOT / "fusion/output/tabular_ketqua.npz", allow_pickle=False)
    ts_pred = pd.read_parquet(ROOT / "output/ketqua/ts_predictions.parquet")
    assert (tab["id"] == d["id"]).all() and (ts_pred["id"].to_numpy() == d["id"]).all()

    X_ls = np.hstack([tab["X_tab"], ts_pred["score"].to_numpy(np.float32)[:, None]]).astype(np.float32)
    X_img = np.nan_to_num(d["X_img"].astype(np.float32))
    has_img = d["has_image"].astype(bool)
    y = d["y"].astype(np.float32)
    uid, dev, fold = d["patient_uid"], d["split"] == "dev", d["fold_id"]
    n = len(y)
    edge_index = symmetric_with_self_loops(g["edge_index"], n)
    args.maskable = [False, True, True] if args.nb_score else [False, True]

    tag = ("GAT" + (" (tat do thi)" if args.no_graph else "")
           + ("" if args.nb_score else " (khong diem hang xom)"))
    print("=" * 78)
    print(f"PHASE 4 — {tag}   [nguon anh: {args.source} | {DEVICE}]")
    print("=" * 78)
    print(f"{n} node | {edge_index.shape[1]:,} canh (da doi xung + self-loop) | "
          f"dev {int(dev.sum())} ({int(y[dev].sum())} duong)")

    oof = np.zeros(n)
    for k in range(5):
        va = np.flatnonzero(dev & (fold == k))
        tr_mask = dev & (fold != k) & (fold >= 0)
        tr = np.flatnonzero(tr_mask)
        # Chuan hoa fit tren rieng hang train cua fold — khong dung node validation
        blocks = [standardise(tr_mask, X_ls, np.ones(n, bool)),
                  standardise(tr_mask, X_img, has_img)]
        masks = [np.ones(n, bool), has_img]

        if args.nb_score:
            # Giong B3b: probe anh hoc tren rieng hang TRAIN co anh, roi lay trung binh
            # diem cua cac hang xom co anh. Nhan cua node dang danh gia khong tham gia.
            s_img = image_probe_scores(tr_mask & has_img, X_img, y, has_img)
            nbs, cnt = neighbour_score_mean(g["edge_index"], s_img, has_img)
            X_nbs = np.column_stack([nbs, np.log1p(cnt)]).astype(np.float32)
            blocks.append(standardise(tr_mask, X_nbs, cnt > 0))
            masks.append(cnt > 0)

        preds = [train_fold(blocks, masks, y, edge_index, tr, va, s, args, uid)
                 for s in range(args.seeds)]
        oof[va] = np.mean(preds, axis=0)
        print(f"  [fold {k}] xong")

    res = {"toan_bo_dev": score_block(y[dev], oof[dev], uid[dev]),
           "co_anh": score_block(y[dev & has_img], oof[dev & has_img], uid[dev & has_img]),
           "khong_anh": score_block(y[dev & ~has_img], oof[dev & ~has_img], uid[dev & ~has_img])}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"gat{'_nograph' if args.no_graph else ''}"
            f"{'' if args.nb_score else '_nonb'}_{args.source}")
    L = []
    def out(t=""):
        print(t); L.append(t)

    out(f"# Phase 4 — {tag} (nguon anh `{args.source}`)")
    out()
    out(f"- Sinh luc: {datetime.now(timezone.utc).isoformat()} | thiet bi: {DEVICE}")
    out(f"- {args.seeds} seed x toi da {args.epochs} epoch (early stopping, patience "
        f"{args.patience}) | hidden {args.hidden}, {args.heads} head, dropout {args.dropout}, "
        f"lr {args.lr} | LayerNorm | diem hang xom: {'CO' if args.nb_score else 'KHONG'}")
    out()
    out("| Nhom | n | duong | ROC-AUC | PR-AUC |")
    out("|---|---|---|---|---|")
    for t, title in (("toan_bo_dev", "Toan bo dev"), ("co_anh", "CO anh"), ("khong_anh", "KHONG anh")):
        r = res[t]
        out(f"| {title} | {r['n']} | {r['duong']} | {r['roc']:.3f} "
            f"[{r['roc_ci'][0]:.3f}–{r['roc_ci'][1]:.3f}] | {r['pr']:.3f} "
            f"[{r['pr_ci'][0]:.3f}–{r['pr_ci'][1]:.3f}] |")
    out()

    # ---- Doi chieu thang voi Phase 3 tren cung benh nhan ----
    p3_oof = ROOT / f"fusion/output/phase3_oof_{args.source}.npz"   # luon doc ban chinh
    if p3_oof.exists():
        O = np.load(p3_oof)
        key = {k.split(" ")[0]: k for k in O.files}
        out("## Doi chieu voi Phase 3 (bootstrap ghep cap, cung benh nhan)")
        out()
        out("| So sanh | Nhom | Chenh lech ROC | KTC 95% | P | Ket qua |")
        out("|---|---|---|---|---|---|")
        rows = [("B3b", "khong_anh", dev & ~has_img, "phai_hon"),
                ("B3", "khong_anh", dev & ~has_img, "phai_hon"),
                ("B0", "khong_anh", dev & ~has_img, "phai_hon"),
                ("B2", "co_anh", dev & has_img, "khong_kem"),
                ("L2", "co_anh", dev & has_img, "khong_kem")]
        for short, nhom, mask, rule in rows:
            if short not in key:
                continue
            gap, lo, hi, pr = paired_gap(y[mask], oof[mask], O[key[short]][mask], uid[mask])
            ok = (lo > 0) if rule == "phai_hon" else (hi > 0)
            out(f"| GAT − {key[short]} | {nhom} | {gap:+.4f} | [{lo:+.4f}; {hi:+.4f}] | "
                f"{pr:.0%} | {'DAT' if ok else '**TRUOT**'} |")
        out()
        out("*`phai_hon`: KTC phai nam tron ben duong thi GAT moi thuc su hon. "
            "`khong_kem`: chi truot khi GAT thua ro rang.*")
    else:
        out(f"> Chua co {p3_oof.name} — chay fusion/baselines.py truoc de doi chieu duoc.")

    (args.out_dir / f"phase4_{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (args.out_dir / f"phase4_{name}.json").write_text(
        json.dumps({"source": args.source, "no_graph": args.no_graph, "epochs": args.epochs,
                    "seeds": args.seeds, "ket_qua": res}, indent=2, ensure_ascii=False,
                   default=float), encoding="utf-8")
    np.savez_compressed(args.out_dir / f"phase4_oof_{name}.npz", GAT=oof)
    print(f"\nDa ghi -> {args.out_dir}/phase4_{name}.{{md,json}} + phase4_oof_{name}.npz")


if __name__ == "__main__":
    main()

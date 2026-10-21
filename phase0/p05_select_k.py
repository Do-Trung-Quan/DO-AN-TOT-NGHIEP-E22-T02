"""P0.5(a) — dung lai do thi kNN cho cac k khai bao truoc va chon k CHI TREN DEV (PAG_V2 muc 2, 11.9).

  - Dung lai moi k (config p05.k_ung_vien) trong CUNG khong gian X_tab 139 cot, co canh cau noi.
    (Ba do thi cu nam o hai khong gian khac nhau: k10 tren X_tab, k5 va k20 tren X_ts 32-D.)
  - Tieu chi chon k: homophily lift LOP DUONG, chi tinh tren canh co ca hai dau thuoc cac fold
    dev KHAC fold dang bao cao. Hoa thi chon k nho hon (chenh lift < 0.01).
  - Ban don gian de doi chieu: mot k duy nhat chon bang lift tren toan dev.

Nhan cua fold dang bao cao KHONG tham gia vao viec chon k cua fold do. Khong dung nhan test.
Chay: python -m phase0.p05_select_k            (CPU, vai giay moi do thi)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from fusion.build_graph import diagnose
from phase0.common import OUT, ROOT, Nodes, load_config, write_json

GRAPH_DIR = OUT / "graphs"
TIE = 0.01


def build(k: int) -> Path:
    path = GRAPH_DIR / f"graph_ketqua_k{k}_bridge.npz"
    if not path.exists():
        subprocess.run([sys.executable, str(ROOT / "fusion/build_graph.py"), "--k", str(k), "--bridge",
                        "--output-dir", str(GRAPH_DIR)], check=True, cwd=ROOT)
    return path


def lift(edge_index, y, has, fold, mask) -> float:
    d = diagnose(edge_index, y, has, fold, mask)
    return d["positive_homophily"] / float(y[mask].mean())


def pick(lifts: dict[int, float]) -> int:
    best = max(lifts.values())
    return min(k for k, v in lifts.items() if v >= best - TIE)


def main():
    cfg = load_config()
    ks = cfg["p05"]["k_ung_vien"]
    nd = Nodes("control")
    y, has, fold, dev = nd.y, nd.has, nd.fold, nd.dev
    print("=" * 78)
    print(f"P0.5(a) — chon k chi tren dev | k ung vien {ks} | config {cfg['_hash']}")
    print("=" * 78)
    graphs = {k: np.load(build(k))["edge_index"] for k in ks}
    for k, e in graphs.items():
        print(f"  k={k}: {e.shape[1]:,} canh")

    per_fold, sel = {}, {}
    for f in range(5):
        mask = dev & (fold != f)         # 4 fold con lai cua dev; fold f va test khong co mat
        assert not (mask & (fold == f)).any() and not (mask & ~dev).any()
        lifts = {k: lift(e, y, has, fold, mask) for k, e in graphs.items()}
        per_fold[f] = lifts
        sel[f] = pick(lifts)
    dev_lifts = {k: lift(e, y, has, fold, dev) for k, e in graphs.items()}
    k_dev = pick(dev_lifts)

    print("\nLift homophily lop duong (4 fold con lai) va k duoc chon cho tung fold:")
    for f in range(5):
        print(f"  fold {f}: " + " | ".join(f"k={k} {v:.3f}" for k, v in per_fold[f].items())
              + f"  -> chon k={sel[f]}")
    print("Toan dev: " + " | ".join(f"k={k} {v:.3f}" for k, v in dev_lifts.items()) + f"  -> chon k={k_dev}")
    same = all(v == k_dev for v in sel.values())
    print("\nBan theo-fold va ban toan-dev " + ("CHON CUNG MOT k" if same else "KHAC NHAU"))

    write_json(OUT / "p05_k_selection.json", {
        "config_hash": cfg["_hash"], "k_ung_vien": ks, "tieu_chi": cfg["p05"]["tieu_chi_chon_k"],
        "quy_tac_hoa": f"chon k nho nhat co lift >= lift_toi_da - {TIE}",
        "lift_theo_fold": {str(f): {str(k): v for k, v in d.items()} for f, d in per_fold.items()},
        "k_chon_theo_fold": {str(f): k for f, k in sel.items()},
        "lift_toan_dev": {str(k): v for k, v in dev_lifts.items()}, "k_chon_toan_dev": k_dev,
        "hai_ban_chon_cung_k": bool(same),
        "ghi_chu": "do thi dung trong cung khong gian X_tab 139 cot; so lieu cu o k5/k20 nam o X_ts 32-D khong so sanh duoc"})


if __name__ == "__main__":
    main()

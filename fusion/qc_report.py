"""
================================================================================
CONG NGHIEM THU PHASE 1 + 2  [khong co bao cao nay thi KHONG sang Phase 3]
================================================================================
Gop kiem dinh bo du lieu node (Phase 1) va do thi (Phase 2) thanh mot bao cao
markdown. Thoat ma 1 neu bat ky gate cung nao truot.

6 GATE CUNG:
  1. So node = 8030, so node co anh = 1835
  2. 0 NaN/Inf trong dac trung anh o hang CO anh
  3. 0 chieu hang so (KIEM FLOAT64 — float32 chi lo 1/256)
  4. Node khong anh phai co dac trung anh = NaN (khong duoc dien 0)
  5. 0 benh nhan bac cau dev/test
  6. Label homophily > moc ngau nhien          <- GATE QUAN TRONG NHAT

5 CHI SO BAO CAO (khong dat nguong nhung phai co mat):
  7. Homophily lop duong + lift    <- chi so that su quyet dinh voi du lieu 1:37
  8. Isolated-node coverage
  9. Cross-modality edge ratio
 10. Bac trung binh + thanh phan lien thong
 11. Ty le benh theo nhom co/khong anh (do lon cua shortcut has_image)

CHAY:
    python fusion/qc_report.py --source control --label ketqua \
           --graph fusion/output/graph_ketqua_k10_bridge.npz
================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_NODES = 8030
N_IMAGES = 1835


class Report:
    def __init__(self) -> None:
        self.rows: list[str] = []
        self.failed: list[str] = []

    def gate(self, number: int, name: str, value, threshold: str, ok: bool) -> None:
        self.rows.append(f"| {number} | {name} | `{value}` | {threshold} | "
                         f"{'DAT' if ok else '**TRUOT**'} |")
        print(f"  [{'OK ' if ok else 'FAIL'}] {number}. {name} = {value} (can {threshold})")
        if not ok:
            self.failed.append(f"{number}. {name} = {value}, can {threshold}")

    def note(self, number: int, name: str, value) -> None:
        self.rows.append(f"| {number} | {name} | `{value}` | bao cao | — |")
        print(f"  [    ] {number}. {name} = {value}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["control", "frozen"])
    parser.add_argument("--label", default="ketqua", choices=["ketqua", "bnn"])
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset_path = here / "output" / f"fusion_dataset_{args.label}_{args.source}.npz"
    if args.output is None:
        args.output = here / "output" / f"qc_{args.label}_{args.source}_{args.graph.stem}.md"

    print("=" * 78)
    print(f"CONG NGHIEM THU PHASE 1+2  [nhan {args.label} | nguon {args.source} | do thi {args.graph.stem}]")
    print("=" * 78)

    data = np.load(dataset_path, allow_pickle=False)
    graph = np.load(args.graph, allow_pickle=False)
    gman = json.loads((args.graph.parent / f"{args.graph.stem}_manifest.json").read_text(encoding="utf-8"))

    y = data["y"]
    has_image = data["has_image"].astype(bool)
    X_img = data["X_img"].astype(np.float64)
    uid = data["patient_uid"]
    is_dev = data["split"] == "dev"

    r = Report()

    # ---- Gate 1-5: bo du lieu node ----
    r.gate(1, "So node", f"{len(y)}/{N_NODES}", f"= {N_NODES}", len(y) == N_NODES)
    r.gate(2, "Node co anh", f"{int(has_image.sum())}/{N_IMAGES}", f"= {N_IMAGES}",
           int(has_image.sum()) == N_IMAGES)

    imaged = X_img[has_image]
    r.gate(3, "NaN/Inf o hang CO anh", int((~np.isfinite(imaged)).sum()), "= 0",
           bool(np.isfinite(imaged).all()))

    n_dead = int(((imaged.max(0) - imaged.min(0)) == 0).sum())
    r.gate(4, "Chieu hang so (float64)", f"{n_dead}/{X_img.shape[1]}", "= 0", n_dead == 0)

    # Node khong anh PHAI la NaN — neu bi dien 0 thi 6195 node se dinh thanh mot cum
    plain_all_nan = bool(np.isnan(X_img[~has_image]).all())
    r.gate(5, "Node khong anh = NaN (khong dien 0)", plain_all_nan, "True", plain_all_nan)

    straddle = len(set(uid[is_dev]) & set(uid[~is_dev]))
    r.gate(6, "Benh nhan bac cau dev/test", straddle, "= 0", straddle == 0)

    # ---- Gate 7: do thi ----
    margin = gman["homophily_margin"]
    r.gate(7, "Homophily vuot moc ngau nhien",
           f"{gman['homophily']:.4f} vs {gman['homophily_baseline']:.4f} ({margin:+.4f})",
           "> 0", margin > 0)

    # ---- Chi so bao cao ----
    r.note(8, "Homophily lop duong",
           f"{gman['positive_homophily']:.4f} (lift {gman['positive_homophily_lift']}x)")
    r.note(9, "Node duong co >=1 hang xom duong", f"{gman['positive_reach']:.1%}")
    r.note(10, "Isolated-node coverage", f"{gman['isolated_node_coverage']:.1%}")
    r.note(11, "Cross-modality edge ratio", f"{gman['cross_modality_edge_ratio']:.4f}")
    r.note(12, "Bac trung binh", f"{gman['mean_degree']} "
                                 f"(p10/p50/p90 = {'/'.join(map(str, gman['degree_p10_p50_p90']))})")
    r.note(13, "Thanh phan lien thong",
           f"{gman['n_components']} (lon nhat {gman['largest_component']:,})")

    rate_img = float(y[has_image].mean())
    rate_plain = float(y[~has_image].mean())
    r.note(14, "Ty le benh co anh / khong anh",
           f"{rate_img:.2%} vs {rate_plain:.2%} (gap {rate_img/rate_plain:.2f} lan)")

    verdict = "TRUOT — TU CHOI sang Phase 3" if r.failed else "DAT — duoc phep sang Phase 3"
    print("-" * 78)
    print(f"Ket luan: {verdict}")

    doc = [
        f"# Nghiem thu Phase 1+2 — nguon `{args.source}`, do thi `{args.graph.stem}`",
        "",
        f"- Sinh luc: {datetime.now(timezone.utc).isoformat()}",
        f"- Bo du lieu: `{dataset_path.name}`",
        f"- Do thi   : `{args.graph.name}`",
        f"- **Ket luan: {verdict}**",
        "",
        "| # | Chi tieu | Gia tri | Nguong | Ket qua |",
        "|---|---|---|---|---|",
        *r.rows,
        "",
        "## Doc chi so 8 — vi sao homophily lop duong moi la thuoc do that",
        "",
        f"Du lieu lech 1:37 nen homophily tho luon cao san ({gman['homophily']:.4f}) chi vi",
        "97,4% node la am — no khong cho biet do thi co giup phat hien BENH hay khong.",
        "",
        f"Chi so can nhin: mot benh nhan mac benh thi **{gman['positive_homophily']:.1%}** hang xom",
        f"gan nhat cua ho cung mac benh, so voi **{y.mean():.2%}** ky vong ngau nhien",
        f"— **lift {gman['positive_homophily_lift']}x**. Do thi rat giau thong tin cho lop thieu so.",
        "",
        "## Doc chi so 10 — cau noi modality",
        "",
        f"Isolated-node coverage = **{gman['isolated_node_coverage']:.1%}**: ty le node KHONG co anh",
        "ma cham duoc it nhat mot hang xom CO anh. Day chinh la kenh de 6.195 benh nhan",
        "khong co phim X-quang 'hoc ke' thong tin anh — co che trung tam cua do an.",
        "",
        "## Doc chi so 14 — shortcut `has_image`",
        "",
        f"Nhom co anh co ty le benh cao gap **{rate_img/rate_plain:.2f} lan** nhom khong anh.",
        "Model co the hoc 'co anh => nguy co cao' ma khong can nhin noi dung phim.",
        "Phase 4 chan bang 3 lop: missing-token hoc duoc, modality dropout, va ablation",
        "tat co `has_image`.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(doc), encoding="utf-8")
    print(f"Da ghi -> {args.output}")
    print("=" * 78)
    if r.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
================================================================================
PHASE 2 — DUNG DO THI DAN SO + CHAN DOAN
================================================================================
Quyet dinh AI NOI VOI AI. Day la buoc quyet dinh GNN co hon duoc MLP hay khong.

DO THI DUNG CHUNG CHO CA 3 NGUON ANH:
  Canh duoc dung tu vector TIMESERIES + cac cot thuoc tinh + co has_image — toan
  bo deu giong nhau o ca 3 bo dac trung anh. Nen chi can dung MOT LAN, roi Phase
  3+ ghep do thi nay voi tung bo dac trung.

VI SAO DUNG CANH TREN TS, TUYET DOI KHONG TREN VECTOR GHEP [ts (+) anh]:
  6195 node khong anh co nua vector anh giong het nhau (deu la missing-token).
  Tinh khoang cach tren vector ghep -> chung TU DINH thanh mot cum khong lo tach
  biet voi nhom co anh.
  Do chinh xac la hien tuong ta muon QUAN SAT RIENG o DGCNN (no tu tinh lai kNN
  moi layer). Neu dua san vao do thi tinh thi GAT va Graph Transformer cung
  nhiem, va phep so sanh 3 backbone mat het y nghia.
  Vector ts thi MOI node deu co du -> trung lap voi phuong thuc.

NGUYEN TAC TUYET DOI:
  Canh chi duoc dung tu DAC TRUNG, khong bao gio tu NHAN. Khong noi 2 node vi
  cung `bnn`. Day la dang ro ri nghiem trong nhat trong GNN y te va gan nhu
  khong the phat hien sau khi da xay ra.

CHAY:
    python fusion/build_graph.py --k 10 --bridge      # cau hinh mac dinh
    python fusion/build_graph.py --k 5
    python fusion/build_graph.py --k 20
    python fusion/build_graph.py --k 10 --bridge --attr-edges
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

EDGE_KNN, EDGE_ATTR, EDGE_BRIDGE = 0, 1, 2
EDGE_TYPE_NAME = {EDGE_KNN: "knn", EDGE_ATTR: "attr", EDGE_BRIDGE: "bridge"}


# ==============================================================================
# Chuan hoa + tuong dong
# ==============================================================================
def cosine_matrix(X_ts: np.ndarray, dev_mask: np.ndarray) -> np.ndarray:
    """Chuan hoa ts (fit tren node dev) roi tra ve ma tran cosine 8030x8030.

    Fit scaler tren node dev chu khong tren toan bo: giu dung nguyen tac thong ke
    cua tap test khong duoc ro ri. O day chi dung DAC TRUNG, khong dung nhan, nen
    khong co ro ri nhan.
    """
    mean = X_ts[dev_mask].mean(0)
    std = X_ts[dev_mask].std(0)
    std[std == 0] = 1.0
    Z = (X_ts - mean) / std
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (Z / norms).astype(np.float32)


def undirected_unique(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Chuan hoa canh vo huong: sap xep (min,max), bo trung lap, bo self-loop."""
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    keep = lo != hi
    pairs = np.stack([lo[keep], hi[keep]])
    return np.unique(pairs, axis=1)


# ==============================================================================
# Ba loai canh
# ==============================================================================
def knn_edges(Xn: np.ndarray, k: int) -> np.ndarray:
    """kNN cosine: moi node noi voi k nguoi giong nhat ve ho so lam sang."""
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    src = np.repeat(np.arange(len(Xn)), k)
    return undirected_unique(src, idx.ravel())


def attribute_edges(cviec: np.ndarray, tuoi: np.ndarray, max_per_node: int,
                    seed: int) -> np.ndarray:
    """Noi 2 benh nhan neu CUNG cap do nghe VA chenh lech tuoi <= 5.

    Dua tri thuc nghiep vu vao do thi: hai cong nhan cung nghe cung lua tuoi co
    phoi nhiem tuong tu. Gioi han max_per_node de do thi khong qua day.
    Duyet theo tung nhom nghe + sap xep theo tuoi roi dung searchsorted, tranh
    dung ma tran 8030x8030 boolean.
    """
    rng = np.random.default_rng(seed)
    src_all, dst_all = [], []
    for level in np.unique(cviec):
        members = np.where(cviec == level)[0]
        if len(members) < 2:
            continue
        order = members[np.argsort(tuoi[members], kind="stable")]
        ages = tuoi[order]
        lo = np.searchsorted(ages, ages - 5.0, side="left")
        hi = np.searchsorted(ages, ages + 5.0, side="right")
        for pos, node in enumerate(order):
            cand = order[lo[pos]:hi[pos]]
            cand = cand[cand != node]
            if len(cand) == 0:
                continue
            if len(cand) > max_per_node:
                cand = rng.choice(cand, size=max_per_node, replace=False)
            src_all.append(np.full(len(cand), node))
            dst_all.append(cand)
    if not src_all:
        return np.empty((2, 0), dtype=np.int64)
    return undirected_unique(np.concatenate(src_all), np.concatenate(dst_all))


def bridge_edges(Xn: np.ndarray, has_image: np.ndarray) -> np.ndarray:
    """Moi node KHONG anh noi them 1 canh toi node CO anh gan nhat.

    Can thiep truc tiep vao van de trung tam cua do an: mac dinh mot phan node
    khong-anh KHONG cham duoc bat ky nguon thong tin anh nao. Cau noi dua phu
    song len 100%.
    """
    imaged = np.where(has_image)[0]
    plain = np.where(~has_image)[0]
    if len(imaged) == 0 or len(plain) == 0:
        return np.empty((2, 0), dtype=np.int64)
    nearest = np.empty(len(plain), dtype=np.int64)
    for start in range(0, len(plain), 2048):      # chia lo de khong ton bo nho
        block = plain[start:start + 2048]
        nearest[start:start + len(block)] = imaged[(Xn[block] @ Xn[imaged].T).argmax(1)]
    return undirected_unique(plain, nearest)


# ==============================================================================
# Chan doan — 6 chi so
# ==============================================================================
def diagnose(edge_index: np.ndarray, y: np.ndarray, has_image: np.ndarray) -> dict:
    src, dst = edge_index
    n = len(y)
    p = float(y.mean())

    # 1. Label homophily so voi moc ngau nhien
    homophily = float((y[src] == y[dst]).mean())
    baseline = p ** 2 + (1 - p) ** 2

    # 2. Homophily RIENG LOP DUONG — chi so that su quyet dinh voi du lieu lech 1:37.
    #    Homophily tho luon cao san vi 97.4% node la am; no khong cho biet GNN co
    #    giup phat hien BENH hay khong. Cai can biet la: mot benh nhan mac benh thi
    #    bao nhieu % hang xom cua ho cung mac benh.
    n_pos_nb = np.zeros(n)
    np.add.at(n_pos_nb, dst, y[src])
    np.add.at(n_pos_nb, src, y[dst])
    degree = np.zeros(n)
    np.add.at(degree, dst, 1)
    np.add.at(degree, src, 1)
    pos = y == 1
    pos_homophily = float(n_pos_nb[pos].sum() / max(degree[pos].sum(), 1))
    pos_reach = float((n_pos_nb[pos] > 0).mean())

    # 3. Cross-modality edge ratio
    cross_mod = float((has_image[src] != has_image[dst]).mean())

    # 4. Isolated-node coverage
    n_img_nb = np.zeros(n)
    np.add.at(n_img_nb, dst, has_image[src])
    np.add.at(n_img_nb, src, has_image[dst])
    coverage = float((n_img_nb[~has_image] > 0).mean())

    # 5. Bac
    mean_degree = float(degree.mean())

    # 6. So thanh phan lien thong — do thi vo vun thi thong tin khong lan duoc xa
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        adj = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        n_comp, labels = connected_components(adj, directed=False)
        largest = int(np.bincount(labels).max())
    except Exception:
        n_comp, largest = -1, -1

    return {
        "n_edges": int(edge_index.shape[1]),
        "homophily": round(homophily, 4),
        "homophily_baseline": round(baseline, 4),
        "homophily_margin": round(homophily - baseline, 4),
        "positive_homophily": round(pos_homophily, 4),
        "positive_homophily_lift": round(pos_homophily / max(p, 1e-9), 2),
        "positive_reach": round(pos_reach, 4),
        "cross_modality_edge_ratio": round(cross_mod, 4),
        "isolated_node_coverage": round(coverage, 4),
        "mean_degree": round(mean_degree, 2),
        "degree_p10_p50_p90": [int(np.percentile(degree, 10)),
                               int(np.percentile(degree, 50)),
                               int(np.percentile(degree, 90))],
        "n_components": int(n_comp),
        "largest_component": int(largest),
        "imaged_neighbours_of_plain_nodes_mean": round(float(n_img_nb[~has_image].mean()), 2),
    }


def print_diagnosis(d: dict, y: np.ndarray) -> None:
    print("\n[4] Chan doan do thi")
    print(f"    So canh vo huong            : {d['n_edges']:,}")
    print(f"    Bac trung binh              : {d['mean_degree']}  "
          f"(p10/p50/p90 = {'/'.join(map(str, d['degree_p10_p50_p90']))})")
    print(f"    Thanh phan lien thong       : {d['n_components']}  "
          f"(lon nhat {d['largest_component']:,}/{len(y):,} node)")
    print()
    print(f"    Label homophily             : {d['homophily']:.4f}")
    print(f"      moc ngau nhien            : {d['homophily_baseline']:.4f}")
    print(f"      chenh lech                : {d['homophily_margin']:+.4f}"
          f"   {'<- QUA GATE' if d['homophily_margin'] > 0 else '<- TRUOT GATE'}")
    print()
    print(f"    Homophily LOP DUONG         : {d['positive_homophily']:.4f}"
          f"   (lift {d['positive_homophily_lift']}x so voi ty le nen {y.mean():.4f})")
    print(f"      node duong co >=1 hang xom duong: {d['positive_reach']:.1%}")
    print()
    print(f"    Cross-modality edge ratio   : {d['cross_modality_edge_ratio']:.4f}")
    print(f"    Isolated-node coverage      : {d['isolated_node_coverage']:.1%}"
          f"   (node khong anh co >=1 hang xom co anh)")
    print(f"      TB hang xom co anh        : {d['imaged_neighbours_of_plain_nodes_mean']}")


# ==============================================================================
def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=here / "output" / "fusion_dataset_ketqua_control.npz",
                        help="Chi dung X_ts / has_image / cot thuoc tinh — do thi "
                             "GIONG NHAU o ca 3 nguon anh")
    parser.add_argument("--k", type=int, default=10, help="So hang xom kNN")
    parser.add_argument("--attr-edges", action="store_true",
                        help="Them canh cung cap do nghe va |chenh tuoi| <= 5")
    parser.add_argument("--attr-max", type=int, default=5,
                        help="Toi da bao nhieu canh thuoc tinh moi node")
    parser.add_argument("--bridge", action="store_true",
                        help="Moi node khong-anh noi them toi node co anh gan nhat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    args = parser.parse_args()

    # Do thi phu thuoc dac trung ts, ma dac trung ts khac nhau theo nhan -> gan nhan vao ten file.
    label = args.dataset.stem.split("_")[2] if args.dataset.stem.count("_") >= 3 else "bnn"
    tag = f"{label}_k{args.k}" + ("_attr" if args.attr_edges else "") + ("_bridge" if args.bridge else "")
    print("=" * 78)
    print(f"PHASE 2 — DUNG DO THI  [{tag}]")
    print("=" * 78)

    data = np.load(args.dataset, allow_pickle=False)
    X_ts = data["X_ts"].astype(np.float64)
    y = data["y"]
    has_image = data["has_image"].astype(bool)
    dev_mask = data["split"] == "dev"
    print(f"[1] Nap {args.dataset.name}: {len(y)} node | {int(has_image.sum())} co anh "
          f"| {int(y.sum())} ca benh")

    Xn = cosine_matrix(X_ts, dev_mask)
    print(f"[2] Chuan hoa ts (fit tren {int(dev_mask.sum())} node dev) + L2-normalize")

    parts, types = [], []
    e = knn_edges(Xn, args.k)
    parts.append(e); types.append(np.full(e.shape[1], EDGE_KNN, np.int8))
    print(f"[3] Canh kNN (k={args.k})            : {e.shape[1]:,}")

    if args.attr_edges:
        e = attribute_edges(data["cviec"], data["tuoi"].astype(np.float64),
                            args.attr_max, args.seed)
        parts.append(e); types.append(np.full(e.shape[1], EDGE_ATTR, np.int8))
        print(f"    Canh thuoc tinh              : {e.shape[1]:,}")

    if args.bridge:
        e = bridge_edges(Xn, has_image)
        parts.append(e); types.append(np.full(e.shape[1], EDGE_BRIDGE, np.int8))
        print(f"    Canh cau noi modality        : {e.shape[1]:,}")

    # Gop + khu trung lap, uu tien giu loai kNN khi mot canh xuat hien nhieu lan
    all_edges = np.concatenate(parts, axis=1)
    all_types = np.concatenate(types)
    keys = all_edges[0].astype(np.int64) * len(y) + all_edges[1]
    order = np.lexsort((all_types, keys))            # cung key thi type nho len truoc
    keys_sorted = keys[order]
    first = np.concatenate([[True], keys_sorted[1:] != keys_sorted[:-1]])
    sel = order[first]
    edge_index = all_edges[:, sel]
    edge_type = all_types[sel]

    src, dst = edge_index
    edge_weight = np.einsum("ij,ij->i", Xn[src], Xn[dst]).astype(np.float32)
    n_dup = all_edges.shape[1] - edge_index.shape[1]
    print(f"    Sau khi gop + khu trung lap  : {edge_index.shape[1]:,}  (bo {n_dup:,} trung)")

    d = diagnose(edge_index, y, has_image)
    print_diagnosis(d, y)

    if d["homophily_margin"] <= 0:
        raise SystemExit(
            "\nTRUOT GATE PHASE 2: homophily khong vuot moc ngau nhien.\n"
            "  Benh nhan 'giong nhau ve lam sang' khong co xu huong cung nhan ->\n"
            "  gop thong tin hang xom se PHA LOANG tin hieu, GNN se thua MLP.\n"
            "  Phai doi k / doi khong gian tuong dong / them canh thuoc tinh TRUOC khi train."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"graph_{tag}.npz"
    np.savez_compressed(path, edge_index=edge_index.astype(np.int32),
                        edge_weight=edge_weight, edge_type=edge_type)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "fusion/build_graph.py",
        "tag": tag,
        "k": args.k,
        "attr_edges": args.attr_edges,
        "attr_max_per_node": args.attr_max if args.attr_edges else None,
        "bridge": args.bridge,
        "seed": args.seed,
        "similarity_space": "ts 32-D chuan hoa (fit tren node dev) + cosine",
        "edge_source": "CHI tu dac trung + thuoc tinh — KHONG BAO GIO tu nhan",
        "shared_across_sources": True,
        "edge_type_map": EDGE_TYPE_NAME,
        "edge_type_counts": {EDGE_TYPE_NAME[t]: int((edge_type == t).sum())
                             for t in sorted(set(edge_type.tolist()))},
        **d,
    }
    (args.output_dir / f"graph_{tag}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[5] Da ghi -> {path}  ({path.stat().st_size/1e6:.1f} MB)")
    print(f"    Da ghi -> {args.output_dir / f'graph_{tag}_manifest.json'}")
    print("=" * 78)
    print("PHASE 2 HOAN TAT — QUA GATE")
    print("=" * 78)


if __name__ == "__main__":
    main()

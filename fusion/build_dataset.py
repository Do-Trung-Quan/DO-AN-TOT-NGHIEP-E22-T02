"""
================================================================================
PHASE 1 — DUNG BO DU LIEU NODE cho do thi dan so
================================================================================
Gop 4 nguon roi rac thanh MOT bang 8030 hang, moi hang la mot benh nhan voi day
du dac trung + nhan + thong tin chia du lieu.

    output/<nhan>/timeseries_features.parquet  (8030, 34)  id, has_image, ts_feat_0..31
    output/<nhan>/fusion_node_meta.parquet     (8030, 13)  id, <nhan>, split, fold_id, patient_uid...
    imagefeat/output/image_index.parquet       (1835,  7)  img_id, label_ketqua, label_bnn...
    imagefeat/output/image_features_*.parquet (1835, 259) img_id, img_feat_0..255
                            |
                            v  ghep theo id == img_id (LEFT)
    fusion/output/fusion_dataset_<nguon>.npz
        1835 hang co anh, 6195 hang img_feat = NaN

KHOA NOI LA `id`, KHONG PHAI TEN FILE:
  Ten file chua ho ten benh nhan nen da bi loai khoi moi artifact. `id` la khoa
  cua bang lam sang, on dinh, khong PII, da kiem chung khop img_id 1835/1835.

LUU VECTOR THO, KHONG LUU BAN DA CHUAN HOA:
  Scaler phai fit rieng tren tap train cua TUNG FOLD o Phase 3. Neu chuan hoa
  san o day bang thong ke toan bo 8030 node thi mean/std da "nhin thay" tap
  val/test -> ro ri thong ke.

VI SAO DE NaN CHO NODE THIEU ANH, KHONG DIEN 0:
  So 0 la mot toa do THAT trong khong gian. Dien 0 nghia la dat 6195 nguoi vao
  dung cung mot diem, va ho se tu dinh thanh mot cum khi tinh khoang cach.
  De NaN thi cho nao quen xu ly se CRASH NGAY thay vi cho ket qua sai am tham.
  Model thay NaN bang missing-token HOC DUOC o Phase 4.

CHAY:
    python fusion/build_dataset.py --source control
    python fusion/build_dataset.py --source frozen
    python fusion/build_dataset.py --source frozen --label ketqua
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_NODES = 8030
N_IMAGES = 1835
DIM_TS = 32
DIM_IMG = 256

SOURCE_FILES = {
    "control": "image_features_control.parquet",
    "frozen": "image_features_frozen.parquet",
}

# Nhan dich -> (thu muc chua dau ra nhanh ts, ten cot nhan trong node_meta)
# `ketqua` = doc phim X-quang theo ILO (nhan chinh tu 09/2026).
# `bnn`    = tien su da duoc cong nhan benh nghe nghiep (nhan cu, giu de doi chieu).
LABELS = {"ketqua": ("output/ketqua", "ketqua"), "bnn": ("output", "bnn")}


# ==============================================================================
# Nap va ghep
# ==============================================================================
def load_and_join(repo: Path, source: str, label: str) -> pd.DataFrame:
    ts_dir, label_col = LABELS[label]
    ts = pd.read_parquet(repo / ts_dir / "timeseries_features.parquet")
    meta = (pd.read_parquet(repo / ts_dir / "fusion_node_meta.parquet")
              .rename(columns={label_col: "label"}))
    index = (pd.read_parquet(repo / "imagefeat" / "output" / "image_index.parquet")
               .rename(columns={f"label_{label}": "label_img"}))
    img = pd.read_parquet(repo / "imagefeat" / "output" / SOURCE_FILES[source])

    print("[1] Nap nguon")
    for name, frame in (("timeseries", ts), ("node_meta", meta),
                        ("image_index", index), (f"image[{source}]", img)):
        print(f"    {name:18s} {frame.shape}")

    # Doi ten truoc khi ghep: ca hai nhanh deu co `fold_id` va `has_image` nhung
    # NGHIA KHAC NHAU. fold_id cua nhanh anh = fold cross-fit luc trich dac trung;
    # fold_id cua nhanh ts = fold huan luyen GNN. Ghep thang se de len nhau.
    img = img.rename(columns={"fold_id": "img_fold_id", "has_image": "img_has_image"})
    index = index.rename(columns={"patient_uid": "patient_uid_img",
                                  "patient_group": "patient_group_img"})

    node = (meta
            .merge(index, left_on="id", right_on="img_id", how="left")
            .merge(img, left_on="id", right_on="img_id", how="left",
                   suffixes=("", "_imgfeat")))

    # Dac trung ts ghep theo vi tri hang (da assert cung thu tu o buoc sau)
    ts_cols = [f"ts_feat_{i}" for i in range(DIM_TS)]
    node[ts_cols] = ts[ts_cols].to_numpy()
    node["_ts_id"] = ts["id"].to_numpy()

    print(f"[2] Ghep xong: {node.shape}")
    return node


# ==============================================================================
# 8 assertion cung — tha chet som con hon sai am tham
# ==============================================================================
def run_assertions(node: pd.DataFrame) -> None:
    print("\n[3] Kiem dinh")
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"    [{'OK ' if ok else 'LOI'}] {name}" + (f"  ({detail})" if detail else ""))

    # 1. So hang
    check("So hang = 8030", len(node) == N_NODES, str(len(node)))

    # 2. Thu tu hang: dac trung ts ghep theo VI TRI nen `id` phai trung khit.
    #    Lech thu tu = moi benh nhan nhan vector cua nguoi khac -> sai am tham.
    same_order = bool((node["_ts_id"].to_numpy() == node["id"].to_numpy()).all())
    check("Thu tu hang ts == meta", same_order)

    # 3. Join anh
    n_join = int(node["img_id"].notna().sum())
    check("Join anh = 1835", n_join == N_IMAGES, f"{n_join}/{N_IMAGES}")

    has_img = node["has_image"].to_numpy().astype(bool)
    check("has_image = 1835", int(has_img.sum()) == N_IMAGES, str(int(has_img.sum())))

    # 4. Nhan hai nhanh phai khop tren 1835 hang chung
    m = node["img_id"].notna().to_numpy()
    label_ok = bool((node.loc[m, "label"].to_numpy()
                     == node.loc[m, "label_img"].to_numpy()).all())
    check("nhan 2 nhanh khop (1835 hang)", label_ok)

    # 5. patient_uid hai nhanh phai khop
    uid_ok = bool((node.loc[m, "patient_uid"].to_numpy()
                   == node.loc[m, "patient_uid_img"].to_numpy()).all())
    check("patient_uid khop 2 nhanh", uid_ok)

    # 6-7. Dac trung anh phai sach. KIEM BANG FLOAT64:
    #      o float32 phep kiem chieu hang so chi lo ra 1/256 chieu chet.
    img_cols = [f"img_feat_{i}" for i in range(DIM_IMG)]
    F = node.loc[m, img_cols].to_numpy(np.float64)
    check("0 NaN/Inf o hang co anh", bool(np.isfinite(F).all()))
    n_dead = int(((F.max(0) - F.min(0)) == 0).sum())
    check("0 chieu hang so (float64)", n_dead == 0, f"{n_dead}/{DIM_IMG}")

    # 8. Khong benh nhan nao bac cau dev/test
    uid = node["patient_uid"].to_numpy()
    is_dev = (node["split"].to_numpy() == "dev")
    straddle = set(uid[is_dev]) & set(uid[~is_dev])
    check("0 benh nhan bac cau dev/test", len(straddle) == 0, f"{len(straddle)} nguoi")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        raise SystemExit(f"\nTRUOT {len(failed)} kiem dinh: {failed}")
    print(f"    -> {len(checks)}/{len(checks)} kiem dinh DAT")


# ==============================================================================
# Thong ke de dua vao bao cao
# ==============================================================================
def summarize(node: pd.DataFrame) -> dict:
    has_img = node["has_image"].to_numpy().astype(bool)
    y = node["label"].to_numpy()

    print("\n[4] Thong ke")
    print("    Phan bo nhan theo phuong thuc:")
    rows = {}
    for flag, label in ((True, "Co anh   "), (False, "Khong anh")):
        mask = has_img if flag else ~has_img
        rate = float(y[mask].mean())
        rows["co_anh" if flag else "khong_anh"] = {
            "n": int(mask.sum()), "duong": int(y[mask].sum()), "ty_le": rate}
        print(f"      {label}: {int(mask.sum()):5d} node | {int(y[mask].sum()):3d} ca benh"
              f" ({rate:.2%})")
    lift = rows["co_anh"]["ty_le"] / max(rows["khong_anh"]["ty_le"], 1e-9)
    print(f"      -> nhom co anh co ty le benh cao gap {lift:.2f} lan"
          f"  (SHORTCUT can chan o Phase 4)")

    print("\n    Phan bo theo tap:")
    for split in ("dev", "test"):
        mask = (node["split"].to_numpy() == split)
        print(f"      {split:5s}: {int(mask.sum()):5d} node | {int(y[mask].sum()):3d} ca benh"
              f" | {int(has_img[mask].sum()):4d} co anh")

    batch = node.loc[has_img, "batch_date"].value_counts()
    print(f"\n    Lo chup: {len(batch)} lo | lon nhat {batch.iloc[0]} anh"
          f" | nho nhat {batch.iloc[-1]} anh")

    # Thang do hai phuong thuc — ly do bat buoc phai chuan hoa rieng tung ben
    X_ts = node[[f"ts_feat_{i}" for i in range(DIM_TS)]].to_numpy(np.float64)
    X_img = node.loc[has_img, [f"img_feat_{i}" for i in range(DIM_IMG)]].to_numpy(np.float64)
    print("\n    Thang do (ly do phai chuan hoa rieng tung phuong thuc):")
    print(f"      ts  {DIM_TS:3d} chieu | khoang [{X_ts.min():7.3f}, {X_ts.max():6.3f}]"
          f" | std/chieu {X_ts.std(0).mean():.4f}")
    print(f"      anh {DIM_IMG:3d} chieu | khoang [{X_img.min():7.3f}, {X_img.max():6.3f}]"
          f" | std/chieu {X_img.std(0).mean():.4f}")

    # Anisotropy — canh bao bat buoc cho Phase 2
    Xn = X_img / np.linalg.norm(X_img, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Xn), size=min(600, len(Xn)), replace=False)
    iu = np.triu_indices(len(sel), 1)
    cos_raw = float((Xn[sel] @ Xn[sel].T)[iu].mean())
    Xc = X_img - X_img.mean(0)
    Zn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    cos_ctr = float((Zn[sel] @ Zn[sel].T)[iu].mean())
    print(f"\n    Cosine TB giua cac anh: tho {cos_raw:.4f} -> sau mean-center {cos_ctr:.4f}")
    print("      => Phase 2 BAT BUOC mean-center truoc khi dung do thi kNN.")

    return {
        "phan_bo_nhan": rows,
        "lift_co_anh_vs_khong": round(lift, 3),
        "so_lo_chup": int(len(batch)),
        "cosine_anh_tho": round(cos_raw, 4),
        "cosine_anh_da_center": round(cos_ctr, 4),
    }


# ==============================================================================
# Xuat
# ==============================================================================
def export(node: pd.DataFrame, source: str, label: str, out_dir: Path, stats: dict) -> Path:
    ts_cols = [f"ts_feat_{i}" for i in range(DIM_TS)]
    img_cols = [f"img_feat_{i}" for i in range(DIM_IMG)]
    has_img = node["has_image"].to_numpy().astype(bool)

    # copy(): to_numpy() co the tra ve view chi doc khi pandas khong phai sao chep
    X_img = node[img_cols].to_numpy(np.float32).copy()
    X_img[~has_img] = np.nan          # giu NaN — xem giai thich o docstring

    arrays = {
        "X_ts": node[ts_cols].to_numpy(np.float32),
        "X_img": X_img,
        "has_image": has_img.astype(np.int8),
        "y": node["label"].to_numpy(np.int8),
        "split": node["split"].to_numpy().astype("U8"),
        "fold_id": node["fold_id"].to_numpy(np.int8),
        "patient_uid": node["patient_uid"].to_numpy().astype("U16"),
        "id": node["id"].to_numpy(np.int64),
        "batch_date": node["batch_date"].fillna("nan").to_numpy().astype("U12"),
        "tuoi": node["tuoi"].to_numpy(np.float32),
        "gioitinh": node["gioitinh"].to_numpy(np.int16),
        "cviec": node["cviec"].to_numpy(np.int16),
        "pxuong": node["pxuong"].to_numpy(np.int16),
        "tuoinghe": node["tuoinghe"].to_numpy(np.float32),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fusion_dataset_{label}_{source}.npz"
    np.savez_compressed(path, **arrays)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "fusion/build_dataset.py",
        "source": source,
        "label": label,
        "source_file": SOURCE_FILES[source],
        "n_nodes": N_NODES,
        "n_with_image": int(has_img.sum()),
        "dim_ts": DIM_TS,
        "dim_img": DIM_IMG,
        "join_key": "id (== img_id cua nhanh anh), da kiem chung 1835/1835",
        "group_key": "patient_uid (luat sdt / 6 truong, dung chung 2 nhanh)",
        "missing_image_encoding": "NaN — KHONG dien 0",
        "normalisation": "KHONG chuan hoa o day; scaler fit trong tung fold o Phase 3",
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
        **stats,
    }
    (out_dir / f"fusion_dataset_{label}_{source}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[5] Da ghi -> {path}  ({path.stat().st_size/1e6:.1f} MB)")
    print(f"    Da ghi -> {out_dir / f'fusion_dataset_{label}_{source}_manifest.json'}")
    return path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCE_FILES), required=True,
                        help="Nguon dac trung anh")
    parser.add_argument("--label", choices=sorted(LABELS), default="ketqua",
                        help="Nhan dich (mac dinh ketqua)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()

    print("=" * 78)
    print(f"PHASE 1 — DUNG BO DU LIEU NODE  [nhan: {args.label} | nguon anh: {args.source}]")
    print("=" * 78)

    node = load_and_join(repo, args.source, args.label)
    run_assertions(node)
    stats = summarize(node)
    export(node, args.source, args.label, args.output_dir, stats)

    print("=" * 78)
    print("PHASE 1 HOAN TAT")
    print("=" * 78)


if __name__ == "__main__":
    main()

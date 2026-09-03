"""
================================================================================
Tai lap bo du lieu gop 1835 anh:  file goc 8030 dong  +  info.csv 1835 dong
================================================================================
Ban goc `Main_data_Merged_1835_Images.xlsx` duoc tao tren mot may khac va khong
con lay lai duoc (data/ nam trong .gitignore nen chua bao gio len git). Script
nay dung lai no tu hai nguon van con:

    data/Main_data_fixed_Not_Encode_Mapping_New.xlsx   8030 dong x 181 cot
    data/info.csv                                      1835 dong x 230 cot

DOI CHIEU HAI NGUON (da do bang so, xem docstring cuoi file):
  - 1835 `id` cua info.csv la tap con cua 8030 `id` file goc, va trung KHIT
    voi tap img_id trong imagefeat/output/image_index.parquet.
  - Nhan bnn / ketqua / hoten / namsinh khop 1835/1835.
  - Chi 14/179 cot chung co khac biet that; trong so cot notebook dung lam
    dac trung chi ~387 o thay doi.

VI SAO KHONG CO COT `file_name`:
  info.csv khong chua ten file anh — ban goc lay ten file bang cach quet thu
  muc anh roi rut `id` tu ten file (imagefeat/build_index.py). May nay khong
  co thu muc anh nen khong tai lap duoc ten file.
  Thay vao do dung `has_image` (0/1) lay tu image_index.parquet. Day la thay
  doi CO LOI:
    - Khoa noi sang nhanh anh la `id`, da kiem chung khop 1835/1835.
    - `file_name` chua HO TEN BENH NHAN; bo no khoi pipeline se go PII ra
      khoi output/timeseries_features.parquet (dong bo voi viec nhanh anh
      da chuyen sang img_id + file_hash).

CHAY:
    .venv/Scripts/python.exe scripts/build_merged_dataset.py
================================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 8030
EXPECTED_COLS = 181
EXPECTED_IMAGES = 1835
EXPECTED_POSITIVES = 211


def load_image_ids(index_path: Path) -> set[int]:
    if not index_path.exists():
        raise SystemExit(
            f"Khong tim thay {index_path}.\n"
            "Lay ve tu nhanh anh (khong can doi nhanh):\n"
            '  MSYS_NO_PATHCONV=1 git show '
            '"origin/Chest-X-ray:imagefeat/output/image_index.parquet" \\\n'
            f"    > {index_path}"
        )
    return set(pd.read_parquet(index_path)["img_id"].astype(int))


def build(base_path: Path, info_path: Path, index_path: Path, out_path: Path) -> pd.DataFrame:
    print("=" * 78)
    print("TAI LAP BO DU LIEU GOP 1835 ANH")
    print("=" * 78)

    base = pd.read_excel(base_path)
    info = pd.read_csv(info_path, low_memory=False)
    image_ids = load_image_ids(index_path)
    print(f"\n[1] Nap nguon")
    print(f"    goc      : {base.shape}  {base_path}")
    print(f"    info.csv : {info.shape}  {info_path}")
    print(f"    anh      : {len(image_ids)} img_id  {index_path}")

    # ---- Kiem tra khoa truoc khi ghep ----
    if base["id"].duplicated().any() or info["id"].duplicated().any():
        raise SystemExit("Cot `id` bi trung lap — khong dung lam khoa duoc.")
    base_ids = set(base["id"].astype(int))
    info_ids = set(info["id"].astype(int))
    if not info_ids <= base_ids:
        raise SystemExit(f"{len(info_ids - base_ids)} `id` cua info.csv khong co trong file goc.")
    if info_ids != image_ids:
        raise SystemExit(
            f"Tap id cua info.csv khac tap img_id cua nhanh anh "
            f"(chi info: {len(info_ids - image_ids)}, chi anh: {len(image_ids - info_ids)})."
        )
    print(f"[2] Khoa `id`: info.csv ⊆ file goc  va  trung khit tap anh  ✓")

    # ---- Ghep: info.csv la nguon su that cho 1835 dong cua no ----
    # Ghi de TUNG COT MOT va ep sang object truoc: nhieu cot rong o file goc
    # duoc pandas doc thanh float64, trong khi info.csv co gia tri chuoi
    # ('1/1', '2/3' o cac cot doc phim) -> gan thang se loi LossySetitemError.
    # Kieu du lieu se duoc suy lai dung khi doc lai file .xlsx o buoc sau.
    shared = [c for c in base.columns if c in info.columns and c != "id"]
    merged = base.copy()
    row_mask = merged["id"].astype(int).isin(info_ids).values
    target_ids = merged.loc[row_mask, "id"].astype(int)
    info_by_id = info.set_index("id")
    n_cast = 0
    for col in shared:
        new_values = info_by_id[col].reindex(target_ids).to_numpy()
        if merged[col].dtype != object:
            merged[col] = merged[col].astype(object)
            n_cast += 1
        merged.loc[row_mask, col] = new_values
    print(f"[3] Cap nhat {len(shared)} cot chung cho {int(row_mask.sum())} dong "
          f"({n_cast} cot phai ep sang object de tranh xung dot kieu)")

    # ---- Thay file_name (khong tai lap duoc, lai chua PII) bang has_image ----
    merged["has_image"] = merged["id"].astype(int).isin(image_ids).astype(np.int8)
    if "file_name" in merged.columns:
        pos = list(merged.columns).index("file_name")
        merged = merged.drop(columns=["file_name"])
        cols = [c for c in merged.columns if c != "has_image"]
        merged = merged[cols[:pos] + ["has_image"] + cols[pos:]]
    print(f"[4] Thay `file_name` bang `has_image` (khoa noi nhanh anh la `id`)")

    # ---- Nghiem thu ----
    print(f"\n[5] Kiem dinh")
    checks = []

    def check(name, ok, got, want):
        checks.append(ok)
        print(f"    [{'DAT' if ok else 'LOI'}] {name:34s} {got:>12}   (ky vong {want})")

    check("So dong", len(merged) == EXPECTED_ROWS, str(len(merged)), EXPECTED_ROWS)
    check("So cot", merged.shape[1] == EXPECTED_COLS, str(merged.shape[1]), EXPECTED_COLS)
    check("Thu tu dong giu nguyen",
          bool((merged["id"].values == base["id"].values).all()), "khop", "khop")
    n_img = int(merged["has_image"].sum())
    check("So benh nhan co anh", n_img == EXPECTED_IMAGES, str(n_img), EXPECTED_IMAGES)
    n_pos = int((merged["bnn"].astype(str).str.strip().str.lower() == "co").sum())
    check("So ca benh (bnn='co')", n_pos == EXPECTED_POSITIVES, str(n_pos), EXPECTED_POSITIVES)
    n_pos_img = int(((merged["bnn"].astype(str).str.strip().str.lower() == "co")
                     & (merged["has_image"] == 1)).sum())
    check("Ca benh trong nhom co anh", n_pos_img == 99, str(n_pos_img), 99)
    n_ketqua1 = int(((pd.to_numeric(merged["ketqua"], errors="coerce") == 1)
                     & (merged["has_image"] == 1)).sum())
    check("ketqua==1 trong nhom co anh", n_ketqua1 == 462, str(n_ketqua1), 462)

    if not all(checks):
        raise SystemExit("\nCo muc kiem dinh KHONG DAT — khong ghi file.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_path, index=False)
    print(f"\n[6] Da ghi {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("=" * 78)
    print("HOAN TAT — tat ca muc kiem dinh DAT")
    print("=" * 78)
    return merged


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path,
                   default=repo / "data" / "Main_data_fixed_Not_Encode_Mapping_New.xlsx")
    p.add_argument("--info", type=Path, default=repo / "data" / "info.csv")
    p.add_argument("--index", type=Path,
                   default=repo / "imagefeat" / "output" / "image_index.parquet")
    p.add_argument("--out", type=Path,
                   default=repo / "data" / "Main_data_Merged_1835_Images.xlsx")
    a = p.parse_args()
    for path in (a.base, a.info):
        if not path.exists():
            sys.exit(f"Khong tim thay {path}")
    build(a.base, a.info, a.index, a.out)


if __name__ == "__main__":
    main()


"""
================================================================================
P0 — XAY DUNG CHI MUC CHUAN cho nhanh anh  [nguon su that duy nhat]
================================================================================
Ghep 1835 anh X-quang voi 1835 dong trong timeseriesDATA/info.csv, sinh ra
mot bang chi muc duy nhat ma moi buoc sau (crossfit, extract, QC) deu doc.

TAI SAO KHONG MATCH THEO TEN:
  Ten file co loi chinh ta so voi ho ten trong info.csv:
      016_LE_THI_UYEN_20181219.jpg   <->  hoten = "Le Thi Yen"
      155_TRUOGN_VAN_MIEN_...        <->  "Truong Van Mien"
      2007_HAONG_VAN_ANH_...         <->  "Hoang Van Anh"
  Match theo ten chi dat 67.4%. Match theo cot `id` dat 1835/1835.

BANG CHUNG KHOA JOIN DUNG:
  Doi chieu ketqua==1 (info.csv) <-> label==1 (thu muc silicosis) tren 1828
  anh rut duoc id bang 4 pattern dau: 1828 khop / 0 sai lech. Sai lech bang 0
  tren 1828 mau la bang chung khoa join chinh xac.

5 PATTERN TEN FILE -> vi tri cua `id`:
  1. NNNN_TEN_YYYYMMDD          008_MAC_DUY_THANG_20181219.jpg      -> tien to
  2. YYYYMMDD_HHMMSS_ID_PID_SID 20181222_075752_1024_12798_14253.jpg -> truong 3
  3. YYYYMMDD_HHMMSS_ID_TEN     20181223_075707_1235_DO THANH TRI.jpg-> truong 3
  4. YYYYMMDD_HHMMSS_ID_TEN NS  20191113_081107_8001_THAI VAN THI 1971.jpg -> truong 3
  5. TEN_ID[_YYYYMMDD]          NGUYEN QUANG DAO_3931.jpg           -> sau ten

NHAN:
  label_ketqua   = (ketqua == 1)  -> 462 duong (25.2%)  = NHAN HUAN LUYEN
  label_bnn      = (bnn == 'Co')  ->  99 duong ( 5.4%)  = nhan dich, chi bao cao

PII: dau ra KHONG chua ten file goc va KHONG chua ho ten. Bang anh xa nguoc
     ghi rieng ra LOCAL_ONLY_filename_map.csv (da chan trong .gitignore).

CHAY:
    python imagefeat/build_index.py
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_ROWS = 1835
DATA_ROOT = Path(r"C:\Users\Admin\Downloads\archive")

# ---- 5 pattern rut `id` tu ten file (thu tu quan trong) ----
RE_P1 = re.compile(r"^(\d+)_(?=\D)")                    # NNNN_TEN_...
RE_P234 = re.compile(r"^\d{8}_\d{6}_(\d+)_")            # YYYYMMDD_HHMMSS_ID_...
RE_P5 = re.compile(r"^[A-Za-z][A-Za-z ]*_(\d+)(?:_\d{8})?$")   # TEN_ID[_DATE]

RE_DATE_PREFIX = re.compile(r"^(\d{8})_\d{6}_")
RE_DATE_SUFFIX = re.compile(r"_(\d{8})$")


def extract_batch_date(fname: str) -> str:
    """Ngay chup — dung de phan tang fold va bao cao metric tach theo lo."""
    stem = Path(fname).stem
    match = RE_DATE_PREFIX.match(stem) or RE_DATE_SUFFIX.search(stem)
    return match.group(1) if match else "unknown"


def extract_id(fname: str) -> int | None:
    """Rut `id` (khop voi info.csv.id) tu ten file. None neu khong nhan dang duoc."""
    stem = Path(fname).stem
    for pattern in (RE_P1, RE_P234, RE_P5):
        match = pattern.match(stem)
        if match:
            return int(match.group(1))
    return None


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(value))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_name(value: object) -> str:
    """'Tô Văn Diệu' -> 'TO_VAN_DIEU'. Dung de gop benh nhan, khong xuat ra file."""
    ascii_name = strip_accents(value).upper()
    return re.sub(r"[^A-Z]+", "_", ascii_name).strip("_")


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def scan_images(images_root: Path) -> pd.DataFrame:
    """Quet TOAN BO anh duoi images_root, bo qua phan chia train/test cu.

    Cross-fitting can gop ca 1835 anh vao mot be roi chia lai theo benh nhan,
    nen phan chia thu muc train/ test/ cua nhanh anh bi CO Y bo qua o day.
    """
    rows = []
    for path in sorted(images_root.rglob("*")):
        # Bo qua thu muc cache anh da resize (_cache512/...) va thu muc an.
        # Anh trong cache TRUNG TEN voi anh goc -> khong bo qua se bao loi trung
        # ten file va dung han, hoac te hon la chon nham ban da nen lai.
        if any(part.startswith("_cache") or part.startswith(".") for part in path.parts):
            continue
        if path.suffix.lower() in IMG_EXT:
            rows.append({"image_path": str(path), "fname": path.name})
    if not rows:
        raise SystemExit(f"Khong tim thay anh nao duoi {images_root}")
    frame = pd.DataFrame(rows)
    duplicated = frame["fname"].duplicated()
    if duplicated.any():
        raise SystemExit(
            f"Co {int(duplicated.sum())} ten file trung nhau duoi {images_root}; "
            "khong the dung ten file lam khoa. Kiem tra lai thu muc."
        )
    return frame


def resolve_image_paths(index: pd.DataFrame, images_root: Path | None,
                        map_csv: Path | None) -> pd.DataFrame:
    """Gan `image_path` cho tung `img_id` cua chi muc.

    Hai duong, uu tien --images-root:
      1. images_root : quet thu muc, rut `img_id` tu ten file. Chay duoc o BAT KY
         may nao (Colab, may khac) va KHONG can file PII nao.
      2. map_csv     : doc LOCAL_ONLY_filename_map.csv. File nay chua duong dan
         TUYET DOI cua may sinh ra no va chua ten benh nhan trong ten file, nen
         chi dung duoc tren dung may do. Giu lai de tuong thich nguoc.
    """
    if images_root is not None:
        scanned = scan_images(Path(images_root))
        scanned["img_id"] = scanned["fname"].map(extract_id)
        unmatched = scanned["img_id"].isna()
        if unmatched.any():
            print(f"  [CANH BAO] {int(unmatched.sum())} file khong rut duoc `id`, bo qua:")
            for name in scanned.loc[unmatched, "fname"].head(5):
                print(f"      {name}")
            scanned = scanned[~unmatched]
        scanned["img_id"] = scanned["img_id"].astype(int)
        if scanned["img_id"].duplicated().any():
            raise SystemExit("Nhieu anh cung tro ve mot `id` — kiem tra thu muc anh.")
        frame = index.merge(scanned[["img_id", "image_path"]], on="img_id", how="left")
        source = f"quet thu muc {images_root}"
    elif map_csv is not None and Path(map_csv).exists():
        mapping = pd.read_csv(map_csv)
        frame = index.merge(mapping[["img_id", "image_path"]], on="img_id", how="left")
        source = f"bang anh xa {map_csv}"
    else:
        raise SystemExit(
            "Can --images-root (khuyen nghi) hoac --map de biet anh nam o dau."
        )

    missing = frame["image_path"].isna()
    if missing.any():
        raise SystemExit(
            f"{int(missing.sum())}/{len(frame)} anh khong tim thay ({source}).\n"
            f"  Vi du img_id thieu: {frame.loc[missing, 'img_id'].head(5).tolist()}"
        )
    print(f"  Duong dan anh: {len(frame)}/{len(index)} — nguon: {source}")
    return frame.sort_values("img_id").reset_index(drop=True)


def load_clinical(info_csv: Path) -> pd.DataFrame:
    clinical = pd.read_csv(info_csv, low_memory=False)
    if len(clinical) != EXPECTED_ROWS:
        print(f"  [CANH BAO] info.csv co {len(clinical)} dong, ky vong {EXPECTED_ROWS}")
    for column in ("id", "hoten", "namsinh", "bnn", "ketqua"):
        if column not in clinical.columns:
            raise SystemExit(f"info.csv thieu cot bat buoc: '{column}'")
    clinical = clinical.loc[:, ["id", "hoten", "namsinh", "bnn", "ketqua"]].copy()
    clinical["id"] = pd.to_numeric(clinical["id"], errors="coerce").astype("Int64")
    if clinical["id"].isna().any():
        raise SystemExit("info.csv co dong khong doc duoc `id`.")
    if clinical["id"].duplicated().any():
        raise SystemExit("info.csv co `id` trung lap -> khong dung lam khoa duoc.")
    return clinical


def build(images_root: Path, info_csv: Path, output_dir: Path,
          identity_csv: Path | None = None) -> pd.DataFrame:
    print("=" * 80)
    print("P0 — XAY DUNG CHI MUC ANH  (join theo info.csv.id)")
    print("=" * 80)

    images = scan_images(images_root)
    print(f"[1] Quet anh: {len(images)} file duoi {images_root}")

    clinical = load_clinical(info_csv)
    print(f"[2] info.csv : {len(clinical)} dong, {clinical['id'].nunique()} id duy nhat")

    images["img_id"] = images["fname"].map(extract_id)
    unmatched = images["img_id"].isna()
    if unmatched.any():
        print(f"\n[LOI] {int(unmatched.sum())} file khong rut duoc `id`:")
        for name in images.loc[unmatched, "fname"].head(20):
            print(f"    {name}")
        raise SystemExit(
            "Bo sung pattern vao extract_id() roi chay lai. "
            "Khong duoc bo qua anh — chi muc phai phu kin 1835/1835."
        )
    images["img_id"] = images["img_id"].astype(int)

    if images["img_id"].duplicated().any():
        dupes = images.loc[images["img_id"].duplicated(keep=False)].sort_values("img_id")
        print("\n[LOI] Nhieu anh cung tro ve mot `id`:")
        print(dupes[["fname", "img_id"]].head(20).to_string(index=False))
        raise SystemExit("`id` phai la 1-1 voi anh.")

    merged = images.merge(clinical, left_on="img_id", right_on="id", how="left")
    missing = merged["id"].isna()
    if missing.any():
        print(f"\n[LOI] {int(missing.sum())} anh co `id` khong ton tai trong info.csv:")
        print(merged.loc[missing, ["fname", "img_id"]].head(20).to_string(index=False))
        raise SystemExit("Join that bai.")
    print(f"[3] Join   : {len(merged)}/{len(images)} anh khop info.csv  (100%)")

    # ---- nhan ----
    merged["label_ketqua"] = (
        pd.to_numeric(merged["ketqua"], errors="coerce") == 1
    ).astype(np.int8)
    merged["label_bnn"] = (
        merged["bnn"].astype(str).str.strip().str.lower() == "co"
    ).astype(np.int8)

    # ---- nhom benh nhan: lay tu info.csv (sach, khong typo nhu ten file) ----
    merged["_patient_raw"] = (
        merged["hoten"].map(normalize_name) + "|" + merged["namsinh"].astype(str).str.strip()
    )
    merged["patient_group"] = merged["_patient_raw"].map(stable_hash)

    # ---- patient_uid: danh tinh CHUAN, lay tu nhanh Timeseries ----
    # KHONG tu tinh lai o day. Luat can cot `sdt` va cap do nghe cua notebook ts;
    # cai lai o hai noi la nguy co hai nhanh gom khac nhau — dung cai loi dang sua.
    #   patient_group (ten+namsinh) : chi de DOI CHIEU CHEO voi nhanh ts
    #   patient_uid   (sdt/6 truong): DUNG DE CHIA FOLD
    if identity_csv is not None and Path(identity_csv).exists():
        identity = pd.read_parquet(identity_csv)
        merged = merged.merge(identity[["id", "patient_uid"]],
                              left_on="img_id", right_on="id", how="left",
                              suffixes=("", "_ident"))
        n_missing = int(merged["patient_uid"].isna().sum())
        if n_missing:
            raise SystemExit(
                f"{n_missing} anh khong co patient_uid trong {identity_csv}. "
                "Lay ban moi nhat tu nhanh Timeseries:\n"
                '  git show "Timeseries:output/patient_identity.parquet" '
                "> output/patient_identity.parquet"
            )
        print(f"[3b] patient_uid: {merged['patient_uid'].nunique()} nguoi "
              f"(patient_group cu: {merged['patient_group'].nunique()} nhom)")
    else:
        raise SystemExit(
            f"Khong thay {identity_csv}.\n"
            "Bat buoc phai co de chia fold dung — lay tu nhanh Timeseries:\n"
            '  git show "Timeseries:output/patient_identity.parquet" '
            "> output/patient_identity.parquet"
        )

    # ---- ngay chup: de phan tang va bao cao tach lo ----
    merged["batch_date"] = merged["fname"].map(extract_batch_date)

    # ---- go PII: thay ten file bang hash on dinh ----
    merged["file_hash"] = merged["fname"].map(stable_hash)

    index = merged.loc[:, [
        "img_id", "file_hash", "patient_uid", "patient_group",
        "label_ketqua", "label_bnn", "batch_date",
    ]].sort_values("img_id").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "image_index.parquet"
    index.to_parquet(index_path, index=False)

    # bang anh xa nguoc -> CHI LUU LOCAL, da chan trong .gitignore
    mapping = merged.loc[:, ["img_id", "file_hash", "fname", "image_path"]].sort_values("img_id")
    map_path = output_dir / "LOCAL_ONLY_filename_map.csv"
    mapping.to_csv(map_path, index=False, encoding="utf-8")

    n_patients = index["patient_uid"].nunique()
    multi_visit = (index.groupby("patient_uid").size() > 1).sum()
    print(f"[4] Benh nhan: {n_patients} nguoi, {multi_visit} nguoi co nhieu lan kham")
    print(f"[5] Nhan   : ketqua duong = {int(index['label_ketqua'].sum())} "
          f"({100 * index['label_ketqua'].mean():.1f}%)  <- NHAN HUAN LUYEN")
    print(f"           : bnn    duong = {int(index['label_bnn'].sum())} "
          f"({100 * index['label_bnn'].mean():.1f}%)  <- nhan dich, chi bao cao")
    print(f"\nDa ghi -> {index_path}")
    print(f"Da ghi -> {map_path}   [PII — KHONG commit]")
    return index


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, default=DATA_ROOT,
                        help="Thu muc goc chua anh (quet de quy, gom ca train/ va test/)")
    parser.add_argument("--info-csv", type=Path, default=DATA_ROOT / "info.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--identity", type=Path,
                        default=repo / "output" / "patient_identity.parquet",
                        help="Bang id -> patient_uid do nhanh Timeseries sinh ra")
    args = parser.parse_args()
    build(args.images_root, args.info_csv, args.output_dir, args.identity)


if __name__ == "__main__":
    main()

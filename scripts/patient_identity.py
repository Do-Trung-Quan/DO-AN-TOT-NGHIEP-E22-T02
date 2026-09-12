"""
================================================================================
DINH DANH BENH NHAN — nguon su that duy nhat cho ca hai nhanh (ts + anh)
================================================================================
8030 dong du lieu la 8030 LUOT KHAM, khong phai 8030 NGUOI. Muon chia du lieu
ma khong ro ri thi phai biet luot kham nao la cua cung mot nguoi.

LUAT XAC DINH "CUNG MOT NGUOI":
  1. Trung SO DIEN THOAI                      -> cung mot nguoi.
  2. Neu it nhat mot ben THIEU so dien thoai  -> phai trung CA 6 truong:
     ho ten, nam sinh, tinh, gioi tinh, cap do nghe `cviec`, cap do nghe `pxuong`.
  3. Con lai                                  -> hai nguoi KHAC NHAU.

Cap do nghe dung dung ham classify_job_10_levels() ben duoi, nen
"Van hanh may" va "van hanh may moc" quy ve cung mot cap.

HAI COT DAU RA, HAI VIEC KHAC NHAU:
  * patient_uid   = nhan nhom theo 3 luat tren  -> DUNG DE CHIA DU LIEU.
  * patient_group = hash(ho ten + nam sinh)     -> chi de DOI CHIEU CHEO voi
                    imagefeat/output/image_index.parquet (nhanh anh dang luu
                    cot nay). Giu lai thi moi con phep kiem "hai nhanh doc ten
                    giong nhau khong".
  Ca hai deu la NHAN NHOM: nhieu `id` khac nhau co the mang cung mot gia tri.
  Chi rieng `id` la duy nhat tren tung dong.

CHAY DOC LAP (xuat output/patient_identity.parquet cho nhanh anh dung chung):
    .venv/Scripts/python.exe scripts/patient_identity.py
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

# ==============================================================================
# PHAN 1 — Phan loai nghe 10 cap
# Sao y NGUYEN VAN tu notebook. Notebook import nguoc lai tu day de hai noi
# khong bao gio lech nhau.
# ==============================================================================
import re
import unicodedata
import pandas as pd

try:
    from unidecode import unidecode as _unidecode
except ImportError:
    _unidecode = None

JOB_COLS = ["cviec", "pxuong", "cviec1", "cviec2"]
LV9_ID = 8  # lv9: Không đi làm / không xác định

level_to_id = {f"lv{i}": i - 1 for i in range(1, 10)}
level_to_id["lv10"] = 9

def _remove_vietnamese_accents(text: str) -> str:
    if _unidecode is not None:
        return _unidecode(text)
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

def clean_job_text(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip().lower()
    if s in {"", "nan", "none", "<na>"}:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = s.replace("đ", "d").replace("Đ", "d")
    s = _remove_vietnamese_accents(s)
    s = re.sub(r"\bsau chua\b", "sua chua", s)
    s = s.replace("x-ray", "x ray").replace("x ray", "x ray")
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _has_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None

def _match_keywords(text: str, keywords, short_boundary=None) -> bool:
    short_boundary = short_boundary or set()
    for kw in sorted(keywords, key=len, reverse=True):
        if kw in short_boundary:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        elif _has_phrase(text, kw):
            return True
    return False

def classify_job_10_levels(clean_text: str) -> str:
    t = clean_text or ""
    if t == "" or t in {"khong", "nan", "o nha", "tu do"}:
        return "lv9"
    lv1_keywords = ["luyen", "duc", "nau", "nan", "no min", "ren", "dot", "chung cat", "khai thac", "cat", "nung", "coc", "thep", "gang", "silic", "da", "than", "lo", "mo", "kim loai", "xi mang", "clinker", "nghien", "amiang"]
    if _match_keywords(t, lv1_keywords): return "lv1"
    lv2_keywords = ["gia cong", "xay", "boc", "boc vac", "khoan", "mai", "pha", "trang men", "trang tri", "tron", "cong truong", "phu ho", "xay dung", "dap", "gach", "gach men", "ceramic", "tho"]
    if _match_keywords(t, lv2_keywords): return "lv2"
    lv3_keywords = ["phoi lieu", "tao hinh", "sua", "sua chua", "bao tri", "bao duong", "dien", "han", "ren", "co khi", "phay"]
    if _match_keywords(t, lv3_keywords): return "lv3"
    lv4_keywords = ["can", "dong", "det", "phan loai", "xuat vo", "soi"]
    if _match_keywords(t, lv4_keywords): return "lv4"
    if "bao ve" not in t and _has_phrase(t, "bao"): return "lv4"
    lv5_short = {"cn", "lam", "px", "vs", "con", "sx"}
    lv5_keywords = ["cong nhan", "lao dong", "san xuat", "xuat", "nguyen lieu", "nhap lieu", "nap lieu", "san pham", "say", "tron", "ve sinh", "xa", "truc", "bao ve", "cong ty", "cty", "kcn", "cong nghiep", "lien doanh", "nha may", "xi nghiep", "lap rap", "day chuyen", *lv5_short]
    if _match_keywords(t, lv5_keywords, short_boundary=lv5_short): return "lv5"
    lv6_short = {"vh", "dieu", "in", "ky"}
    lv6_keywords = ["chay", "dieu khien", "dung", "ky thuat", "lai", "may", "xe", "truc", "xu ly", "van chuyen", "van tai", *lv6_short]
    if _match_keywords(t, lv6_keywords, short_boundary=lv6_short): return "lv6"
    lv7_short = {"xn"}
    lv7_keywords = ["thi nghiem", "xet nghiem", "dem", "kiem", "ktra", "lay mau", "lab", "qc", "kcs", "x ray", *lv7_short]
    if _match_keywords(t, lv7_keywords, short_boundary=lv7_short): return "lv7"
    lv8_keywords = ["giam sat", "ke hoach", "kinh doanh", "kho vat tu", "quan ly", "ql", "qly", "giam doc", "nhan vien", "nv", "pho", "phong", "quan", "quat", "quy cach", "so lieu", "truong", "chi huy", "giay to", "thong", "cung cap", "ban", "buon ban", "chup anh", "chuyen vien", "di hoc", "bo doi", "quan su", "du lich", "ks", "nuoi", "nong nghiep", "gia suc", "gia cam", "noi", "quang cao", "sales", "shop", "giay", "quan ao", "thuc pham", "tiep thi", "bep", "nau an", "tap vu", "y te", "thiet ke", "ke toan"]
    if _match_keywords(t, lv8_keywords, short_boundary={"ql", "nv", "ks"}): return "lv8"
    return "lv10"

# ==============================================================================
# PHAN 2 — Chuan hoa cac truong dinh danh
# ==============================================================================
def strip_accents_nfd(value) -> str:
    """Bo dau bang NFD.

    KHONG dung _remove_vietnamese_accents() o Phan 1: ham do uu tien thu vien
    `unidecode` nen co the cho ket qua khac. `patient_group` phai trung khit
    cong thuc cua imagefeat/build_index.py, neu lech thi mat phep doi chieu
    cheo giua hai nhanh.
    """
    decomposed = unicodedata.normalize("NFD", str(value))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_name(value) -> str:
    """'To Van Dieu' -> 'TO_VAN_DIEU'"""
    return re.sub(r"[^A-Z]+", "_", strip_accents_nfd(value).upper()).strip("_")


def normalize_field(value) -> str:
    """Chuan hoa tinh / gioi tinh: bo dau, viet hoa, gop ky tu la thanh '_'."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", "_", strip_accents_nfd(value).upper()).strip("_")


def normalize_namsinh(value) -> str:
    """'1970', 1970 va 1970.0 phai cho ra CUNG mot chuoi.

    Excel doc ra int, CSV doc ra float -> khong chuan hoa la hash lech hoan toan.
    """
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def normalize_sdt(value) -> str | None:
    """Tra ve so da chuan hoa, hoac None neu thieu / khong hop le.

    Bo '.0' do Excel doc so thanh float, bo ky tu khong phai chu so, bo so 0
    dau. Duoi 8 chu so coi nhu rac.
    """
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value).split(".")[0]).lstrip("0")
    return digits if len(digits) >= 8 else None


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


class UnionFind:
    """Gom cac luot kham lai theo quan he 'cung mot nguoi'."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ==============================================================================
# PHAN 3 — Dung danh tinh
# ==============================================================================
def job_level_ids(frame, cols=("cviec", "pxuong")):
    """Quy cac cot nghe dang van ban ve ID cap do 0..9 (giong het notebook)."""
    present = [c for c in cols if c in frame.columns]
    values = pd.unique(pd.concat([frame[c] for c in present], ignore_index=True))
    mapping = {raw: level_to_id[classify_job_10_levels(clean_job_text(raw))] for raw in values}
    return pd.DataFrame(
        {c: frame[c].map(mapping).fillna(LV9_ID).astype(int) for c in present},
        index=frame.index,
    )


def build_identity(df_raw, verbose: bool = True):
    """Tra ve DataFrame: row_id, id, patient_group, patient_uid, n_visits, has_phone."""
    n = len(df_raw)
    levels = job_level_ids(df_raw)
    key = pd.DataFrame({
        "sdt": df_raw["sdt"].map(normalize_sdt),
        "hoten": df_raw["hoten"].map(normalize_name),
        "namsinh": df_raw["namsinh"].map(normalize_namsinh),
        "tinh": df_raw["tinh"].map(normalize_field) if "tinh" in df_raw.columns else "",
        "gioitinh": df_raw["gioitinh"].map(normalize_field) if "gioitinh" in df_raw.columns else "",
        "lv_cviec": levels["cviec"] if "cviec" in levels else LV9_ID,
        "lv_pxuong": levels["pxuong"] if "pxuong" in levels else LV9_ID,
    }).reset_index(drop=True)
    has_phone = key["sdt"].notna()

    uf = UnionFind(n)

    # --- LUAT 1: trung so dien thoai ---
    # Gom tren `key` DAY DU (RangeIndex 0..n-1) de vi tri groupby().indices tra
    # ve trung voi chi so dong goc. Gom tren frame da loc se cho vi tri sai.
    n_rule1 = 0
    for _, idxs in key.groupby("sdt", dropna=True).indices.items():
        idxs = np.asarray(idxs)
        for j in idxs[1:]:
            uf.union(int(idxs[0]), int(j))
            n_rule1 += 1

    # --- LUAT 2: it nhat mot ben thieu sdt + trung ca 6 truong ---
    six = ["hoten", "namsinh", "tinh", "gioitinh", "lv_cviec", "lv_pxuong"]
    n_rule2 = 0
    for _, idxs in key.groupby(six, dropna=False).indices.items():
        idxs = np.asarray(idxs)
        if len(idxs) < 2:
            continue
        missing = idxs[~has_phone.values[idxs]]
        having = idxs[has_phone.values[idxs]]
        if len(missing) == 0:
            continue                       # ca hai deu co sdt -> luat 3 lo
        for j in missing[1:]:
            uf.union(int(missing[0]), int(j))
            n_rule2 += 1
        for j in having:
            uf.union(int(missing[0]), int(j))
            n_rule2 += 1

    # --- Xung dot luat 1 vs luat 2: p1(sdt A) ~ m(thieu) ~ p2(sdt B) ---
    # Bac cau se gom p1 voi p2 du khac so, trai luat 3. Cat lai theo so dien
    # thoai vi do la bang chung manh nhat.
    comp = np.array([uf.find(i) for i in range(n)])
    final = comp.astype(object)
    n_conflict = 0
    for root, idxs in pd.Series(comp).groupby(comp).indices.items():
        phones = set(key["sdt"].values[idxs][has_phone.values[idxs]])
        if len(phones) <= 1:
            continue
        n_conflict += 1
        for i in idxs:
            p = key["sdt"].values[i]
            final[i] = f"{root}#{p}" if p is not None else f"{root}#nophone"

    codes = pd.factorize(pd.Series(final).astype(str))[0]
    sizes = pd.Series(codes).value_counts()

    patient_uid = np.array([stable_hash(f"pid::{c}") for c in codes])
    patient_group = (key["hoten"] + "|" + key["namsinh"]).map(stable_hash).to_numpy()

    if verbose:
        print(f"   Luot kham                    : {n}")
        print(f"   Co so dien thoai hop le      : {int(has_phone.sum())} ({has_phone.mean():.1%})")
        print(f"   Luat 1 (trung sdt)           : gop {n_rule1} cap")
        print(f"   Luat 2 (thieu sdt + 6 truong): gop {n_rule2} cap")
        print(f"   Xung dot luat 1 vs 2         : {n_conflict} cum (da cat theo sdt)")
        print(f"   => SO NGUOI                  : {len(sizes)}")
        print(f"   => Nguoi kham > 1 lan        : {int((sizes > 1).sum())}")

    return pd.DataFrame({
        "row_id": np.arange(n),
        "id": df_raw["id"].to_numpy(),
        "patient_group": patient_group,
        "patient_uid": patient_uid,
        "n_visits": sizes.reindex(codes).to_numpy(),
        "has_phone": has_phone.to_numpy().astype(np.int8),
    })


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=repo / "data" / "Main_data_Merged_1835_Images.xlsx")
    p.add_argument("--out", type=Path, default=repo / "output" / "patient_identity.parquet")
    args = p.parse_args()

    print("=" * 78)
    print("DINH DANH BENH NHAN")
    print("=" * 78)
    df_raw = pd.read_excel(args.data)
    identity = build_identity(df_raw)

    assert identity["id"].is_unique, "Cot `id` bi trung — khong dung lam khoa duoc"
    assert identity["patient_uid"].nunique() < len(identity), "Khong ai kham nhieu lan?"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Chi xuat `id` + `patient_uid`: khong ten, khong so dien thoai -> commit duoc.
    identity[["id", "patient_uid"]].to_parquet(args.out, index=False)
    print(f"\nDa ghi {args.out}  {identity[['id', 'patient_uid']].shape}")
    print("   (chi gom `id` + `patient_uid` — khong chua PII)")


if __name__ == "__main__":
    main()

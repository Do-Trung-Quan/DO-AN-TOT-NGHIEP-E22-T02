"""Quy chu nghe tu do -> cap 'lv1'..'lv10' — CHEP NGUYEN VAN tu Fusion/.../scripts/patient_identity.py
(clean_job_text, classify_job_10_levels) va PAG/scripts_v3/v3_base.py (job_level, keep_na=True cho ca 4 cot).

CO CHU DICH khong dung `unidecode`: may hoc PG khong cai unidecode nen patient_identity chay nhanh NFKD.
Cai unidecode vao moi truong app KHONG anh huong module nay (da tat cung).
"""
import re
import unicodedata

import numpy as np
import pandas as pd

JOB_COLS = ["cviec", "pxuong", "cviec1", "cviec2"]


def _remove_vietnamese_accents(text: str) -> str:
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


def job_level(v):
    """v3_base.job_level(v, keep_na=True): o trong -> NA; con lai -> cap nghe."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan"):
        return np.nan
    return classify_job_10_levels(clean_job_text(v))


def them_cap_nghe(df: pd.DataFrame) -> pd.DataFrame:
    """Them cot <c>_lv cho 4 cot nghe (v3_base.add_job_levels)."""
    df = df.copy()
    for c in JOB_COLS:
        df[f"{c}_lv"] = (df[c] if c in df else pd.Series([np.nan] * len(df), index=df.index)).map(job_level)
    return df

"""Thu vien dung do thi PG ban 3 (PG_update muc 4 + chot hop nhom 04/11).

Khac pag_lib (ban 2):
  - Nut "Phoi nhiem/nghe": cviec, pxuong, cviec1, cviec2 KHONG dua chu tho (one-hot toi da 15 ten, bo gan het
    1038 gia tri cviec) ma dua 10 cap nghe theo muc bui `classify_job_10_levels` (scripts/patient_identity.py),
    dung bo gom da dung o nhanh timeseries (DO_AN.ipynb cell 7). Gia tri giu dang chu "lv1".."lv10" de
    ClinEncoder ma hoa one-hot, khong coi la bien lien tuc. Ca 4 cot: o trong -> NA (khong gop vao lv9; sua 06/11).
    Phep gom chi dung quy tac tu khoa, khong dung nhan -> tinh tren toan bo du lieu khong ro ri.
  - Hai bo cuc do thi: "global" (1 nut anh = C, trung binh 6 vector vung) va "vung" (6 nut vung).
  - Ma hoa fit trong fold, moi fold hai ban: "ngoai" (fit tren 4 fold train) va "noi" (fit tren 3 fold,
    chua fold validation noi bo (k+1)%5 — giong v3_baselines.py) de chon sieu tham so khong ro ri.
Cache chua du lieu that -> chi luu trong cache_v3, KHONG commit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_base  # noqa: E402
from v3_base import load_base_v3, CACHE3, RES3, MERG, JOB_COLS, JOB_LV, NGHE10  # noqa: E402,F401
from pag_lib import (GROUPS, GNAMES, CLIN_SLOTS, E1_PAIRS, E2_PAIRS, NONE, SELF, E1, E2, E3, E4, E5,  # noqa: E402,F401
                     encode_fold, NOT_USED)

LEVEL_NAME = {"lv1": "bụi nặng (khai thác, đá, nghiền, luyện...)", "lv2": "gia công, khoan, mài, xây",
              "lv3": "sửa chữa, cơ khí, hàn, điện", "lv4": "cân, đóng gói, phân loại", "lv5": "công nhân sản xuất chung",
              "lv6": "vận hành máy, lái xe", "lv7": "kiểm nghiệm, xét nghiệm", "lv8": "văn phòng, quản lý, dịch vụ",
              "lv9": "ghi \"không / ở nhà / tự do\"", "lv10": "không xác định"}
EDGE_NAME = {E1: "E1", E2: "E2", E3: "E3", E4: "E4", E5: "E5"}
LAYOUTS = ("global", "vung")
SLOT_NAME = ["RU", "RM", "RL", "LU", "LM", "LL"] + list(GNAMES) + ["Bối cảnh", "Bệnh nhân"]


def load_base_graph():
    """Du lieu cho do thi PG: luon dung 10 cap nghe (khong phu thuoc PG_NGHE10)."""
    B = load_base_v3()
    if not NGHE10:
        v3_base.add_job_levels(B)
    return B


def edge_template(layout: str, noi_vung: bool = True) -> np.ndarray:
    """Ma tran loai canh 14x14 (khe 0-5 anh, 6-11 lam sang, 12 boi canh, 13 benh nhan ao). 0 = khong canh.
    global: chi khe 0 la nut anh (C), khong co E1/E2."""
    n_img = 1 if layout == "global" else 6
    T = np.zeros((14, 14), np.int64)

    def s(i, j, t):
        T[i, j] = T[j, i] = t
    for i in range(13):
        s(13, i, E5)
    if layout == "vung" and noi_vung:
        for a, b in E1_PAIRS:
            s(a, b, E1)
        for a, b in E2_PAIRS:
            s(a, b, E2)
    for a in CLIN_SLOTS:
        for b in CLIN_SLOTS:
            if a != b:
                s(a, b, E3)
        for b in range(n_img):
            s(a, b, E4)
    return T


def edges_of(T: np.ndarray, present: np.ndarray):
    """Canh vo huong (i<j) giua hai nut cung co mat -> (edge_index (2,E), edge_type (E,))."""
    i, j = np.nonzero(np.triu(T, 1))
    keep = present[i] & present[j]
    return np.stack([i[keep], j[keep]]).astype(np.int64), T[i[keep], j[keep]].astype(np.int64)

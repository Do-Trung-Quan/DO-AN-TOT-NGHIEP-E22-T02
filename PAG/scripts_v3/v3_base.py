"""Du lieu chung cho viec (2) (PGS 27/10: chay lai toan bo baseline tren fold moi).
= pag_lib.load_base() (lam sang ma hoa trong fold nhu ban 2) nhung anh thay bang ban 3:
  BioViL-T dong bang tren anh V2 (che chu moi + xoa ngoai phoi), glob_a = C (trung binh 6 vector vung, chot o viec 1),
  reg_a = 6 vector vung (T2 dung, PCA 128 chung cho ca 6 vung = projection chung), area = dien tich vung ban 3.
Fold/dev/uid doi chieu voi cache_v3/p05_folds.npz (chuan duy nhat). Cache chua id -> PII, KHONG dua len GitHub.

Tu 04/11 (hop nhom): nut "Phoi nhiem/nghe" dung 10 cap nghe theo muc bui (`classify_job_10_levels` trong
scripts/patient_identity.py, cung bo gom cua nhanh timeseries) thay cho chu tho cviec/pxuong/cviec1/cviec2.
Bat mac dinh; dat bien moi truong PG_NGHE10=0 de tai hien so cu (chu tho). Ket qua chay voi 10 cap mang duoi TAG = "_nghe10"
de khong ghi de ket qua cu."""
import os
import sys
import numpy as np, pandas as pd
from v3_lib import CACHE3, RES3, run_config          # noqa: F401  (v3_lib them scripts_v2 vao sys.path)
from pag_lib import *                                 # noqa: F401,F403
from pag_lib import GROUPS

IMG3 = CACHE3 / "img_v3.npz"
FOLDS = CACHE3 / "p05_folds.npz"

NGHE10 = os.environ.get("PG_NGHE10", "1") == "1"
TAG = "_nghe10" if NGHE10 else ""
JOB_COLS = ["cviec", "pxuong", "cviec1", "cviec2"]
JOB_LV = [f"{c}_lv" for c in JOB_COLS]

sys.path.insert(0, str(MERG / "scripts"))
from patient_identity import classify_job_10_levels, clean_job_text, level_to_id  # noqa: E402,F401


def job_level(v, keep_na: bool):
    """Cap nghe 'lv1'..'lv10' (dang chu -> ClinEncoder ma hoa one-hot). keep_na: o trong giu NA (cviec1/cviec2)."""
    if keep_na and (v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan")):
        return np.nan
    return classify_job_10_levels(clean_job_text(v))


def add_job_levels(B):
    """Them 4 cot cap nghe vao B['raw'] va thay 4 cot chu tho trong nhom 'Phoi nhiem/nghe' (pag_lib.GROUPS).
    Chi dung quy tac tu khoa, khong dung nhan -> tinh tren toan bo du lieu khong ro ri."""
    raw = B["raw"]
    for c in JOB_COLS:
        raw[f"{c}_lv"] = raw[c].map(lambda v, k=c in ("cviec1", "cviec2"): job_level(v, k))
    g = GROUPS["Phơi nhiễm/nghề"]
    if JOB_LV[0] not in g:
        GROUPS["Phơi nhiễm/nghề"] = [c for c in g if c not in JOB_COLS] + JOB_LV
    return B


def load_base_v3():
    B = load_base(); core = load_core()
    P5 = np.load(FOLDS, allow_pickle=True)
    assert (P5["id"] == core.id.to_numpy()).all()
    assert (P5["fold2"].astype(int) == B["fold"]).all() and (P5["dev2"].astype(bool) == B["dev"]).all()
    assert (pd.factorize(P5["uid2"])[0] == pd.factorize(B["uid"])[0]).all()
    Z = np.load(IMG3, allow_pickle=False)
    pos = pd.Index(Z["img_id"]).get_indexer(core.id); hi = B["has_img"]
    assert ((pos >= 0) == hi).all(), "anh ban 3 / has_img lech"
    def put(a):
        out = np.zeros((len(hi),) + a.shape[1:], np.float32); out[hi] = a[pos[hi]].astype(np.float32); return out
    for k in ("b", "c", "d"): B.pop(f"reg_{k}"); B.pop(f"glob_{k}")
    B["reg_a"] = put(Z["regf_biovil_V2"]); B["glob_a"] = B["reg_a"].mean(1); B["area"] = put(Z["area"])
    if NGHE10:
        add_job_levels(B)
    return B

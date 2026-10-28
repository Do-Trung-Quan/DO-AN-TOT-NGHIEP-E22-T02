"""Du lieu chung cho viec (2) (PGS 27/10: chay lai toan bo baseline tren fold moi).
= pag_lib.load_base() (lam sang ma hoa trong fold nhu ban 2) nhung anh thay bang ban 3:
  BioViL-T dong bang tren anh V2 (che chu moi + xoa ngoai phoi), glob_a = C (trung binh 6 vector vung, chot o viec 1),
  reg_a = 6 vector vung (T2 dung, PCA 128 chung cho ca 6 vung = projection chung), area = dien tich vung ban 3.
Fold/dev/uid doi chieu voi cache_v3/p05_folds.npz (chuan duy nhat). Cache chua id -> PII, KHONG dua len GitHub."""
import numpy as np, pandas as pd
from v3_lib import CACHE3, RES3, run_config          # noqa: F401  (v3_lib them scripts_v2 vao sys.path)
from pag_lib import *                                 # noqa: F401,F403

IMG3 = CACHE3 / "img_v3.npz"
FOLDS = CACHE3 / "p05_folds.npz"


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
    return B

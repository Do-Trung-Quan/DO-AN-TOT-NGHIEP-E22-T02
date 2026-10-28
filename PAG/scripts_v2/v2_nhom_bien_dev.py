"""Do muc quan trong cua tung nhom bien lam sang (chi tren dev, 5 fold theo nguoi, hoi quy logistic).
In AUC khi (a) chi dung 1 nhom, (b) bo 1 nhom khoi bo day du. Khong cham test. Chi in so tong hop."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pag_lib import load_base, encode_fold, GNAMES

B = load_base()
dev, fold, y = B["dev"], B["fold"], B["y"]
NAMES = GNAMES + ["Bối cảnh"]
IMG = B["has_img"]


def oof(groups, mask):
    """AUC gop OOF tren cac dong dev thoa `mask`, chi dung cac nhom lam sang `groups`."""
    p = np.full(len(y), np.nan)
    for f in sorted(set(fold[dev])):
        tr, va = dev & (fold != f), dev & (fold == f)
        E = encode_fold(B, tr, image=None)
        X = np.concatenate([E["clin"][g] for g in groups], 1)
        m = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
    k = dev & mask
    return roc_auc_score(y[k], p[k])


for label, mask in [("TOAN BO dev", np.ones(len(y), bool)), ("dev KHONG anh", ~IMG)]:
    full = oof(NAMES, mask)
    print(f"\n== {label}: bo day du = {full:.3f}")
    print(f"{'nhom':28s} {'chi nhom nay':>13s} {'bo nhom nay':>12s} {'delta':>7s}")
    for g in NAMES:
        a = oof([g], mask)
        d = oof([x for x in NAMES if x != g], mask)
        print(f"{g:28s} {a:13.3f} {d:12.3f} {d - full:+7.3f}")
    sys.stdout.flush()

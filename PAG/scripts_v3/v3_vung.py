"""Viec (1) PGS 27/10: dac trung vung thap hon toan anh do MAT THONG TIN hay do QUA NHIEU CHIEU? (BioViL-T dong bang, chi V2)
Chay tren fold moi sau P0.5 (dev2/fold2/uid2, cache_v3/p05_folds.npz). Cung mo hinh L1 (hoi quy logistic, chuan hoa, C=0,1, can lop).
So 5 cach bieu dien (A, C-F; ten giu nguyen):
  A toan anh img_embedding                         512   - moc hien tai. Da kiem: img_embedding = trung binh patch_embeddings
                                                           (BioViL-T tra pooled_x, chua qua lop chieu) -> cung khong gian voi vung;
                                                           ban dau co them B = gpool de tach "lop chieu", nhung B trung khop A nen bo.
  C trung binh 6 vector vung                       512   - PGS yeu cau
  D 6 vung noi lai                                3072   - moc
  E 6 vung + PCA 32 chieu/vung                     192   - PGS yeu cau; PCA fit rieng tung vung, CHI tren phan huan luyen cua fold
  F 6 vung + PCA 64 chieu/vung                     384   - nhu E
Chi so: AUC trong lo + KTC, Delta(lo + anh so voi chi lo) + KTC, hieu ghep cap AUC trong lo giua cac cach. Bootstrap theo uid2.
Test KHONG mo. Chi ghi so lieu gop.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from v3_lib import *
from v3_lib import CACHE3, run_config

OUT = "tuan1_phase0/viec1_vung_vs_toan_anh_V2"
core = load_core()
P5 = np.load(CACHE3 / "p05_folds.npz", allow_pickle=True)
assert (P5["id"] == core.id.to_numpy()).all()
uid, dev, fold = P5["uid2"], P5["dev2"].astype(bool), P5["fold2"].astype(int)
Z = np.load(CACHE3 / "img_v3.npz", allow_pickle=False)
pos = pd.Series(np.arange(len(Z["img_id"])), index=Z["img_id"])
row = core.id.map(pos); hi = core.has_img.to_numpy() & row.notna().to_numpy()
y, batch = core.y.to_numpy(), core.batch.to_numpy()
m = dev & hi
D = Z["regf_biovil_V2"].shape[-1]


def take(a):
    out = np.zeros((len(core),) + a.shape[1:], np.float32); out[hi] = a[row[hi].astype(int)].astype(np.float32); return out.reshape(len(core), -1)


reg = take(Z["regf_biovil_V2"])                                              # N x (6*D)
assert np.array_equal(Z["glob_biovil_V2"], Z["gpool_biovil_V2"])          # img_embedding = trung binh moi patch
FEAT = {"A — Toàn ảnh (img_embedding = trung bình mọi patch)": (take(Z["glob_biovil_V2"]), None),
        "C — Trung bình 6 vector vùng": (reg.reshape(len(core), 6, D).mean(1), None),
        "D — 6 vùng nối lại": (reg, None),
        "E — 6 vùng + PCA 32 chiều/vùng": (reg, 32),
        "F — 6 vùng + PCA 64 chiều/vùng": (reg, 64)}

oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(batch[hi].reshape(-1, 1))
Bx = np.zeros((len(core), len(oh.categories_[0])), np.float32); Bx[hi] = oh.transform(batch[hi].reshape(-1, 1))


def maker(k, extra=0):
    """L1 = lr(); voi PCA: PCA rieng tung vung (fit trong fold), cot lo (neu co) di thang."""
    if k is None: return lr
    def make():
        parts = [(f"vung{j}", PCA(k, random_state=SEED), slice(j * D, (j + 1) * D)) for j in range(6)]
        if extra: parts.append(("lo", "passthrough", slice(6 * D, 6 * D + extra)))
        return make_pipeline(ColumnTransformer(parts), StandardScaler(),
                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=0))
    return make


yi, ui, bi = y[m], uid[m], batch[m]
pb = oof_cv(Bx, y, fold, m)[m]
res, P = [], {}
for name, (X, k) in FEAT.items():
    p = oof_cv(X, y, fold, m, make=maker(k))[m]
    pbi = oof_cv(np.column_stack([X, Bx]), y, fold, m, make=maker(k, Bx.shape[1]))[m]
    P[name] = p
    a, alo, ahi, _ = boot_ci(lambda i: roc_auc_score(yi[i], p[i]), ui)
    w, wlo, whi, _ = boot_ci(lambda i: within_batch_auc(yi[i], p[i], bi[i]), ui)
    d, dlo, dhi, _ = boot_ci(lambda i: roc_auc_score(yi[i], pbi[i]) - roc_auc_score(yi[i], pb[i]), ui)
    res.append(dict(cach=name, so_chieu=(6 * k if k else X.shape[1]), L1_auc=a, L1_ci=f"[{alo:.3f}; {ahi:.3f}]",
                    auc_trong_lo=w, auc_trong_lo_ci=f"[{wlo:.3f}; {whi:.3f}]", delta=d, delta_ci=f"[{dlo:+.3f}; {dhi:+.3f}]"))
    print(f"{name}: {res[-1]['so_chieu']} chieu | trong lo {w:.3f} [{wlo:.3f}; {whi:.3f}] | delta {d:+.3f} [{dlo:+.3f}; {dhi:+.3f}]", flush=True)
R_ = pd.DataFrame(res)

names = list(FEAT)
pairs = [(1, 0, "Chỉ lấy vùng phổi, gộp trung bình (cùng không gian, cùng số chiều)"),
         (2, 1, "Giữ riêng 6 vùng thay vì gộp (C là phép chiếu tuyến tính của D → D chứa đủ thông tin của C)"),
         (3, 2, "Giảm chiều PCA 32 so với 6 vùng nối lại"),
         (4, 2, "Giảm chiều PCA 64 so với 6 vùng nối lại"),
         (3, 1, "PCA 32 so với trung bình 6 vùng"),
         (4, 1, "PCA 64 so với trung bình 6 vùng"),
         (2, 0, "6 vùng nối lại so với toàn ảnh (khoảng cách cần giải thích)")]
cmp = []
for i, j, why in pairs:
    pi, pj = P[names[i]], P[names[j]]
    d, lo, hi_, v = boot_ci(lambda k: within_batch_auc(yi[k], pi[k], bi[k]) - within_batch_auc(yi[k], pj[k], bi[k]), ui)
    cmp.append(dict(so_sanh=f"{names[i][0]} − {names[j][0]}", y_nghia=why, hieu_auc_trong_lo=d, ktc=f"[{lo:+.3f}; {hi_:+.3f}]", P_hieu_duong=float((v > 0).mean())))
C_ = pd.DataFrame(cmp)

nb = len(set(bi)); nb2 = sum(1 for b in set(bi) if 0 < y[m][bi == b].sum() < (bi == b).sum())
Rp = Report(OUT, "Việc (1) — Đặc trưng vùng thấp hơn toàn ảnh: do mất thông tin hay do quá nhiều chiều? (BioViL-T đóng băng, V2)")
Rp.text(f"Tập: **dev mới sau P0.5** (ghép người, chia lại fold), có ảnh: {int(m.sum())} lượt, {int(yi.sum())} dương, {len(set(ui))} người, {nb} lô "
        f"({nb2} lô đủ hai nhãn dùng cho AUC trong lô). Mô hình L1 = hồi quy logistic (chuẩn hóa, C=0,1, cân lớp), 5 fold mới theo người. "
        "PCA fit riêng từng vùng, **chỉ trên phần huấn luyện của mỗi fold**. KTC = khoảng tin cậy 95% bootstrap theo người (1000 lần). Tập test không mở.\n\n"
        "Đã kiểm tra: vector toàn ảnh `img_embedding` của BioViL-T đúng bằng trung bình mọi `patch_embeddings` (chưa qua lớp chiếu), "
        "tức cùng không gian với vector vùng — nên A, C, D khác nhau chỉ ở cách gộp patch. C (trung bình 6 vùng) là phép chiếu tuyến tính của D "
        "(6 vùng nối lại): mọi thông tin trong C đều có trong D.")
Rp.table(R_, "1_ket_qua", "AUC trong lô và Δ theo cách biểu diễn",
         note="*`auc_trong_lo`: ROC-AUC (xác suất xếp ca dương cao hơn ca âm; 0,5 = đoán bừa) tính riêng trong từng lô đủ hai nhãn, trung bình theo cỡ lô. "
              "`delta`: AUC(lô + ảnh) − AUC(chỉ lô); KTC trọn trên 0 = ảnh thêm thông tin ngoài lô. `L1_auc`: AUC gộp.*")
Rp.table(C_, "2_so_sanh_ghep_cap", "So sánh ghép cặp AUC trong lô (cùng người, cùng mẫu bootstrap)",
         note="*`hieu_auc_trong_lo`: AUC trong lô cách trước − cách sau. KTC chứa 0 = chưa phân biệt được. `P_hieu_duong`: tỉ lệ lần bootstrap hiệu > 0.*")
Rp.save()
run_config(OUT, __file__, data=[CACHE3 / "img_v3.npz", CACHE3 / "p05_folds.npz", FUS / "fusion_dataset_ketqua_frozen.npz"],
           encoder="BioViL-T dong bang, V2", fold="fold2 (P0.5), group = uid2", mo_hinh="StandardScaler + LogisticRegression(C=0.1, balanced)",
           pca="PCA(k, random_state=SEED) rieng tung vung, fit tren phan train cua fold; k in {32, 64}", bootstrap=f"theo uid2, {NBOOT} lan, seed {SEED}")
pd.set_option("display.width", 250); print(C_.to_string(index=False))

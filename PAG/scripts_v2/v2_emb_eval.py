"""PAG_V2 §7 — Nhanh embedding + phan loai co dien + truy hoi ca tuong tu (kNN). Chi dev OOF.

Hai bo vector:
  E1  = embedding CLS cua PG (cache/emb_PG.npz, fold k lay tu mo hinh khong thay fold k)
  tho = dac trung tho: lam sang da ma hoa + vector anh toan cuc BioViL-T (PCA 128 trong fold) + co/khong phim
Phan loai: LR, SVM (RBF), kNN — sieu tham so chon bang CV noi bo 3 phan theo NGUOI trong 4 fold train (khong cham fold bao cao).
Truy hoi: moi lan kham o fold k tim 20 nguoi gan nhat (cosin) trong 4 fold con lai; Precision@5/10/20 so voi ti le nen;
kiem tra lang gieng co bi don theo tinh / theo viec co phim.
E2 (GraphCL) va E3 (Weisfeiler–Lehman kernel) KHONG chay: moi do thi co cung mot cau truc 14 node (chi khac dac trung node),
nen WL kernel suy bien ve so khop dac trung; E2 de ngo (tuy chon trong PDF).
Ghi PAG/results/tuan5_nhanh_embedding. Chi so lieu gop.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from pag_lib import *

B = load_base()
y, fold, hi, uid, dev = B["y"], B["fold"], B["has_img"], B["uid"], B["dev"]
tinh = pd.Series(B["tinh"]).astype(str).to_numpy()
EMB = np.load(CACHE / "emb_PG.npz")
N = len(y)

def feats(kind, k, tr):
    if kind == "E1": return EMB[f"emb_{k}"]
    E = encode_fold(B, tr, "a", region=False)
    return np.concatenate([E["flat_clin"], E["img"][:, 0], hi[:, None].astype(np.float32)], 1)

CLF = {
    "LR":  (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")), {"logisticregression__C": [0.01, 0.1, 1]}),
    "SVM": (lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")), {"svc__C": [0.3, 1, 3]}),
    "kNN": (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(metric="cosine", weights="distance")), {"kneighborsclassifier__n_neighbors": [10, 25, 50, 100]}),
}
KS = (5, 10, 20)
oof = {(f, c): np.full(N, np.nan) for f in ("E1", "tho") for c in CLF}
chosen, ret = [], []
for k in range(5):
    tr, va = dev & (fold != k) & (fold >= 0), dev & (fold == k)
    for f in ("E1", "tho"):
        X = feats(f, k, tr)
        for c, (mk, grid) in CLF.items():
            gs = GridSearchCV(mk(), grid, scoring="roc_auc", cv=GroupKFold(3), n_jobs=-1).fit(X[tr], y[tr], groups=uid[tr])
            m = gs.best_estimator_
            oof[(f, c)][va] = m.decision_function(X[va]) if hasattr(m, "decision_function") else m.predict_proba(X[va])[:, 1]
            chosen.append(dict(fold=k, vector=f, bo_phan_loai=c, **{kk.split("__")[1]: vv for kk, vv in gs.best_params_.items()}))
        # truy hoi: cosin tren vector da chuan hoa theo train
        sc = StandardScaler().fit(X[tr]); Xt, Xv = normalize(sc.transform(X[tr])), normalize(sc.transform(X[va]))
        nb = NearestNeighbors(n_neighbors=max(KS), metric="cosine").fit(Xt).kneighbors(Xv, return_distance=False)
        ytr, htr, ttr = y[tr][nb], hi[tr][nb], tinh[tr][nb]
        prev_tr = y[tr].mean()
        share_t = pd.Series(tinh[tr]).value_counts(normalize=True)
        yv, hv, tv = y[va], hi[va], tinh[va]
        for K in KS:
            for g, gm in (("có ảnh", hv), ("không ảnh", ~hv)):
                for lab in (1, 0):
                    q = gm & (yv == lab)
                    if q.sum() == 0: continue
                    ret.append(dict(fold=k, vector=f, k=K, nhom_truy_van=g, nhan_truy_van=lab, n=int(q.sum()),
                                    ti_le_lang_gieng_duong=ytr[q, :K].mean(), ti_le_duong_nen_train=prev_tr,
                                    ti_le_duong_nen_nhom=y[tr & (hi == (g == "có ảnh"))].mean(),
                                    cung_tinh=(ttr[q, :K] == tv[q, None]).mean(), cung_tinh_ky_vong=share_t.reindex(tv[q]).fillna(0).mean(),
                                    cung_trang_thai_phim=(htr[q, :K] == hv[q, None]).mean(),
                                    cung_trang_thai_phim_ky_vong=np.where(hv[q], hi[tr].mean(), 1 - hi[tr].mean()).mean()))
    print(f"fold {k} xong", flush=True)

R = Report("tuan5_nhanh_embedding", "Tuần 5 — Nhánh embedding + phân loại cổ điển + truy hồi ca tương tự (PAG_V2 §7)")
R.text("**E1** = vector của node bệnh nhân (CLS) ở lớp cuối PG; ở fold k vector lấy từ mô hình PG huấn luyện trên 4 fold còn lại (1 seed, cùng cấu hình đã chọn). "
       "**Thô** = đặc trưng lâm sàng đã mã hoá + vector ảnh toàn cục BioViL-T (PCA 128 trong fold) + cờ có phim — mốc so sánh. "
       "Siêu tham số của LR/SVM/kNN chọn bằng CV nội bộ 3 phần theo người trong 4 fold train. "
       "E2 (tự giám sát GraphCL) và E3 (graph kernel Weisfeiler–Lehman) **không chạy**: mọi đồ thị bệnh nhân có cùng một khung 14 node, "
       "khác nhau chỉ ở đặc trưng node, nên graph kernel suy biến về so khớp đặc trưng (≈ kNN trên vector thô); E2 là tuỳ chọn.")
rows = []
pg = np.load(CACHE / "run_PG.npz")["oof"]
for g, gm in (("có ảnh", dev & hi), ("không ảnh", dev & ~hi)):
    a, lo, h_, _ = boot_ci(lambda i: roc_auc_score(y[gm][i], pg[gm][i]), uid[gm], n=300)
    rows.append(dict(vector="—", bo_phan_loai="đầu PG (3 seed)", nhom=g, roc_auc=a, ci=f"[{lo:.3f}–{h_:.3f}]"))
    for (f, c), p in oof.items():
        a, lo, h_, _ = boot_ci(lambda i: roc_auc_score(y[gm][i], p[gm][i]), uid[gm], n=300)
        rows.append(dict(vector=f, bo_phan_loai=c, nhom=g, roc_auc=a, ci=f"[{lo:.3f}–{h_:.3f}]"))
R.table(pd.DataFrame(rows), "phan_loai", "Phân loại từ vector (ROC-AUC dev OOF, KTC 95% bootstrap theo người 300 lần)",
        note="*ROC-AUC = xác suất xếp một ca dương cao hơn một ca âm (0,5 = đoán bừa). Dùng điểm thô của bộ phân loại (AUC không cần hiệu chỉnh). "
             "PDF dự báo E1 + LR/SVM thường không hơn chính đầu phân loại của PG.*")
RT = pd.DataFrame(ret)
agg = RT.groupby(["vector", "k", "nhom_truy_van", "nhan_truy_van"]).apply(
    lambda d: pd.Series({c: np.average(d[c], weights=d.n) for c in RT.columns if c not in ("fold", "vector", "k", "nhom_truy_van", "nhan_truy_van", "n")} | {"n": d.n.sum()})).reset_index()
R.table(agg, "truy_hoi_knn", "Truy hồi kNN: k người gần nhất trong fold train của mỗi lượt khám ở fold validation",
        note="*`ti_le_lang_gieng_duong` = Precision@k cho truy vấn dương (nhan_truy_van=1): tỉ lệ láng giềng cũng dương; với truy vấn âm (=0) thì 1 − giá trị này là tỉ lệ láng giềng cùng nhãn. "
             "So với `ti_le_duong_nen_train` (tỉ lệ dương toàn train) và `ti_le_duong_nen_nhom` (tỉ lệ dương trong train cùng nhóm phim). "
             "`cung_tinh` = tỉ lệ láng giềng cùng tỉnh với truy vấn, so với `cung_tinh_ky_vong` (nếu chọn láng giềng ngẫu nhiên). "
             "`cung_trang_thai_phim` = tỉ lệ láng giềng cùng có/không phim với truy vấn, so với kỳ vọng ngẫu nhiên. Cao hơn nhiều so với kỳ vọng nghĩa là vector bị “dồn” theo tỉnh / theo phim.*")
R.table(pd.DataFrame(chosen), "sieu_tham_so_chon", "Siêu tham số chọn trong từng fold (CV nội bộ)")
R.save()
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(pd.DataFrame(rows).to_string(index=False))
    print(agg[agg.nhan_truy_van == 1].round(3).to_string(index=False))
print("EMB_EVAL XONG")

"""Viec (1) lan 2 (PGS 27/10, sau khi duyet P0.5): chay lai vung vs toan anh voi
  - C cua hoi quy logistic tinh chinh bang inner CV cho TUNG bieu dien (luoi C = 1e-6..10, 15 diem nua bac;
    lan chay dau luoi 1e-4..10 thi F chon dung mep duoi o ca 5 fold -> mo rong),
    inner CV = StratifiedGroupKFold(5) theo uid2 tren phan huan luyen cua fold ngoai, tieu chi ROC-AUC (user chon);
  - them S = stacking: 6 LR theo vung (C moi vung chinh bang inner CV), diem out-of-fold nested (inner CV tren phan
    huan luyen), roi 1 LR (C chinh bang inner CV) tren 6 diem. So voi C (trung binh 6 vung).
Bieu dien: A toan anh, C trung binh 6 vung, D 6 vung noi lai, E/F 6 vung + PCA 32/64 moi vung (PCA fit trong fold), S stacking.
Delta: moi mo hinh chay lai voi cot lo (one-hot) them vao, C cung chinh lai. Mo hinh CHI LO giu C = 0,1 co dinh: AUC inner CV
  cua no gan nhu khong doi theo C (thu tu 18 lo khong phu thuoc C), nen inner CV chon C ngau nhien (lan dau 3e-4..3,2) va C rat nho
  nen diem moi fold ve sat 0,5 voi lech khac nhau -> AUC gop cua moc tut 0,707 -> 0,674, lam Delta phong dai gia ~0,03.
Fold ngoai = fold2 (P0.5, chuan duy nhat). Test KHONG mo. Chi ghi so lieu gop.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from v3_lib import *
from v3_lib import CACHE3, run_config

OUT = "tuan1_phase0/viec1b_vung_tinh_chinh_C_stacking_V2"
GRID = np.logspace(-6, 1, 15)
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
FEAT = {"A — Toàn ảnh (img_embedding = trung bình mọi patch)": (take(Z["glob_biovil_V2"]), None),
        "C — Trung bình 6 vector vùng": (reg.reshape(len(core), 6, D).mean(1), None),
        "D — 6 vùng nối lại": (reg, None),
        "E — 6 vùng + PCA 32 chiều/vùng": (reg, 32),
        "F — 6 vùng + PCA 64 chiều/vùng": (reg, 64)}
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(batch[hi].reshape(-1, 1))
Bx = np.zeros((len(core), len(oh.categories_[0])), np.float32); Bx[hi] = oh.transform(batch[hi].reshape(-1, 1))


def base(k=None, extra=0):
    """chuan hoa (+ PCA rieng tung vung neu k) + LR; C de inner CV chinh."""
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0)
    if k is None: return make_pipeline(StandardScaler(), clf)
    parts = [(f"vung{j}", PCA(k, random_state=SEED), slice(j * D, (j + 1) * D)) for j in range(6)]
    if extra: parts.append(("lo", "passthrough", slice(6 * D, 6 * D + extra)))
    return make_pipeline(ColumnTransformer(parts), StandardScaler(), clf)


def inner(tr):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(tr.sum()), y[tr], uid[tr]))


def tuned(est, X, yy, cv):
    g = GridSearchCV(est, {"logisticregression__C": GRID}, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True).fit(X, yy)
    return g.best_estimator_, g.best_params_["logisticregression__C"]


def run_plain(X, k, extra=0):
    p, Cs = np.full(len(y), np.nan), []
    for f in range(5):
        tr, va = m & (fold != f), m & (fold == f)
        est, c = tuned(base(k, extra), X[tr], y[tr], inner(tr)); Cs.append(c)
        p[va] = est.predict_proba(X[va])[:, 1]
    return p, Cs


def run_stack(with_batch):
    """6 LR theo vung -> diem OOF nested (inner CV tren phan train) -> LR tren 6 diem (+ lo)."""
    p, Cs = np.full(len(y), np.nan), []
    for f in range(5):
        tr, va = m & (fold != f), m & (fold == f)
        cv = inner(tr); ytr = y[tr]
        S_tr, S_va, creg = np.zeros((tr.sum(), 6)), np.zeros((va.sum(), 6)), []
        for j in range(6):
            Xj = reg[:, j * D:(j + 1) * D]
            est, c = tuned(base(), Xj[tr], ytr, cv); creg.append(c)
            for a, b in cv:                                            # diem OOF nested tren phan train, C da chon
                e = base().set_params(logisticregression__C=c).fit(Xj[tr][a], ytr[a])
                S_tr[b, j] = e.decision_function(Xj[tr][b])
            S_va[:, j] = est.decision_function(Xj[va])                 # est da refit tren toan phan train
        Mtr, Mva = (np.column_stack([S_tr, Bx[tr]]), np.column_stack([S_va, Bx[va]])) if with_batch else (S_tr, S_va)
        meta, cm = tuned(base(), Mtr, ytr, cv)
        p[va] = meta.predict_proba(Mva)[:, 1]; Cs.append(dict(C_vung=[float(c) for c in creg], C_meta=float(cm)))
    return p, Cs


yi, ui, bi = y[m], uid[m], batch[m]
pb = oof_cv(Bx, y, fold, m, make=lambda: lr(0.1))[m]
print("chi lo (C=0,1 co dinh): AUC gop", round(roc_auc_score(yi, pb), 3), flush=True)
res, P, CHOSEN = [], {}, {}
SNAME = "S — Stacking: 6 LR theo vùng → LR trên 6 điểm"
for name in list(FEAT) + [SNAME]:
    if name == SNAME:
        p, cs = run_stack(False); pbi, cs2 = run_stack(True); dim = "6 điểm"
    else:
        X, k = FEAT[name]
        p, cs = run_plain(X, k); pbi, cs2 = run_plain(np.column_stack([X, Bx]), k, Bx.shape[1] if k else 0)
        dim = 6 * k if k else X.shape[1]
    p, pbi = p[m], pbi[m]; P[name] = p; CHOSEN[name] = cs; CHOSEN[name + " (+ lô)"] = cs2
    a, alo, ahi, _ = boot_ci(lambda i: roc_auc_score(yi[i], p[i]), ui)
    w, wlo, whi, _ = boot_ci(lambda i: within_batch_auc(yi[i], p[i], bi[i]), ui)
    d, dlo, dhi, _ = boot_ci(lambda i: roc_auc_score(yi[i], pbi[i]) - roc_auc_score(yi[i], pb[i]), ui)
    edge = (sum(c["C_meta"] in (GRID[0], GRID[-1]) or any(v in (GRID[0], GRID[-1]) for v in c["C_vung"]) for c in cs) if name == SNAME
            else sum(c in (GRID[0], GRID[-1]) for c in cs))
    res.append(dict(cach=name, so_chieu=dim, C_chon_5_fold=("xem bảng 3" if name == SNAME else ", ".join(f"{c:g}" for c in cs)),
                    so_fold_C_o_mep_luoi=int(edge),
                    L1_auc=a, L1_ci=f"[{alo:.3f}; {ahi:.3f}]", auc_trong_lo=w, auc_trong_lo_ci=f"[{wlo:.3f}; {whi:.3f}]",
                    delta=d, delta_ci=f"[{dlo:+.3f}; {dhi:+.3f}]"))
    print(f"{name}: L1 {a:.3f} | trong lo {w:.3f} [{wlo:.3f}; {whi:.3f}] | delta {d:+.3f} [{dlo:+.3f}; {dhi:+.3f}] | C {cs}", flush=True)
R_ = pd.DataFrame(res)

names = list(P)
pairs = [(1, 0, "Chỉ lấy vùng phổi, gộp trung bình"),
         (5, 1, "Stacking 6 vùng so với trung bình 6 vùng (so sánh thầy yêu cầu)"),
         (2, 1, "6 vùng nối lại so với trung bình 6 vùng"),
         (3, 1, "PCA 32 so với trung bình 6 vùng"),
         (4, 1, "PCA 64 so với trung bình 6 vùng"),
         (3, 2, "PCA 32 so với 6 vùng nối lại"),
         (4, 2, "PCA 64 so với 6 vùng nối lại"),
         (5, 0, "Stacking so với toàn ảnh")]
cmp = []
for i, j, why in pairs:
    pi, pj = P[names[i]], P[names[j]]
    d, lo, hi_, v = boot_ci(lambda k: within_batch_auc(yi[k], pi[k], bi[k]) - within_batch_auc(yi[k], pj[k], bi[k]), ui)
    cmp.append(dict(so_sanh=f"{names[i][0]} − {names[j][0]}", y_nghia=why, hieu_auc_trong_lo=d, ktc=f"[{lo:+.3f}; {hi_:+.3f}]", P_hieu_duong=float((v > 0).mean())))
C_ = pd.DataFrame(cmp)
S_ = pd.DataFrame([dict(fold=f, **{f"C_vung{j + 1}": c["C_vung"][j] for j in range(6)}, C_meta=c["C_meta"],
                        C_meta_co_lo=CHOSEN[SNAME + " (+ lô)"][f]["C_meta"]) for f, c in enumerate(CHOSEN[SNAME])])

nb = len(set(bi)); nb2 = sum(1 for b in set(bi) if 0 < yi[bi == b].sum() < (bi == b).sum())
Rp = Report(OUT, "Việc (1) lần 2 — Vùng so với toàn ảnh: tinh chỉnh C bằng inner CV và thêm stacking (BioViL-T đóng băng, V2)")
Rp.text(f"Tập: dev mới sau P0.5, có ảnh: {int(m.sum())} lượt, {int(yi.sum())} dương, {len(set(ui))} người, {nb} lô ({nb2} lô đủ hai nhãn). "
        "Fold ngoài = 5 fold mới (P0.5). **C của hồi quy logistic** (độ mạnh điều chuẩn; C nhỏ = mô hình bị kìm nhiều hơn) chọn riêng cho từng "
        f"biểu diễn và từng fold ngoài bằng **inner CV**: 5 fold con theo người trên phần huấn luyện, lưới C = {', '.join(f'{c:g}' for c in GRID)}, "
        "tiêu chí ROC-AUC gộp; mô hình có cột lô (dùng tính Δ) và mô hình chỉ lô cũng được chọn C như vậy. PCA fit riêng từng vùng trong fold.\n\n"
        "**S — Stacking:** trong mỗi fold ngoài, 6 hồi quy logistic, mỗi cái chỉ thấy một vùng (C chọn bằng inner CV); điểm của chúng trên phần "
        "huấn luyện lấy **out-of-fold nested** (mỗi điểm do mô hình chưa thấy lượt đó, theo 5 fold con); rồi một hồi quy logistic (C chọn bằng inner CV) "
        "học trên 6 điểm. Để tính Δ, hồi quy cuối nhận thêm cột lô. KTC = khoảng tin cậy 95% bootstrap theo người (1000 lần). Tập test không mở.")
Rp.table(R_, "1_ket_qua", "AUC trong lô và Δ theo cách biểu diễn (C đã tinh chỉnh)",
         note="*`auc_trong_lo`: ROC-AUC (xác suất xếp ca dương cao hơn ca âm; 0,5 = đoán bừa) tính riêng trong từng lô đủ hai nhãn, trung bình theo cỡ lô. "
              "`delta`: AUC(lô + ảnh) − AUC(chỉ lô). `L1_auc`: AUC gộp. `C_chon_5_fold`: C được inner CV chọn ở 5 fold ngoài.*")
Rp.table(C_, "2_so_sanh_ghep_cap", "So sánh ghép cặp AUC trong lô (cùng người, cùng mẫu bootstrap)",
         note="*`hieu_auc_trong_lo`: AUC trong lô cách trước − cách sau. KTC chứa 0 = chưa phân biệt được. `P_hieu_duong`: tỉ lệ lần bootstrap hiệu > 0.*")
Rp.table(S_, "3_C_stacking", "C được chọn trong stacking (theo fold ngoài)")
Rp.text(f"Mốc Δ (mô hình chỉ lô) giữ C = 0,1 cố định, AUC gộp = {roc_auc_score(yi, pb):.3f}. Lý do: AUC inner CV của nó gần như không đổi theo C "
        "(thứ tự 18 lô không phụ thuộc C), nên inner CV chọn C gần như ngẫu nhiên; khi chọn C rất nhỏ, điểm mỗi fold dồn sát 0,5 với độ lệch khác nhau, "
        "AUC gộp của mốc tụt (thử: 0,707 → 0,674) và Δ bị phóng đại giả khoảng 0,03. "
        "Lưới C mở rộng xuống 1e-6 vì ở lần chạy thử với lưới 1e-4..10, F chọn đúng mép dưới ở cả 5 fold.")
Rp.save()
run_config(OUT, __file__, data=[CACHE3 / "img_v3.npz", CACHE3 / "p05_folds.npz", FUS / "fusion_dataset_ketqua_frozen.npz"],
           encoder="BioViL-T dong bang, V2", fold_ngoai="fold2 (P0.5), group = uid2",
           inner_cv="StratifiedGroupKFold(5, shuffle=True, random_state=SEED) theo uid2 tren phan train; GridSearchCV scoring=roc_auc",
           luoi_C=[float(c) for c in GRID], mo_hinh="StandardScaler + LogisticRegression(balanced, max_iter=5000)",
           pca="PCA(k, random_state=SEED) rieng tung vung trong pipeline (fit lai trong moi fold con)",
           stacking="6 LR theo vung (C tinh chinh) -> decision_function OOF nested tren 5 fold con -> LR meta (C tinh chinh) [+ one-hot lo cho delta]",
           moc_delta="chi lo, LR C=0.1 co dinh", bootstrap=f"theo uid2, {NBOOT} lan, seed {SEED}", C_chon={k: v for k, v in CHOSEN.items()})
pd.set_option("display.width", 250); print(C_.to_string(index=False))

"""P0.1 ban 3 (PG_update §2): chay lai kiem tra nhieu theo lo chup tren anh da xu ly V1, V2; so voi dac trung cu.
4 chi so: probe du doan lo tu embedding · AUC trong tung lo · Δ(lo + anh so voi lo) · LOBO (de nguyen mot lo).
Them phan tich do nhay loai 55 nguoi bnn=Co & ketqua=2 (P0.4). Chi dung dev goc (split/fold_id/patient_uid cua file fold, chua ghep nguoi); test KHONG mo.
Chi ghi so lieu gop. Luu diem OOF L1 vao cache_v3 cho P0.4.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from v3_lib import *
from v3_lib import CACHE3

core = load_core()
Z = np.load(CACHE3 / "img_v3.npz", allow_pickle=False)
zm = np.load(CACHE3 / "zone_missing.npy") if (CACHE3 / "zone_missing.npy").exists() else np.zeros(len(Z["img_id"]), bool)
pos = pd.Series(np.arange(len(Z["img_id"])), index=Z["img_id"])
row = core.id.map(pos); hi = core.has_img.to_numpy() & row.notna().to_numpy()
y, uid, fold, batch, dev = core.y.to_numpy(), core.uid.to_numpy(), core.fold.to_numpy(), core.batch.to_numpy(), (core.split == "dev").to_numpy()


def take(a):
    out = np.zeros((len(core),) + a.shape[1:], np.float32); out[hi] = a[row[hi].astype(int)].astype(np.float32); return out.reshape(len(core), -1)


def regions(e, v):
    r = Z[f"regf_{e}_{v}"].astype(np.float32).copy()
    r[zm] = Z[f"gpool_{e}_{v}"][zm].astype(np.float32)[:, None, :]     # zone_missing: 6 vung = vector toan anh
    return take(r)


FEAT = {"Cũ — BioViL-T đóng băng, toàn ảnh (CenterCrop, chưa che chữ)": np.stack(core.X_frozen)}
for e, en in (("biovil", "BioViL-T đóng băng"), ("chexfound", "CheXFound")):
    for v in ("V1", "V2"):
        FEAT[f"{v} — {en}, toàn ảnh"] = take(Z[f"glob_{e}_{v}"])
        FEAT[f"{v} — {en}, 6 vùng"] = regions(e, v)
MAIN = list(FEAT)                                                    # do nhay cho moi bo dac trung (ca CheXFound)

core_img = core[core.has_img]
grp57 = (core.bnn == 1) & (core.y == 0) & core.has_img
excl_uid = set(core.uid[grp57])                                      # 55 nguoi (theo patient_uid) — loai MOI luot cua ho
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(batch[hi].reshape(-1, 1))
Bx = np.zeros((len(core), len(oh.categories_[0])), np.float32); Bx[hi] = oh.transform(batch[hi].reshape(-1, 1))


def four(X, m, probe=True):
    yi, ui, bi = y[m], uid[m], batch[m]
    r = {}
    if probe:
        lab = pd.factorize(bi)[0]; acc = []
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X[m], lab, ui):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[m][tr], lab[tr])
            acc.append((mdl.predict(X[m][va]) == lab[va]).mean())
        r.update(probe_doan_lo=np.mean(acc), probe_doan_bua=np.bincount(lab).max() / len(lab))
    p = oof_cv(X, y, fold, m)[m]
    a, lo, hi_, _ = boot_ci(lambda i: roc_auc_score(yi[i], p[i]), ui)
    w, wlo, whi, _ = boot_ci(lambda i: within_batch_auc(yi[i], p[i], bi[i]), ui)
    pb = oof_cv(Bx, y, fold, m)[m]; pbi = oof_cv(np.column_stack([Bx, X]), y, fold, m)[m]
    d, dlo, dhi, _ = boot_ci(lambda i: roc_auc_score(yi[i], pbi[i]) - roc_auc_score(yi[i], pb[i]), ui)
    pl = np.full(m.sum(), np.nan)
    for b in np.unique(bi):                                         # LOBO that: 18 lan, moi lan de nguyen 1 lo
        tr, va = bi != b, bi == b
        pl[va] = lr().fit(X[m][tr], yi[tr]).predict_proba(X[m][va])[:, 1]
    lb, llo, lhi, _ = boot_ci(lambda i: roc_auc_score(yi[i], pl[i]), ui)
    r.update(n=int(m.sum()), duong=int(yi.sum()), L1_auc=a, L1_ci=f"[{lo:.3f}; {hi_:.3f}]", auc_trong_lo=w, auc_trong_lo_ci=f"[{wlo:.3f}; {whi:.3f}]",
             auc_chi_lo=roc_auc_score(yi, pb), delta=d, delta_ci=f"[{dlo:+.3f}; {dhi:+.3f}]", delta_lo=dlo,
             lobo_auc=lb, lobo_ci=f"[{llo:.3f}; {lhi:.3f}]")
    return r, p


m_all = dev & hi
m_sens = m_all & ~np.isin(uid, list(excl_uid))
res, sens, oof = [], [], {}
for name, X in FEAT.items():
    r, p = four(X, m_all); res.append(dict(dac_trung=name, **r)); oof[name] = p
    print(f"{name}: doan lo {r['probe_doan_lo']:.2f} (bua {r['probe_doan_bua']:.2f}) | L1 {r['L1_auc']:.3f} | trong lo {r['auc_trong_lo']:.3f} {r['auc_trong_lo_ci']} "
          f"| delta {r['delta']:+.3f} {r['delta_ci']} | LOBO {r['lobo_auc']:.3f} {r['lobo_ci']}", flush=True)
    if name in MAIN:
        s, _ = four(X, m_sens, probe=False); sens.append(dict(dac_trung=name, **s))
R = pd.DataFrame(res); S_ = pd.DataFrame(sens)
np.savez(CACHE3 / "p01_oof.npz", rows=np.where(m_all)[0], **{f"f{j}": oof[k] for j, k in enumerate(FEAT)}, names=np.array(list(FEAT)))


def decide(w, dlo):
    if w >= 0.70 and dlo > 0: return "TIẾP TỤC — AUC trong lô là chỉ số chính"
    if w >= 0.60: return "BÁO CÁO PGS trước khi làm tiếp"
    return "DỪNG nhánh ảnh"


R1 = Report("tuan1_phase0/P0_1_nhieu_lo_V1_V2", "P0.1 (bản 3) — Nhiễu theo lô chụp trên ảnh đã xử lý V1, V2")
R1.text(f"Tập: dev gốc (chưa ghép người), có ảnh: {int(m_all.sum())} lượt, {int(y[m_all].sum())} dương, {len(set(uid[m_all]))} người, "
        f"{len(set(batch[m_all]))} lô. Mô hình L1 = hồi quy logistic (chuẩn hóa, C=0,1, cân lớp) chỉ trên vector ảnh, 5 fold theo người. "
        "KTC = khoảng tin cậy 95% bootstrap theo người (1000 lần). Tập test không mở. "
        f"zone_missing (thay 6 vùng bằng vector toàn ảnh): {int(zm.sum())} ảnh.")
R1.table(R[["dac_trung", "probe_doan_lo", "probe_doan_bua"]], "1_probe_doan_lo", "Chỉ số 1 — Probe đoán lô (18 lớp) từ embedding",
         note="*Độ chính xác của hồi quy logistic đoán ảnh thuộc lô nào. `probe_doan_bua` = luôn đoán lô đông nhất. "
              "Càng cao hơn mức đoán bừa thì embedding càng mang dấu vết lô (máy chụp, thời điểm, chữ in…).*")
R1.table(R[["dac_trung", "n", "duong", "L1_auc", "L1_ci", "auc_trong_lo", "auc_trong_lo_ci", "auc_chi_lo", "delta", "delta_ci", "lobo_auc", "lobo_ci"]],
         "2_ket_qua_4_chi_so", "Chỉ số 2–4 — AUC trong lô, Δ(lô + ảnh so với lô), LOBO",
         note="*`L1_auc`: ROC-AUC gộp (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0,5 = đoán bừa). "
              "`auc_trong_lo`: AUC tính riêng trong từng lô có đủ hai nhãn, trung bình theo cỡ lô — loại phần mô hình chỉ \"nhận ra lô\". "
              "`auc_chi_lo`: AUC khi chỉ biết lô (one-hot). `delta`: AUC(lô + ảnh) − AUC(chỉ lô); KTC nằm trọn trên 0 nghĩa là ảnh thêm thông tin ngoài lô. "
              "`lobo_auc`: leave-one-batch-out — 18 lần, mỗi lần huấn luyện trên 17 lô và dự đoán lô còn lại; AUC tính trên dự đoán gộp. "
              "Mô hình không bao giờ thấy lô nó dự đoán.*")
old = pd.DataFrame([dict(nguon="Máy khác (PGS trích trong PG_update)", lobo_auc=0.742, delta="+0.078 [+0.052; +0.106]"),
                    dict(nguon="Máy này — đặc trưng cũ, cùng giao thức bản 3 (hàng đầu bảng trên)", lobo_auc=R.lobo_auc[0], delta=f"{R.delta[0]:+.3f} {R.delta_ci[0]}")])
R1.table(old, "3_so_voi_cu", "So với kết quả trên đặc trưng cũ",
         note="*Cả hai dùng dev chưa ghép người và fold gốc.*")
R1.table(S_[["dac_trung", "n", "duong", "L1_auc", "auc_trong_lo", "auc_trong_lo_ci", "delta", "delta_ci", "lobo_auc", "lobo_ci"]],
         "4_do_nhay_loai_55_nguoi", f"Phân tích độ nhạy — loại {len(excl_uid)} người có lượt bnn=Có nhưng ketqua=2 (P0.4; không sửa nhãn)")
dec = R[["dac_trung", "auc_trong_lo", "delta_lo"]].assign(quyet_dinh=[decide(w, d) for w, d in zip(R.auc_trong_lo, R.delta_lo)])
R1.table(dec.rename(columns={"delta_lo": "can_duoi_KTC_delta"}), "5_quyet_dinh",
         "Quyết định theo luật PG_update §2 (AUC trong lô của L1: ≥0,70 và KTC Δ trọn trên 0 → tiếp tục; 0,60–0,70 → báo PGS; <0,60 → dừng)")
R1.save()
R.to_csv(CACHE3 / "p01_res.csv", index=False, encoding="utf-8-sig"); S_.to_csv(CACHE3 / "p01_sens.csv", index=False, encoding="utf-8-sig")

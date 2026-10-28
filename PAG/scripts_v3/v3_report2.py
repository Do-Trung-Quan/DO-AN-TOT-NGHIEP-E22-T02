"""Bao cao viec (2) (PGS 27/10): toan bo baseline tren fold moi, anh ban 3 (C = trung binh 6 vung BioViL-T V2).
Doc cache_v3/run_baselines_v3.npz (+ run_T0/run_T2.npz neu da co). Chi dev OOF, xac suat da hieu chinh trong fold; test KHONG cham.
So voi ban 2 (results_v2/tong_hop/bang_tong_hop.csv; L1 ban 2 doi chieu voi C). Chi ghi so lieu gop.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from v3_base import *
from v3_base import load_base_v3, CACHE3, RES3, FOLDS, IMG3, run_config

B = load_base_v3()
y, hi, uid, batch, dev = B["y"], B["has_img"], B["uid"], B["batch"], B["dev"]
BL = np.load(CACHE3 / "run_baselines_v3.npz", allow_pickle=False)
NAMES = ["L0", "C", "L2", "X0", "X2", "B3b", "GAT", "T0", "T2"]
GRP = {"toàn bộ": dev, "có ảnh": dev & hi, "không ảnh": dev & ~hi}
NOTE_M = ("*Giải thích chỉ số: `roc_auc` = ROC-AUC (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0.5 = đoán bừa, 1 = hoàn hảo), "
          "kèm KTC 95% (khoảng tin cậy, bootstrap theo người 1000 lần). `pr_auc` = PR-AUC (diện tích dưới đường precision–recall; mốc đoán bừa = tỉ lệ dương). "
          "`brier` = sai số bình phương trung bình của xác suất (càng thấp càng tốt), so với `brier_nen` = Brier của việc luôn đoán tỉ lệ nền. "
          "`ece` = sai lệch hiệu chỉnh trung bình (xác suất dự đoán lệch tần suất thật bao nhiêu, 10 khoảng). "
          "`do_doc_hc` = độ dốc hiệu chỉnh (1 là lý tưởng; <1 là dự đoán quá cực đoan). "
          "`auc_trong_lo` = AUC tính riêng trong từng lô chụp rồi lấy trung bình theo cỡ lô (loại phần mô hình chỉ “nhận ra lô”). "
          "`nhay_tai_dh80` = độ nhạy (tỉ lệ ca bệnh bắt được) khi chọn ngưỡng sao cho độ đặc hiệu = 0,80 (20% người âm bị báo nhầm).*")


def pred(name):
    if f"oof_cal_{name}" in BL.files: return BL[f"oof_cal_{name}"]
    f = CACHE3 / f"run_{name}.npz"
    return np.load(f)["oof_cal"] if f.exists() else None


def sens_at(yy, pp, spec):
    thr = np.quantile(pp[yy == 0], spec); return float((pp[yy == 1] > thr).mean())


def mrow(name, g, m, p):
    if p is None or not np.isfinite(p[m]).all(): return None               # C chi co nhom co anh
    yy, pp = y[m], p[m]
    r = metrics(yy, pp, batch[m] if g == "có ảnh" else None)
    a, lo, hi_, _ = boot_ci(lambda i: roc_auc_score(yy[i], pp[i]), uid[m])
    r = dict(mo_hinh=name, nhom=g, n=r.pop("n"), duong=r.pop("duong"), roc_auc=a, roc_auc_ci=f"[{lo:.3f}–{hi_:.3f}]",
             **{k: v for k, v in r.items() if k != "roc_auc"})
    r["nhay_tai_dh80"] = sens_at(yy, pp, 0.8)
    return r


done = [n for n in NAMES if pred(n) is not None]
T = pd.DataFrame([r for n in done for g, m in GRP.items() if (r := mrow(n, g, m, pred(n)))])
cols = ["mo_hinh", "n", "duong", "roc_auc", "roc_auc_ci", "pr_auc", "brier", "brier_nen", "ece", "do_doc_hc", "nhay_tai_dh80"]

# trong lo, cap: L2 - C (lam sang them vao anh co lam AUC trong lo giam that khong)
m = dev & hi; yy, bb, uu = y[m], batch[m], uid[m]
pair = []
for a, b in (("L2", "C"), ("X2", "C"), ("B3b", "C"), ("GAT", "C"), ("T2", "C"), ("L2", "L0")):
    pa, pb = pred(a), pred(b)
    if pa is None or pb is None: continue
    A, Bb = pa[m], pb[m]
    d, lo, h, _ = boot_ci(lambda i: within_batch_auc(yy[i], A[i], bb[i]) - within_batch_auc(yy[i], Bb[i], bb[i]), uu)
    g, glo, gh, _ = boot_ci(lambda i: roc_auc_score(yy[i], A[i]) - roc_auc_score(yy[i], Bb[i]), uu)
    pair.append(dict(so_sanh=f"{a} − {b}", chenh_auc_gop=g, ci_gop=f"[{glo:+.3f}; {gh:+.3f}]",
                     chenh_auc_trong_lo=d, ci_trong_lo=f"[{lo:+.3f}; {h:+.3f}]"))

# so voi ban 2 (nhom co anh)
V2 = pd.read_csv(PAG / "results_v2/tong_hop/bang_tong_hop.csv", encoding="utf-8-sig").set_index("mo_hinh")
cmp = []
for n in done:
    n2 = "L1" if n == "C" else n
    r3 = T[(T.mo_hinh == n) & (T.nhom == "có ảnh")]
    if n2 not in V2.index or r3.empty: continue
    cmp.append(dict(mo_hinh=n, ban2=n2, auc_co_anh_ban2=V2.loc[n2, "có ảnh | roc_auc"], auc_co_anh_ban3=r3.roc_auc.iloc[0],
                    trong_lo_ban2=V2.loc[n2, "có ảnh | auc_trong_lo"], trong_lo_ban3=r3.auc_trong_lo.iloc[0]))
CMP = pd.DataFrame(cmp)

R = Report("tuan1_phase0/viec2_baseline_fold_moi", "Việc (2) — Chạy lại toàn bộ baseline trên fold mới")
R.text("Lời nhắn PGS 27/10: chạy lại L0, C (thay L1), L2, X0/X2, T0/T2, GAT/B3b với k chọn trên dev, trên fold mới (chuẩn duy nhất: "
       "`cache_v3/p05_folds.npz` — dev2/fold2/uid2). Dev = 6683 lượt khám (1500 có ảnh), 5 fold theo người; test khoá, không chấm.\n\n"
       "Khác bản 2: đặc trưng ảnh là **C** = trung bình 6 vector vùng BioViL-T đóng băng trên ảnh V2 (che chữ mới + xoá ngoài phổi; chốt ở việc 1). "
       "C dùng làm ảnh trong L2/X2/B3b/GAT; T2 dùng 6 token vùng qua một phép chiếu PCA 128 chung cho cả 6 vùng.\n\n"
       "Giao thức: LR (hồi quy logistic, cân bằng lớp) tinh chỉnh C (hệ số điều chuẩn, C nhỏ = phạt mạnh) bằng CV nội bộ 5 fold theo người trong phần train, "
       "tiêu chí ROC-AUC gộp, lưới 1e-6…10; XGBoost tham số cố định như bản 2, trung bình 3 seed; B3b/GAT chọn k ∈ {5, 10, 20} (số láng giềng đồ thị kNN) "
       "trên validation nội bộ (fold (k+1)%5 trong 4 fold train); T0/T2 dò 30 cấu hình trên cùng validation nội bộ, 3 seed. "
       "Hiệu chỉnh xác suất Platt trong fold, riêng cho lượt có ảnh / không ảnh.\n")
if not {"T0", "T2"} <= set(done):
    R.text(f"**Chưa có:** {', '.join(sorted({'T0', 'T2'} - set(done)))} (đang chạy — báo cáo sẽ được cập nhật).\n")
for g in GRP:
    c = cols + (["auc_trong_lo"] if g == "có ảnh" else [])
    R.table(T[T.nhom == g][c], f"chi_so_{'toan_bo' if g == 'toàn bộ' else 'co_anh' if g == 'có ảnh' else 'khong_anh'}",
            f"Nhóm {g} — dev OOF, xác suất đã hiệu chỉnh", note=NOTE_M if g == "có ảnh" else None)
if pair:
    R.table(pd.DataFrame(pair), "so_sanh_cap_co_anh", "So sánh cặp trên nhóm có ảnh (chênh ROC-AUC gộp và AUC trong lô, KTC 95% bootstrap theo người)",
            note="*Âm = mô hình đứng trước kém hơn. AUC trong lô loại phần mô hình chỉ “nhận ra lô chụp”.*")
if len(CMP):
    R.table(CMP, "so_voi_ban_2", "So với bản 2 — nhóm có ảnh (bản 2: ảnh cũ, cùng fold; L1 bản 2 ↔ C bản 3)")
ch = pd.read_csv(CACHE3 / "chon_baselines_v3.csv")
R.table(ch.pivot_table(index="fold", columns="mo_hinh", values="gia_tri").reset_index(), "tham_so_chon",
        "Tham số chọn theo fold (C của LR; k của B3b/GAT)", nd=4)
R.save()
run_config("tuan1_phase0/viec2_baseline_fold_moi/bao_cao", __file__, data=[FOLDS, IMG3, CACHE3 / "run_baselines_v3.npz"], mo_hinh_co=done)
show = T[T.nhom == "có ảnh"][["mo_hinh", "roc_auc", "roc_auc_ci", "pr_auc", "brier", "ece", "auc_trong_lo"]]
print(show.round(3).to_string(index=False)); print(pd.DataFrame(pair).round(3).to_string(index=False)); print(CMP.round(3).to_string(index=False))
print(T[T.nhom != "có ảnh"][["mo_hinh", "nhom", "roc_auc", "roc_auc_ci", "pr_auc", "ece"]].round(3).to_string(index=False))

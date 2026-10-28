"""P0.1 buoc 5 (PAG_V2 §2): lam lai kiem tra nhieu lo chup tren dac trung SAU che chu + cat theo phoi,
va bao cao tien xu ly anh (P0.2). Chi ghi so lieu gop + hinh mask (khong anh goc, khong ten tep)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from v2_lib import *

core = load_core()
Z = np.load(CACHE / "img_v2.npz", allow_pickle=False)
pos = pd.Series(np.arange(len(Z["img_id"])), index=Z["img_id"])
assert set(core.id[core.has_img]) <= set(pos.index), "anh thieu"
row = core.id.map(pos)                                     # chi so anh cho moi luot (nan neu khong anh)
hi = core.has_img.to_numpy() & row.notna().to_numpy()
y, uid, fold, batch, dev = core.y.to_numpy(), core.uid2.to_numpy(), core.fold2.to_numpy(), core.batch.to_numpy(), core.dev2.to_numpy()

def take(a):
    out = np.zeros((len(core),) + a.shape[1:], np.float32); out[hi] = a[row[hi].astype(int)].astype(np.float32); return out.reshape(len(core), -1)

FEAT = {}
for enc, nm in (("biovil_frozen", "BioViL-T đóng băng"), ("biovil_control", "BioViL-T control"), ("xrv", "ResNet50-xrv"), ("chexfound", "CheXFound")):
    FEAT[f"{nm} — 6 vùng"] = take(Z[f"regf_{enc}"])
    FEAT[f"{nm} — toàn ảnh"] = take(Z[f"glob_{enc}"])
OLD = pd.read_csv(OLDRES /"tuan1_phase0/P0_1_nhieu_lo_chup/buoc2_4_ket_qua.csv", encoding="utf-8-sig")

# =========================================================================== P0.2 bao cao tien xu ly
R2 = Report("tuan1_phase0/P0_2_tien_xu_ly_anh", "P0.2 — Tiền xử lý ảnh: che chữ in trên phim → phân đoạn phổi → cắt theo phổi → 6 vùng")
R2.text("Thứ tự theo PAG_V2: (1) đưa ảnh về vuông 1024; (2) phân đoạn phổi sơ bộ (PSPNet torchxrayvision) chỉ để biết chỗ nào là phổi; "
        "(3) che chữ in trên phim (burn-in: tên, ngày, ký hiệu L/R) bằng lọc top-hat + ngưỡng sáng + lọc hình dạng, chỉ ở NGOÀI phổi, rồi tô lấp (inpaint Telea); "
        "(4) phân đoạn lại trên ảnh đã che; (4b) tô đen mọi thứ ngoài bao lồi hai phổi nới 2% (QA 50 ảnh thấy 18/50 còn sót chữ); (5) cắt theo khung bao hai phổi + lề 6% (bỏ CenterCrop); (6) chia mỗi phổi thành trên/giữa/dưới ở 1/3 và 2/3 chiều cao → 6 vùng ILO (RU RM RL LU LM LL).")
q = {k: Z[k] for k in ["n_chu", "pct_chu", "pct_crop", "empty_R", "empty_L", "chu_trong_phoi"]}
qa = pd.DataFrame([
    dict(muc="Số ảnh", gia_tri=len(q["n_chu"])),
    dict(muc="Ảnh có ít nhất một vùng bị che", gia_tri=f"{int((q['n_chu'] > 0).sum())} ({100*(q['n_chu'] > 0).mean():.1f}%)"),
    dict(muc="Số vùng chữ / ảnh — trung vị [p5–p95]", gia_tri=f"{np.median(q['n_chu']):.0f} [{np.percentile(q['n_chu'],5):.0f}–{np.percentile(q['n_chu'],95):.0f}]"),
    dict(muc="Diện tích bị che (% ảnh) — trung vị [p95]", gia_tri=f"{np.median(q['pct_chu']):.2f}% [{np.percentile(q['pct_chu'],95):.2f}%]"),
    dict(muc="Ảnh có vùng che chạm phổi (phải = 0)", gia_tri=int(q["chu_trong_phoi"].sum())),
    dict(muc="Phổi phải / trái phân đoạn rỗng (<1% ảnh)", gia_tri=f"{int(q['empty_R'].sum())} / {int(q['empty_L'].sum())}"),
    dict(muc="Khung cắt chiếm (% ảnh) — trung vị [p5–p95]", gia_tri=f"{np.median(q['pct_crop']):.0f}% [{np.percentile(q['pct_crop'],5):.0f}–{np.percentile(q['pct_crop'],95):.0f}%]"),
    dict(muc="Vùng có diện tích 0 (không có đặc trưng)", gia_tri=int((Z["area"] == 0).sum())),
])
R2.table(qa, "kiem_tra_chat_luong", "Kiểm tra chất lượng (1835 ảnh)")
ar = pd.DataFrame(Z["area"] * 100, columns=REG).describe(percentiles=[.05, .5, .95]).T[["mean", "5%", "50%", "95%"]]
R2.table(ar.round(2), "dien_tich_vung", "Diện tích mỗi vùng (% khung cắt 512×512)", index=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
ax[0].imshow(Z["txt"].mean(0), cmap="magma"); ax[0].set_title("Tần suất bị che chữ (toàn khung, 1835 ảnh)"); ax[0].axis("off")
ax[1].imshow(np.stack([(Z["masks"] == r + 1).mean(0) for r in range(6)]).argmax(0) * ((Z["masks"] > 0).mean(0) > 0.5), cmap="tab10", vmin=0, vmax=9)
ax[1].set_title("Vùng ILO chiếm đa số (trong khung cắt)"); ax[1].axis("off")
R2.fig(fig, "ban_do_che_va_vung", "Bản đồ tổng hợp: nơi bị che chữ và 6 vùng (chỉ mask trung bình, không có ảnh bệnh nhân)")
rng = np.random.default_rng(0); sel = rng.choice(len(Z["masks"]), 24, replace=False)
fig, ax = plt.subplots(3, 8, figsize=(16, 6))
for a, i in zip(ax.flat, sel): a.imshow(Z["masks"][i], cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); a.axis("off")
R2.fig(fig, "mau_mask_vung", "24 mask vùng ngẫu nhiên (chỉ mask nhãn vùng, 128×128, không ảnh gốc)")
R2.save(); print(qa.to_string(index=False))

# =========================================================================== P0.1 buoc 5
R1 = Report("tuan1_phase0/P0_1_nhieu_lo_chup_buoc5", "P0.1 bước 5 — Nhiễu lô chụp SAU che chữ + cắt theo phổi")
m = dev & hi; yi, ui, bi = y[m], uid[m], batch[m]
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(batch[hi].reshape(-1, 1))
Bx = np.zeros((len(core), len(oh.categories_[0])), np.float32); Bx[hi] = oh.transform(batch[hi].reshape(-1, 1))
res, probe = [], []
for name, X in FEAT.items():
    lab = pd.factorize(bi)[0]; acc = []
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X[m], lab, ui):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[m][tr], lab[tr])
        acc.append((mdl.predict(X[m][va]) == lab[va]).mean())
    probe.append(dict(dac_trung=name, do_chinh_xac_doan_lo=np.mean(acc), muc_doan_bua=np.bincount(lab).max() / len(lab)))
    p = oof_cv(X, y, fold, m)[m]
    a, lo, hi_, _ = boot_ci(lambda i: roc_auc_score(yi[i], p[i]), ui)
    w, wlo, whi, _ = boot_ci(lambda i: within_batch_auc(yi[i], p[i], bi[i]), ui)
    pb = oof_cv(Bx, y, fold, m)[m]; pbi = oof_cv(np.column_stack([Bx, X]), y, fold, m)[m]
    dlt, dlo, dhi, _ = boot_ci(lambda i: roc_auc_score(yi[i], pbi[i]) - roc_auc_score(yi[i], pb[i]), ui)
    plb = np.full(m.sum(), np.nan)
    for tr, va in GroupKFold(5).split(X[m], yi, bi):
        plb[va] = lr().fit(X[m][tr], yi[tr]).predict_proba(X[m][va])[:, 1]
    res.append(dict(dac_trung=name, L1_auc=a, L1_auc_ci=f"[{lo:.3f}–{hi_:.3f}]", auc_trong_lo=w, auc_trong_lo_ci=f"[{wlo:.3f}–{whi:.3f}]",
                    auc_chi_lo=roc_auc_score(yi, pb), delta_anh=dlt, delta_ci=f"[{dlo:.3f}–{dhi:.3f}]", delta_ci_duoi=dlo,
                    auc_de_nguyen_lo=roc_auc_score(yi, plb)))
    print(f"{name}: L1 {a:.3f} | trong lo {w:.3f} [{wlo:.3f}-{whi:.3f}] | delta {dlt:+.3f} [{dlo:.3f},{dhi:.3f}] | de nguyen lo {res[-1]['auc_de_nguyen_lo']:.3f} | doan lo {np.mean(acc):.2f}", flush=True)
R = pd.DataFrame(res)
R1.table(pd.DataFrame(probe), "doan_lo_tu_anh", "Đoán lô (18 lớp) từ đặc trưng ảnh đã che chữ + cắt",
         note="*Độ chính xác đoán lô càng gần `muc_doan_bua` (luôn đoán lô đông nhất) thì đặc trưng càng ít mang dấu vết của lô.*")
R1.table(R.drop(columns="delta_ci_duoi"), "ket_qua", "AUC gộp, AUC trong lô, Δ do ảnh so với chỉ biết lô, và fold để nguyên lô",
         note="*`L1_auc`: ROC-AUC (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0.5 = đoán bừa) của hồi quy logistic chỉ trên ảnh. "
              "`auc_trong_lo`: AUC tính riêng trong từng lô chụp rồi lấy trung bình theo cỡ lô; nó loại phần mô hình chỉ “nhận ra lô”. "
              "`delta_anh`: AUC(lô + ảnh) − AUC(chỉ lô); dương nghĩa là ảnh thêm thông tin ngoài lô. "
              "`auc_de_nguyen_lo`: mỗi lô nằm trọn trong một fold; mô hình phải dự đoán cho lô nó chưa thấy. CI = khoảng tin cậy 95% bootstrap theo người.*")
cmp = OLD[["dac_trung", "auc_trong_lo", "delta_anh"]].rename(columns={"dac_trung": "dac_trung_truoc_che (toàn ảnh, CenterCrop)", "auc_trong_lo": "auc_trong_lo_truoc", "delta_anh": "delta_truoc"})
R1.table(cmp, "truoc_che_de_so_sanh", "Để so sánh: kết quả bước 1–4 (đặc trưng cũ, chưa che chữ)")
def decide(w, dlo):
    if w >= 0.70 and dlo > 0: return "ĐI TIẾP — AUC trong lô thành chỉ số chính"
    if w >= 0.60: return "ĐI TIẾP CÓ ĐIỀU KIỆN — che chữ + cắt bắt buộc"
    return "DỪNG nhánh ảnh"
R1.table(R[["dac_trung", "auc_trong_lo", "delta_ci_duoi"]].assign(quyet_dinh=[decide(w, d) for w, d in zip(R.auc_trong_lo, R.delta_ci_duoi)]),
         "quyet_dinh", "Quyết định cuối P0.1 theo luật §2 (≥0.70 và Δ>0 → đi tiếp; 0.60–0.70 → có điều kiện; <0.60 → dừng)")
R1.save()

"""Phase 0 ban 2 (PAG_V2 §2): P0.6 khoi kham 6x7, P0.4 nhan ketqua vs bnn, P0.5 kiem toan, P0.1 nhieu lo chup (buoc 1-4).
Buoc 5 cua P0.1 (lam lai sau che chu + cat phoi) nam trong v2_p01_after.py.
Chi dung tap dev — tap test KHONG mo. Chi ghi so lieu gop.
"""
import re, unicodedata
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from v2_lib import *

core = load_core()
dev = core.dev2.to_numpy(); hi = core.has_img.to_numpy()          # dev sau khi gop nguoi (P0.5), fold chia lai theo uid2
y, uid, fold, batch = core.y.to_numpy(), core.uid2.to_numpy(), core.fold2.to_numpy(), core.batch.to_numpy()
raw = pd.read_pickle(MERG / "cache_pg" / "clin_raw.pkl")          # cot lam sang tho, khong PII, cung thu tu
FEAT = {"BioViL-T frozen (toàn ảnh)": np.stack(core.X_frozen), "BioViL-T control (toàn ảnh)": np.stack(core.X_control)}
fm = np.load(MERG / "cache_pg" / "region_features_fm.npz")
pos = pd.Index(fm["img_id"]).get_indexer(core.id); G = np.zeros((len(core), fm["glob"].shape[1]), np.float32)
G[pos >= 0] = fm["glob"][pos[pos >= 0]]; FEAT["ResNet50-xrv (toàn ảnh, bản 1)"] = G

# =========================================================================== P0.6
R6 = Report("tuan1_phase0/P0_6_kham_6x7", "P0.6 — Khảo sát khối khám thực thể 6 vùng × 7 dấu hiệu (dev)")
rows = []
for p, nm in EXAM_SIGNS.items():
    for i, r in enumerate(REG, 1):
        s = raw[f"{p}{i}"].astype(str).where(raw[f"{p}{i}"].notna())
        for lab, m in (("ketqua=1", dev & (y == 1)), ("ketqua=2", dev & (y == 0))):
            rows.append(dict(dau_hieu=nm, vung=r, nhom=lab, n=int(m.sum()), pct_co_ghi=100 * s[m].notna().mean(),
                             pct_duong_tren_tong=100 * (s[m] == "Co").mean()))
T6 = pd.DataFrame(rows)
piv = T6.pivot_table(index=["dau_hieu", "vung"], columns="nhom", values=["pct_co_ghi", "pct_duong_tren_tong"])
piv.columns = [f"{a}|{b}" for a, b in piv.columns]
R6.table(piv.reset_index(), "bang_6x7_theo_ketqua", "Tỉ lệ có ghi và tỉ lệ dương tính của 42 ô khám theo vùng, tách theo ketqua (%)", nd=2)
GLOBAL = {"rungthan": "Rung thanh (chung)", "go": "Gõ (chung)", "riraopn": "Rì rào phế nang", "ran": "Ran (chung)",
          "longnguc": "Lồng ngực", "khoangls": "Khoảng liên sườn"}
g_rows = []
for c, nm in GLOBAL.items():
    s = raw[c].astype(str).where(raw[c].notna())
    for lab, m in (("ketqua=1", dev & (y == 1)), ("ketqua=2", dev & (y == 0))):
        vc = s[m].value_counts(normalize=True)
        normal = vc.index[0] if len(vc) else ""
        g_rows.append(dict(dau_hieu=nm, nhom=lab, pct_co_ghi=100 * s[m].notna().mean(),
                           pct_bat_thuong=100 * (s[m].notna() & (s[m] != normal)).mean(), muc_binh_thuong=normal))
R6.table(pd.DataFrame(g_rows), "dau_hieu_khong_theo_vung", "Dấu hiệu khám KHÔNG theo vùng (tỉ lệ bất thường = khác mức phổ biến nhất)", nd=2)
any_reg = np.zeros(len(core), bool); n_rec = np.zeros(len(core), bool)
for p in EXAM_SIGNS:
    for i in range(1, 7):
        s = raw[f"{p}{i}"].astype(str); any_reg |= (s == "Co").to_numpy(); n_rec |= raw[f"{p}{i}"].notna().to_numpy()
max_pos = T6.groupby(["dau_hieu", "vung"]).apply(lambda g: (g.pct_duong_tren_tong * g.n).sum() / g.n.sum()).max()
fallback = max_pos < 1.0
dec6 = pd.DataFrame([
    dict(tieu_chi="Số lượt khám dev có ghi ít nhất 1 ô vùng", gia_tri=f"{int((n_rec & dev).sum())}/{int(dev.sum())} ({100*(n_rec&dev).mean()/dev.mean():.2f}%)"),
    dict(tieu_chi="Số lượt khám dev có ít nhất 1 ô vùng dương tính", gia_tri=f"{int((any_reg & dev).sum())}"),
    dict(tieu_chi="Tỉ lệ dương tính cao nhất trong 42 ô (dev)", gia_tri=f"{max_pos:.3f}%"),
    dict(tieu_chi="Luật §4: mọi ô < 1% → gộp 6 nút khám thành 1, bỏ E6", gia_tri="KÍCH HOẠT phương án dự phòng" if fallback else "không kích hoạt"),
    dict(tieu_chi="Khám có độc lập với đọc phim?", gia_tri="Không kiểm chứng được từ dữ liệu (không có thời điểm khám / người khám) — cần hỏi PGS"),
])
R6.table(dec6, "quyet_dinh_P0_6", "Quyết định P0.6")
R6.save()
print(f"P0.6: co ghi {int((n_rec&dev).sum())} luot, duong max {max_pos:.3f}% -> fallback={fallback}")

# =========================================================================== P0.4
R4 = Report("tuan1_phase0/P0_4_nhan_ketqua_bnn", "P0.4 — Ý nghĩa nhãn ketqua so với bnn (dev)")
d = core[dev].copy(); d["uid"] = d.uid2; d["nhom"] = np.select([(d.bnn == 1) & (d.y == 1), (d.bnn == 1) & (d.y == 0), (d.bnn == 0) & (d.y == 1)],
                                            ["bnn=Có, ketqua=1", "bnn=Có, ketqua=2", "bnn=Không, ketqua=1"], "bnn=Không, ketqua=2")
ct = d.groupby("nhom").agg(luot=("y", "size"), nguoi=("uid", "nunique"), co_anh=("has_img", "sum"),
                           nguoi_co_anh=("uid", lambda s: d.loc[s.index][d.loc[s.index].has_img].uid.nunique()))
R4.table(ct.reset_index(), "bang_cheo_bnn_ketqua", "Bảng chéo bnn × ketqua (lượt khám / người)")
# L1 (BioViL-T frozen) OOF tren nhom co anh
X = FEAT["BioViL-T frozen (toàn ảnh)"]; rows_img = dev & hi
pL1 = oof_cv(X, y, fold, rows_img)
d["L1"] = pL1[dev]
sc = d[d.has_img].groupby("nhom").L1.describe()[["count", "25%", "50%", "75%"]]
R4.table(sc.reset_index(), "diem_L1_theo_nhom", "Điểm L1 (LR trên BioViL-T frozen, OOF) theo nhóm — chỉ ca có ảnh")
ilo = d[d.nhom == "bnn=Có, ketqua=2"].matdo.replace("nan", "(trống)").value_counts().rename("so_luot").reset_index()
R4.table(ilo, "matdo_ILO_bnn_co_ketqua2", "Trường ILO mật độ tổn thương ở nhóm bnn=Có, ketqua=2")
pos_uid = set(d.loc[d.y == 1, "uid"])
grp = d[d.nhom == "bnn=Có, ketqua=2"]
other = pd.DataFrame([dict(chi_so="Người bnn=Có, ketqua=2", gia_tri=grp.uid.nunique()),
                      dict(chi_so="… trong đó có lượt khám KHÁC với ketqua=1", gia_tri=len(set(grp.uid) & pos_uid)),
                      dict(chi_so="… có > 1 lượt khám", gia_tri=int((d[d.uid.isin(grp.uid)].groupby("uid").size() > 1).sum())),
                      dict(chi_so="Người bnn=Có, ketqua=2 có ảnh", gia_tri=grp[grp.has_img].uid.nunique())])
R4.table(other, "luot_kham_khac", "Các lượt khám khác của cùng người")
R4.text("**Diễn giải.** Nếu nhóm bnn=Có, ketqua=2 có điểm L1 gần nhóm ketqua=2 thì ảnh của họ trông bình thường. Khi đó ketqua có thể là kết luận phim *tại lần khám này*, không phải chẩn đoán bệnh nghề nghiệp, và hai nhãn đo hai thứ khác nhau. Cần PGS xác nhận.")
R4.save()
print("P0.4:", ct.to_dict("index"))

# =========================================================================== P0.5
R5 = Report("tuan1_phase0/P0_5_kiem_toan", "P0.5 — Kiểm toán dữ liệu (ảnh trùng, người bị tách mã)")
idx = pd.read_parquet(CXR / "image_index.parquet")
xl = pd.read_excel(XLSX, usecols=["id", "hoten", "namsinh", "gioitinh", "tinh"])       # PII — chi dem, khong in
def norm(s):
    s = unicodedata.normalize("NFD", str(s).lower()); s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z ]", "", s.replace("đ", "d")).strip()
xl = xl.set_index("id").loc[core.id].reset_index()
key = xl.hoten.map(norm) + "|" + xl.namsinh.astype(str) + "|" + xl.gioitinh.astype(str) + "|" + xl.tinh.astype(str)
kk = pd.DataFrame(dict(key=key, uid=core.uid, split=core.split, y=y))
multi = kk.groupby("key").uid.nunique(); multi = multi[multi > 1]
km = kk[kk.key.isin(multi.index)]
cross = km.groupby("key").split.nunique().gt(1).sum()
audit = pd.DataFrame([
    dict(muc="Ảnh trùng SHA giữa các lượt khám", gia_tri=int(idx.file_hash.duplicated().sum())),
    dict(muc="Khóa lỏng (tên chuẩn hóa + năm sinh + giới + tỉnh) ứng với >1 patient_uid", gia_tri=len(multi)),
    dict(muc="… số patient_uid liên quan", gia_tri=int(km.uid.nunique())),
    dict(muc="… khóa trải qua cả dev và test (nguy cơ rò rỉ)", gia_tri=int(cross)),
    dict(muc="… khóa có ít nhất một lượt ketqua=1", gia_tri=int(km.groupby("key").y.max().sum())),
    dict(muc="Sau gộp: số người", gia_tri=f"{core.uid.nunique()} → {core.uid2.nunique()}"),
    dict(muc="Lượt dev bị loại vì cùng người với test", gia_tri=f"{int(((core.split=='dev')&~core.dev2).sum())} (dương {int((((core.split=='dev')&~core.dev2)&(core.y==1)).sum())}, có ảnh {int((((core.split=='dev')&~core.dev2)&core.has_img).sum())})"),
    dict(muc="Dev mới", gia_tri=f"{int(core.dev2.sum())} lượt; có ảnh {int((core.dev2&core.has_img).sum())} (dương {int((core.dev2&core.has_img&(core.y==1)).sum())}); không ảnh {int((core.dev2&~core.has_img).sum())} (dương {int((core.dev2&~core.has_img&(core.y==1)).sum())})"),
    dict(muc="Fold dev cũ: khóa người nằm ở >1 fold", gia_tri="484 → 0 sau khi chia lại 5 fold theo người đã gộp"),
    dict(muc="Chọn lại k của đồ thị kNN chỉ trên dev", gia_tri="làm trong tuần 2 (mô hình so sánh GAT/B3b dựng lại, k chọn bằng validation nội bộ)"),
])
R5.table(audit, "kiem_toan", "Kết quả kiểm toán", note="**Quyết định:** từ đây mọi phân tích dùng mã người đã gộp, dev mới và 5 fold mới. Tập test giữ nguyên và vẫn khóa. Kết quả bản 1 bị ảnh hưởng bởi lỗi này.\n\n*Ghép khóa lỏng chạy cục bộ trên dữ liệu có PII. Chỉ số đếm được ghi ra; không có tên hay mã nào rời máy.*")
R5.save()
print("P0.5:", audit.gia_tri.tolist()[:5])

# =========================================================================== P0.1 buoc 1-4
R1 = Report("tuan1_phase0/P0_1_nhieu_lo_chup", "P0.1 — Kiểm tra nhiễu theo lô chụp (bước 1–4, đặc trưng cũ chưa che chữ)")
m = dev & hi; yi, ui, bi, fi = y[m], uid[m], batch[m], fold[m]
lots = pd.crosstab(bi, yi); lots.columns = ["ketqua=2", "ketqua=1"]
R1.table(lots.reset_index().rename(columns={"row_0": "lo"}).assign(lo=lambda t: [f"lô {i+1:02d}" for i in range(len(t))]),
         "phan_bo_nhan_theo_lo", "Phân bố nhãn theo 18 lô (dev, có ảnh; mã lô ẩn danh)")
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(batch[hi].reshape(-1, 1))
Bx = np.zeros((len(core), len(oh.categories_[0])), np.float32); Bx[hi] = oh.transform(batch[hi].reshape(-1, 1))
res, probe = [], []
for name, X in FEAT.items():
    # buoc 1: doan lo tu embedding (5 fold theo nguoi)
    lab = pd.factorize(bi)[0]; acc = []
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X[m], lab, ui):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[m][tr], lab[tr])
        acc.append((mdl.predict(X[m][va]) == lab[va]).mean())
    probe.append(dict(dac_trung=name, so_lo=len(set(lab)), do_chinh_xac=np.mean(acc), muc_doan_bua=np.bincount(lab).max() / len(lab)))
    # buoc 2: L1 OOF -> AUC gop + AUC trong lo
    p = oof_cv(X, y, fold, m)[m]
    a, lo, hi_, _ = boot_ci(lambda i: roc_auc_score(yi[i], p[i]), ui)
    w, wlo, whi, _ = boot_ci(lambda i: within_batch_auc(yi[i], p[i], bi[i]), ui)
    # buoc 3: LR(lo) vs LR(lo + anh)
    pb = oof_cv(Bx, y, fold, m)[m]; pbi = oof_cv(np.column_stack([Bx, X]), y, fold, m)[m]
    dlt, dlo, dhi, _ = boot_ci(lambda i: roc_auc_score(yi[i], pbi[i]) - roc_auc_score(yi[i], pb[i]), ui)
    # buoc 4: fold de nguyen lo (GroupKFold theo lo)
    plb = np.full(m.sum(), np.nan)
    for tr, va in GroupKFold(5).split(X[m], yi, bi):
        plb[va] = lr().fit(X[m][tr], yi[tr]).predict_proba(X[m][va])[:, 1]
    res.append(dict(dac_trung=name, L1_auc=a, L1_auc_ci=f"[{lo:.3f}–{hi_:.3f}]", auc_trong_lo=w, auc_trong_lo_ci=f"[{wlo:.3f}–{whi:.3f}]",
                    auc_chi_lo=roc_auc_score(yi, pb), auc_lo_cong_anh=roc_auc_score(yi, pbi), delta_anh=dlt, delta_ci=f"[{dlo:.3f}–{dhi:.3f}]",
                    auc_de_nguyen_lo=roc_auc_score(yi, plb), auc_trong_lo_de_nguyen_lo=within_batch_auc(yi, plb, bi)))
    print(f"P0.1 {name}: L1 {a:.3f} | trong lo {w:.3f} [{wlo:.3f}-{whi:.3f}] | delta {dlt:+.3f} [{dlo:.3f},{dhi:.3f}] | de nguyen lo {res[-1]['auc_de_nguyen_lo']:.3f}", flush=True)
R1.table(pd.DataFrame(probe), "buoc1_doan_lo_tu_anh", "Bước 1 — Đoán lô (18 lớp) từ embedding ảnh",
         note="*`muc_doan_bua` là độ chính xác khi luôn đoán lô đông nhất. Độ chính xác cao hơn nhiều mức này nghĩa là embedding mang dấu vết của lô.*")
R = pd.DataFrame(res)
R1.table(R, "buoc2_4_ket_qua", "Bước 2–4 — AUC gộp, AUC trong lô, Δ do ảnh so với chỉ lô, và fold để nguyên lô")
def decide(w, dlo):
    if w >= 0.70 and dlo > 0: return "ĐI TIẾP — AUC trong lô thành chỉ số chính"
    if w >= 0.60: return "ĐI TIẾP CÓ ĐIỀU KIỆN — bắt buộc che chữ + cắt theo phổi"
    return "DỪNG nhánh ảnh"
dec = R[["dac_trung", "auc_trong_lo"]].assign(delta_ci_duoi=[float(s[1:6]) for s in R.delta_ci],
                                               quyet_dinh=[decide(w, float(s[1:6])) for w, s in zip(R.auc_trong_lo, R.delta_ci)])
R1.table(dec, "quyet_dinh_buoc_1_4", "Quyết định tạm (trước che chữ) theo luật §2")
R1.save()

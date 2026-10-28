"""P0.4, P0.5 (PG_update §2) + so kiem toan CSV. Chay sau v3_p01.py (dung diem OOF L1).
P0.4: boxplot diem L1 OOF cua 57 ca bnn=Co & ketqua=2 so voi hai nhom con lai; ketqua cua ho o luot kham khac. Khong sua nhan.
P0.5: dem luot kham thuoc nhom ghep noi long vat qua hai fold hoac qua dev/test; <1% giu fold, >=1% chia lai fold theo khoa ghep.
Test khong mo: luot test chi duoc dem, khong doc nhan / diem. Chi ghi so lieu gop; so kiem toan dung ma an danh.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from v3_lib import *
from v3_lib import CACHE3, QA_DIR, RES3, anon

core = load_core()
y, dev, hi = core.y.to_numpy(), core.dev2.to_numpy(), core.has_img.to_numpy()
O = np.load(CACHE3 / "p01_oof.npz", allow_pickle=False); names = list(O["names"])
MAINF = "V1 — BioViL-T đóng băng, toàn ảnh"
score = np.full(len(core), np.nan); score[O["rows"]] = O[f"f{names.index(MAINF)}"]
score_old = np.full(len(core), np.nan); score_old[O["rows"]] = O["f0"]
g57 = ((core.bnn == 1) & (y == 0) & core.has_img).to_numpy()
audit = []

# =========================================================================== P0.4
R4 = Report("tuan1_phase0/P0_4_57_ca_bnn_ketqua", "P0.4 — 57 lượt bnn=Có nhưng ketqua=2 (55 người)")
R4.text(f"Toàn bộ: {int(g57.sum())} lượt / {core.uid[g57].nunique()} người (theo mã gốc), trong đó dev {int((g57 & (core.split=='dev')).sum())}, "
        f"test {int((g57 & (core.split=='test')).sum())} (test chỉ đếm, không chấm điểm). Sau ghép người (P0.5) còn {int((g57 & dev).sum())} lượt trong dev2. "
        f"Điểm = xác suất out-of-fold (dự đoán bởi mô hình chưa thấy người đó) của L1 trên `{MAINF}`; so thêm với đặc trưng cũ.")
nhom = np.select([g57, (y == 1), (core.bnn == 0) & (y == 0)], ["bnn=Có, ketqua=2 (nhóm P0.4)", "ketqua=1", "bnn=Không, ketqua=2"], "khác")
m = dev & hi & ~np.isnan(score)
rows = []
for gname in ["bnn=Có, ketqua=2 (nhóm P0.4)", "ketqua=1", "bnn=Không, ketqua=2"]:
    k = m & (nhom == gname)
    rows.append(dict(nhom=gname, luot=int(k.sum()), trung_vi=np.median(score[k]), p25=np.percentile(score[k], 25), p75=np.percentile(score[k], 75),
                     trung_vi_dac_trung_cu=np.median(score_old[k])))
T4 = pd.DataFrame(rows)
a, b, c = (m & (nhom == T4.nhom[0])), (m & (nhom == "ketqua=1")), (m & (nhom == "bnn=Không, ketqua=2"))
auc_vs_neg = roc_auc_score(np.r_[np.ones(a.sum()), np.zeros(c.sum())], np.r_[score[a], score[c]])
auc_vs_pos = roc_auc_score(np.r_[np.ones(b.sum()), np.zeros(a.sum())], np.r_[score[b], score[a]])
R4.table(T4, "diem_L1_theo_nhom", "Điểm L1 out-of-fold theo nhóm (dev2, có ảnh)",
         note=f"*AUC nhóm P0.4 so với bnn=Không, ketqua=2 = {auc_vs_neg:.3f} (0,5 = không phân biệt; >0,5 = nhóm P0.4 có điểm cao hơn âm thường) — "
              f"p Mann–Whitney {mannwhitneyu(score[a], score[c]).pvalue:.3g}. "
              f"AUC ketqua=1 so với nhóm P0.4 = {auc_vs_pos:.3f} (>0,5 = dương thật có điểm cao hơn nhóm P0.4).*")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([score[m & (nhom == g)] for g in T4.nhom], labels=[f"{g}\n(n={n})" for g, n in zip(T4.nhom, T4.luot)], showfliers=True)
ax.set_ylabel("Điểm L1 out-of-fold"); ax.set_title("P0.4 — điểm ảnh L1 theo nhóm (dev2)")
R4.fig(fig, "boxplot_diem_L1", "Boxplot điểm L1 out-of-fold")
# ketqua o luot kham khac (chi dev; luot test chi dem)
P = core.assign(g57=g57)
ppl = set(P.uid2[g57])
oth = P[P.uid2.isin(ppl) & ~P.g57]
od, ot = oth[oth.split == "dev"], oth[oth.split == "test"]
per = od.groupby("uid2").y.agg(["size", "max"])
R4.table(pd.DataFrame([
    dict(muc="Người trong nhóm (theo uid2)", gia_tri=len(ppl)),
    dict(muc="Người có lượt khám khác trong dev", gia_tri=int(od.uid2.nunique())),
    dict(muc="Số lượt khám khác trong dev", gia_tri=len(od)),
    dict(muc="  trong đó ketqua=1", gia_tri=f"{int(od.y.sum())} ({100*od.y.mean():.0f}%)" if len(od) else "0"),
    dict(muc="  trong đó có ảnh", gia_tri=int(od.has_img.sum())),
    dict(muc="Người có ít nhất một lượt khác ketqua=1", gia_tri=int((per["max"] == 1).sum())),
    dict(muc="Lượt khám khác nằm trong test (chỉ đếm, không đọc nhãn)", gia_tri=len(ot)),
]), "ketqua_luot_khac", "ketqua của 55 người này ở các lượt khám khác")
R4.text("Quyết định: **không sửa nhãn** (theo PGS). Phân tích độ nhạy loại toàn bộ lượt của những người này nằm ở báo cáo P0.1, bảng `4_do_nhay_loai_55_nguoi.csv`.")
R4.save()
for u in sorted(ppl):
    k = g57 & (core.uid2 == u).to_numpy()
    s = score[k & dev]; s = f"điểm L1 OOF {np.nanmean(s):.2f}" if (k & dev).any() and np.isfinite(s).any() else "lượt ở test hoặc bị loại khỏi dev2 — không chấm điểm"
    o = od[od.uid2 == u]
    audit.append(dict(ma_an_danh=anon([f"nguoi{u}"])[0], van_de="bnn=Có nhưng ketqua=2 (nhãn đọc phim âm)",
                      bang_chung=f"{int(k.sum())} lượt nhóm này; {s}; {len(o)} lượt dev khác, {int(o.y.sum())} lượt ketqua=1",
                      quyet_dinh="Giữ nhãn; phân tích độ nhạy loại người này", quy_tac="PG_update §2 P0.4"))

# =========================================================================== P0.5
R5 = Report("tuan1_phase0/P0_5_ghep_nguoi", "P0.5 — Ghép người bị tách mã (khóa nới lỏng)")
multi = core.groupby("uid2").uid.transform("nunique") > 1
span_split = core.groupby("uid2").split.transform("nunique") > 1
span_fold = core[core.split == "dev"].groupby("uid2").fold.nunique()
span_fold = core.uid2.map(span_fold).fillna(0) > 1
bad = multi & (span_split | span_fold)
pct = 100 * bad.mean()
R5.table(pd.DataFrame([
    dict(muc="Nhóm ghép nới lỏng (họ tên chuẩn hóa + năm sinh + giới + tỉnh) gộp > 1 mã", gia_tri=int(core[multi].uid2.nunique())),
    dict(muc="  lượt khám trong các nhóm đó", gia_tri=int(multi.sum())),
    dict(muc="Nhóm vắt qua dev/test", gia_tri=int(core[multi & span_split].uid2.nunique())),
    dict(muc="Nhóm vắt qua hai fold dev (fold gốc)", gia_tri=int(core[multi & span_fold].uid2.nunique())),
    dict(muc="Lượt khám thuộc nhóm vắt qua fold hoặc dev/test", gia_tri=f"{int(bad.sum())}/{len(core)} = {pct:.2f}%"),
    dict(muc="Luật: <1% giữ fold + độ nhạy; ≥1% chia lại fold theo khóa ghép", gia_tri="≥1% → CHIA LẠI FOLD" if pct >= 1 else "<1% → giữ fold"),
    dict(muc="Đã áp dụng", gia_tri=f"uid2 = mã sau ghép; test giữ nguyên; loại {int(((core.split=='dev') & ~core.dev2).sum())} lượt dev của người đã có trong test; "
                                    f"chia lại 5 fold dev theo uid2 (StratifiedGroupKFold, seed {SEED}) → dev2 = {int(core.dev2.sum())} lượt"),
]), "dem_va_quyet_dinh", "Đếm và quyết định P0.5",
    note="*Khóa ghép chỉ dùng cục bộ, không lưu. Mọi kết quả bản 3 (P0.1, P0.4) đã dùng dev2/fold2/uid2.*")
R5.save()
for u, g in core[multi].groupby("uid2"):
    audit.append(dict(ma_an_danh=anon([f"nguoi{u}"])[0], van_de="Một người bị tách thành nhiều mã",
                      bang_chung=f"{g.uid.nunique()} mã, {len(g)} lượt, vắt dev/test: {'có' if g.split.nunique() > 1 else 'không'}, "
                                 f"vắt fold: {'có' if g[g.split=='dev'].fold.nunique() > 1 else 'không'}",
                      quyet_dinh="Gộp thành uid2" + ("; loại lượt dev (người đã có ở test)" if g.split.nunique() > 1 else "") + "; chia lại fold",
                      quy_tac=f"PG_update §2 P0.5 (tỉ lệ {pct:.2f}% ≥ 1%)" if pct >= 1 else "PG_update §2 P0.5"))

# =========================================================================== QA anh -> so kiem toan
Z = np.load(CACHE3 / "img_v3.npz", allow_pickle=False)
qm = pd.read_csv(CACHE3 / "qa_map.csv", encoding="utf-8-sig")
G = pd.read_csv(QA_DIR / "cham_diem.csv", encoding="utf-8-sig").merge(qm, on="qa_id")
zm = np.load(CACHE3 / "zone_missing.npy")
for r in G[G.bi_co].itertuples():
    i = r.img_pos; dg = str(r.danh_gia) if isinstance(r.danh_gia, str) and r.danh_gia.strip() else "chưa chấm"
    audit.append(dict(ma_an_danh=anon([f"anh{Z['img_id'][i]}"])[0], van_de=f"QA ảnh tự động: {r.co}",
                      bang_chung=f"phổi P/T {100*Z['phoi_R'][i]:.1f}%/{100*Z['phoi_L'][i]:.1f}%, tỉ lệ T/P {Z['ti_le_LR'][i]:.2f}, chữ còn lại {int(Z['n_chu_con_lai'][i])}, chấm trực quan: {dg}",
                      quyet_dinh="zone_missing (6 vùng = vector toàn ảnh)" if zm[i] else "Giữ ảnh; chờ/đã chấm trực quan",
                      quy_tac="PG_update §2 QA (không loại âm thầm)"))
audit.append(dict(ma_an_danh="toàn bộ", van_de="Khám thực thể theo vùng gần như trống (P0.6)", bang_chung="xem results_v2 P0_6_kham_6x7",
                  quyet_dinh="Gộp khối khám thành 1 node, bỏ cạnh E6", quy_tac="PG_update §2 P0.6 (đóng)"))
A = pd.DataFrame(audit).rename(columns={"ma_an_danh": "mã ẩn danh", "van_de": "vấn đề", "bang_chung": "bằng chứng", "quyet_dinh": "quyết định", "quy_tac": "quy tắc áp dụng"})
A.to_csv(RES3 / "tuan1_phase0" / "so_kiem_toan.csv", index=False, encoding="utf-8-sig")
print(T4.to_string(index=False)); print(f"P0.4 AUC vs am {auc_vs_neg:.3f}, duong vs P0.4 {auc_vs_pos:.3f}")
print(f"P0.5: {int(bad.sum())}/{len(core)} = {pct:.2f}%"); print("so kiem toan:", A["vấn đề"].str.split(":").str[0].value_counts().to_dict())

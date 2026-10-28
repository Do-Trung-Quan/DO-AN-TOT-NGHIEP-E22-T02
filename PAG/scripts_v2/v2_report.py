"""Bao cao PAG ban 2 tu cache: tuan2 (bien->node + baseline), tuan3 (PG/T0/T2), tuan4 (gia thuyet, Holm), tuan5 (ablation).
python v2_report.py          # chi dev OOF (xac suat da hieu chinh trong fold)
python v2_report.py --test   # mo test MOT lan o cuoi (tuan6_test)
Chi ghi so lieu gop."""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pag_lib import *

TEST = "--test" in sys.argv
B = load_base()
y, hi, uid, batch = B["y"], B["has_img"], B["uid"], B["batch"]
dn = pd.Series(B["tinh"]).astype(str).str.lower().str.contains("nai").to_numpy()
BL = np.load(CACHE / "run_baselines.npz", allow_pickle=False)

def pred(name, split="oof"):
    """Xac suat da hieu chinh (NaN ngoai pham vi). split = oof | test."""
    if f"{split}_cal_{name}" in BL.files: return BL[f"{split}_cal_{name}"]
    if name == "PG-drop": return np.load(CACHE / "run_PG.npz")[f"{split}_drop_cal"]
    f = CACHE / f"run_{name}.npz"
    return np.load(f)[f"{split}_cal"] if f.exists() else None

base_rows = B["test"] if TEST else B["dev"]
GRP = {"toàn bộ": base_rows, "có ảnh": base_rows & hi, "không ảnh": base_rows & ~hi,
       "không ảnh — Đồng Nai": base_rows & ~hi & dn, "không ảnh — tỉnh khác": base_rows & ~hi & ~dn}
SPL = "test" if TEST else "oof"

NOTE_M = ("*Giải thích chỉ số: `roc_auc` = ROC-AUC (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0.5 = đoán bừa, 1 = hoàn hảo), "
          "kèm KTC 95% (khoảng tin cậy, bootstrap theo người 1000 lần). `pr_auc` = PR-AUC (diện tích dưới đường precision–recall; mốc đoán bừa = tỉ lệ dương). "
          "`brier` = sai số bình phương trung bình của xác suất (càng thấp càng tốt), so với `brier_nen` = Brier của việc luôn đoán tỉ lệ nền. "
          "`ece` = sai lệch hiệu chỉnh trung bình (xác suất dự đoán lệch tần suất thật bao nhiêu, 10 khoảng). "
          "`do_doc_hc` = độ dốc hiệu chỉnh (1 là lý tưởng; <1 là dự đoán quá cực đoan), `chan_hc` = chặn hiệu chỉnh (0 là lý tưởng). "
          "`auc_trong_lo` = AUC tính riêng trong từng lô chụp rồi lấy trung bình theo cỡ lô (loại phần mô hình chỉ “nhận ra lô”). "
          "`nhay_tai_dh80/90` = độ nhạy (tỉ lệ ca bệnh bắt được) khi chọn ngưỡng sao cho độ đặc hiệu = 0,80 / 0,90 (chỉ 20% / 10% người âm bị báo nhầm) — chỉ số cho sàng lọc.*")

def sens_at(yy, pp, spec):
    """Do nhay tai do dac hieu >= spec (nguong = phan vi cua ca am)."""
    thr = np.quantile(pp[yy == 0], spec); return float((pp[yy == 1] > thr).mean())

def mrow(name, g, m, p):
    k = m & np.isfinite(p) if p is not None else None
    if p is None or k.sum() < 20 or len(np.unique(y[k])) < 2: return None
    yy, pp = y[k], p[k]
    r = metrics(yy, pp, batch[k] if g == "có ảnh" else None)
    a, lo, hi_, _ = boot_ci(lambda i: roc_auc_score(yy[i], pp[i]), uid[k], n=300 if g.startswith("không ảnh —") else NBOOT)
    r = dict(mo_hinh=name, nhom=g, n=r.pop("n"), duong=r.pop("duong"), roc_auc=a, roc_auc_ci=f"[{lo:.3f}–{hi_:.3f}]", **{kk: vv for kk, vv in r.items() if kk != "roc_auc"})
    r["nhay_tai_dh80"], r["nhay_tai_dh90"] = sens_at(yy, pp, 0.8), sens_at(yy, pp, 0.9)
    return r

def table(names, groups=GRP):
    out = [mrow(n, g, m, pred(n, SPL)) for n in names for g, m in groups.items()]
    return pd.DataFrame([r for r in out if r])

def paired(a, b, m, margin=-0.02):
    pa, pb = pred(a, SPL), pred(b, SPL)
    if pa is None or pb is None: return None
    k = m & np.isfinite(pa) & np.isfinite(pb); yy, A, Bb = y[k], pa[k], pb[k]
    d, lo, hi_, v = boot_ci(lambda i: roc_auc_score(yy[i], A[i]) - roc_auc_score(yy[i], Bb[i]), uid[k])
    return dict(n=int(k.sum()), auc_a=roc_auc_score(yy, A), auc_b=roc_auc_score(yy, Bb), delta=d, ci=f"[{lo:.3f}–{hi_:.3f}]", ci_duoi=lo,
                P_delta_duong=float((v > 0).mean()), ti_le_duoi_bien=float((v <= margin).mean()))

def paired_wb(a, b, m):
    pa, pb = pred(a, SPL), pred(b, SPL); k = m & np.isfinite(pa) & np.isfinite(pb)
    yy, A, Bb, bb = y[k], pa[k], pb[k], batch[k]
    d, lo, hi_, _ = boot_ci(lambda i: within_batch_auc(yy[i], A[i], bb[i]) - within_batch_auc(yy[i], Bb[i], bb[i]), uid[k])
    return d, f"[{lo:.3f}–{hi_:.3f}]"

def holm_adj(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj

def holm(ps, alpha=0.025):
    """Holm mot phia; tra danh sach True/False theo thu tu goc."""
    o = np.argsort(ps); ok = np.zeros(len(ps), bool)
    for r, i in enumerate(o):
        if ps[i] <= alpha / (len(ps) - r): ok[i] = True
        else: break
    return ok

if TEST:
    R = Report("tuan6_test", "Tuần 6 — Đánh giá trên tập test (mở MỘT lần)")
    R.text("Tập test được giữ khoá suốt quá trình chọn mô hình; đây là lần chấm điểm duy nhất. Xác suất đã hiệu chỉnh trong fold rồi lấy trung bình 5 fold.")
    names = ["L0", "L1", "L2", "X2", "L3b", "B3b", "GAT", "T0", "T2", "PG", "PG-nobnn"]
    R.table(table(names), "chi_so_test", "Chỉ số trên test", note=NOTE_M)
    rows = []
    for code, a, b, g in (("H1a", "PG", "L1", "có ảnh"), ("H1b", "PG", "L0", "không ảnh"), ("H3a", "PG", "T2", "có ảnh"), ("H3b", "PG", "T2", "không ảnh")):
        r = paired(a, b, GRP[g]); r and rows.append(dict(ma=code, so_sanh=f"{a} vs {b}", nhom=g, **r))
    T = pd.DataFrame(rows); T["dat_khong_kem"] = T.ci_duoi > -0.02
    R.table(T.drop(columns="ci_duoi"), "gia_thuyet_test", "Bốn so sánh chính trên test",
            note="*Δ = AUC(PG) − AUC(đối chứng). Đạt “không kém” khi cận dưới KTC 95% của Δ > −0,02. `P_delta_duong` = tỉ lệ lần bootstrap Δ>0 (không phải p-value).*")
    R.save(); print(T.to_string(index=False)); sys.exit()

# =================================================================== tuan2
R = Report("tuan2_du_lieu_do_thi", "Tuần 2 — Biến → node, cạnh và các mô hình đối chứng (bản 2)")
R.text("Dữ liệu: dev2 = 6683 lượt khám (5 fold theo NGƯỜI sau khi gộp mã người trùng); test khoá đến cuối. "
       "Cấu trúc dự phòng (P0.6 kích hoạt vì khám theo vùng quá thưa): 6 node vùng phổi + 5 node nhóm lâm sàng + 1 node khám thực thể + 1 node bối cảnh (tỉnh) + 1 node CLS = 14 node; "
       "không có cạnh E6 (khám ↔ đúng vùng). Bỏ biến `nam` (năm công nhận BNN → rò nhãn); giữ `bnn` làm đặc trưng và có ablation PG-nobnn.")
dev = B["dev"]
ce = ClinEncoder().fit(B["raw"], dev); clin, names, present = ce.transform(B["raw"])
vt = []
for g in GNAMES + ["Bối cảnh"]:
    cols = ce.cols(g)
    vt.append(dict(node=g, so_bien=len(cols), so_cot_ma_hoa=clin[g].shape[1],
                   pct_thieu_toan_bo_nhom=100 * (1 - present[g][dev].mean()),
                   pct_thieu_tb_moi_bien=100 * B["raw"].loc[dev, cols].isna().mean().mean(), bien=", ".join(cols)))
R.table(pd.DataFrame(vt), "bien_vao_node", "Biến → node (mã hoá học trên toàn dev chỉ để mô tả; khi huấn luyện mã hoá fit trong từng fold)", nd=1,
        note="*`pct_thieu_toan_bo_nhom`: % lượt mà cả nhóm đều trống (node đó gần như vô tin). `pct_thieu_tb_moi_bien`: % trống trung bình từng biến.*")
kc = CACHE / "k_chon_baselines.csv"
if kc.exists(): R.table(pd.read_csv(kc), "k_chon", "Số láng giềng k của đồ thị bệnh nhân (kNN cosin) chọn theo validation nội bộ từng fold")
BASE_M = ["L0", "L1", "Z1", "L1-khám", "L2", "X0", "X2", "L3b", "B3b", "GAT"]
TB = table(BASE_M)
R.table(TB, "chi_so_doi_chung", "Mô hình đối chứng — dev OOF, xác suất đã hiệu chỉnh trong fold", note=NOTE_M +
        "\n\n*L0 = hồi quy logistic chỉ lâm sàng; L1 = LR chỉ ảnh toàn cục; Z1 = LR trên 6 vùng; L1-khám = LR chỉ biến khám thực thể; L2 = LR lâm sàng + ảnh; "
        "X0/X2 = XGBoost tương ứng L0/L2; L3b/B3b = LR/MLP + điểm rủi ro của láng giềng có ảnh; GAT = mạng chú ý đồ thị trên đồ thị bệnh nhân. "
        "Đồ thị bệnh nhân dựng trên toàn bộ 8030 lượt (chỉ đặc trưng, không nhãn test) — transductive như giao thức Fusion.*")
R.save()

# =================================================================== tuan3
R = Report("tuan3_PG_T0_T2", "Tuần 3 — PG (đồ thị trong từng bệnh nhân) và hai mô hình Transformer bảng")
R.text("PG: 14 node / bệnh nhân, Graph Transformer có thiên lệch theo loại cạnh, modality dropout. T0 = FT-Transformer chỉ lâm sàng; "
       "T2 = FT-Transformer trên đúng các node của PG nhưng KHÔNG có cấu trúc cạnh (mọi token nhìn nhau). "
       "Mỗi mô hình dò 30 cấu hình (cùng không gian với bản 1) trên validation nội bộ, 3 seed ở vòng ngoài, hiệu chỉnh Platt trong fold.")
R.table(table(["PG", "T0", "T2"]), "chi_so", "PG / T0 / T2 — dev OOF đã hiệu chỉnh", note=NOTE_M)
sex = B["raw"]["gioitinh"].astype(str).to_numpy(); st = []
for n in ("PG", "L0", "L1", "X2", "T2"):
    p = pred(n)
    for lab, col in (("tỉnh", B["tinh"].astype(str)), ("giới", sex)):
        for v_ in pd.unique(col):
            for g in ("có ảnh", "không ảnh"):
                k = GRP[g] & (col == v_) & np.isfinite(p)
                if k.sum() >= 30 and 0 < y[k].sum() < k.sum():
                    st.append(dict(mo_hinh=n, theo=lab, gia_tri=v_, nhom=g, n=int(k.sum()), duong=int(y[k].sum()), roc_auc=roc_auc_score(y[k], p[k])))
R.table(pd.DataFrame(st), "phan_tang", "Phân tầng theo tỉnh và giới (ROC-AUC; chỉ ô có ≥30 lượt và đủ hai nhãn)")
for n in ("PG", "T0", "T2"):
    f = CACHE / f"selected_{n}.json"
    if f.exists(): R.table(pd.DataFrame(json.loads(f.read_text())), f"cau_hinh_chon_{n}", f"Cấu hình chọn theo fold — {n}")
p = CACHE / "run_PG.npz"
if p.exists():
    Z = np.load(p); k = dev & hi & np.isfinite(Z["oof_drop_cal"])
    if k.sum():
        R.table(pd.DataFrame([dict(nhom="có ảnh", AUC_day_du=roc_auc_score(y[k], Z["oof_cal"][k]), AUC_khi_bo_anh=roc_auc_score(y[k], Z["oof_drop_cal"][k]))]),
                "bo_anh", "PG khi che toàn bộ node ảnh ở lúc dự đoán (người có phim)",
                note="*Chênh lệch = phần đóng góp của ảnh trong PG đã huấn luyện.*")
    att = Z["att"][dev & hi]
    if np.isfinite(att).any():
        R.table(pd.DataFrame([np.nanmean(att, 0)], columns=REG).round(3), "chu_y_vung", "Chú ý trung bình CLS → 6 vùng (người có phim)",
                note="*Không đánh giá được H4 (chú ý có trúng vùng tổn thương không) vì dữ liệu không có cột `vungtonthuong`.*")
R.save()

# =================================================================== tuan4
R = Report("tuan4_so_sanh_chinh", "Tuần 4 — Kiểm định các giả thuyết khai báo trước (dev OOF)")
R.text("Khai báo trước: `gia_thuyet_khai_bao_truoc_v2.json`. Biên không kém = −0,02 AUC. Holm (một phía, α = 0,025) trên 4 so sánh chính, "
       "dùng tỉ lệ lần bootstrap có Δ ≤ −0,02 làm đại lượng sắp thứ tự. H2 (PG vs PG-E6shuffle) và H4 (chú ý vs vùng tổn thương) KHÔNG đánh giá được.")
rows = []
for code, a, b, g in (("H1a", "PG", "L1", "có ảnh"), ("H1b", "PG", "L0", "không ảnh"), ("H3a", "PG", "T2", "có ảnh"), ("H3b", "PG", "T2", "không ảnh")):
    r = paired(a, b, GRP[g]); r and rows.append(dict(ma=code, so_sanh=f"{a} vs {b}", nhom=g, **r))
H = pd.DataFrame(rows)
if len(H):
    H["dat_theo_KTC"] = H.ci_duoi > -0.02
    H["p_Holm"] = holm_adj(H.ti_le_duoi_bien.to_numpy())
    H["dat_sau_Holm"] = holm(H.ti_le_duoi_bien.to_numpy())
    d, c = paired_wb("PG", "L1", GRP["có ảnh"]); H["delta_trong_lo"] = np.where(H.ma == "H1a", d, np.nan); H["ci_trong_lo"] = np.where(H.ma == "H1a", c, "")
    R.table(H.drop(columns="ci_duoi"), "gia_thuyet_chinh", "Bốn so sánh chính",
            note="*Δ = AUC(PG) − AUC(đối chứng), ghép cặp trên cùng lượt khám. `ci` = KTC 95% bootstrap theo người. `P_delta_duong` = tỉ lệ lần bootstrap Δ>0 "
                 "(mức chắc chắn PG tốt hơn; KHÔNG phải p-value). `ti_le_duoi_bien` = tỉ lệ lần bootstrap Δ ≤ −0,02 (càng nhỏ càng chắc PG không kém). "
                 "`dat_theo_KTC`: cận dưới KTC > −0,02. `p_Holm`: tỉ lệ bootstrap dưới biên sau hiệu chỉnh Holm (dùng như p một phía cho giả thuyết “PG kém hơn quá 0,02”; đạt khi ≤ 0,025). `dat_sau_Holm`: vẫn đạt sau hiệu chỉnh đa so sánh Holm. `delta_trong_lo`: Δ tính bằng AUC trong lô chụp (H1a theo PDF yêu cầu báo cả chỉ số này).*")
    h1 = H[H.ma.str.startswith("H1")].dat_sau_Holm.all()
    R.text(f"**Điểm dừng:** H1 {'đạt' if h1 else 'KHÔNG đạt'} ở cả hai nhóm; H2 không đánh giá được → "
           + ("tiếp tục ablation." if h1 else "theo luật khai báo trước (H1 và H2 cùng không đạt) cần dừng và xem lại thiết kế trước khi đi tiếp."))
ex = []
for a, b, g in (("PG", "PG-shuffle", "có ảnh"), ("PG", "PG-noedge", "có ảnh"), ("PG", "L2", "có ảnh"), ("PG", "L2", "không ảnh"),
                ("PG", "GAT", "có ảnh"), ("PG", "GAT", "không ảnh"), ("PG", "B3b", "không ảnh")):
    r = paired(a, b, GRP[g]); r and ex.append(dict(so_sanh=f"{a} vs {b}", nhom=g, **r))
if ex: R.table(pd.DataFrame(ex).drop(columns=["ci_duoi", "ti_le_duoi_bien"]), "kham_pha", "So sánh khám phá (không hiệu chỉnh đa so sánh, chỉ để tham khảo)")
R.save()

# =================================================================== tuan5
R = Report("tuan5_ablation", "Tuần 5 — Ablation: mỗi lần bỏ/đổi một thành phần của PG")
AB = {"PG": "đầy đủ", "PG-drop": "PG bình thường, bỏ node ảnh lúc đánh giá", "PG-noedge": "bỏ hết cạnh (chỉ tự nối + CLS)", "PG-complete": "đồ thị đầy đủ, một loại cạnh",
      "PG-shuffle": "xáo loại cạnh giữa các vùng (phá giải phẫu)", "PG-global": "1 node ảnh toàn cục thay 6 vùng",
      "PG-noMD": "không modality dropout", "PG-noclin": "chỉ ảnh + khám (bỏ 5 nhóm lâm sàng, người có phim)",
      "PG-noctx": "bỏ node tỉnh", "PG-RGCN": "R-GCN thay Graph Transformer", "PG-nobnn1": "chỉ bỏ biến bnn", "PG-nobnn": "bỏ bnn + 4 biến tiền sử bệnh (tiensuhh, benhhh, tsnoi, tsngoai)",
      "PG-enc-b": "ảnh bằng ResNet50-xrv", "PG-enc-c": "ảnh bằng BioViL-T control",
      "PG-enc-d": "ảnh bằng CheXFound"}
rows = []
for n, desc in AB.items():
    if pred(n) is None: continue
    r = dict(bien_the=n, mo_ta=desc)
    for g in ("có ảnh", "không ảnh"):
        p = pred(n); k = GRP[g] & np.isfinite(p)
        if k.sum() < 20: continue
        r[f"auc_{g}"] = roc_auc_score(y[k], p[k])
        if g == "có ảnh": r["auc_trong_lo"] = within_batch_auc(y[k], p[k], batch[k])
        if n != "PG":
            d = paired(n, "PG", GRP[g]); r[f"delta_{g}"] = d["delta"]; r[f"ci_{g}"] = d["ci"]
    rows.append(r)
R.table(pd.DataFrame(rows), "ablation", "Ablation — dev OOF đã hiệu chỉnh",
        note="*`delta_*` = AUC(biến thể) − AUC(PG) ghép cặp, kèm KTC 95%; âm nghĩa là bỏ thành phần đó làm mô hình kém đi. "
             "PG-E6shuffle / PG-noE6 / PG-1exam trùng PG trong cấu trúc dự phòng nên không chạy.*")
R.save()
print("BAO CAO XONG")

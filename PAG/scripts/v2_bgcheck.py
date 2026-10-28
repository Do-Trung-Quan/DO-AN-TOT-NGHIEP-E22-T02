"""QA 07/10 buoc 1: so dac trung anh CU (img_v2.npz: che chu bang top-hat) voi MOI (img_v2_bg.npz: them to den moi thu ngoai bao loi 2 phoi).
Ly do: nguoi xem 50 anh QA thay ~18/50 con sot chu / che sai cho (phoi deu dung).
So: (1) cosin dac trung cu-moi tung anh; (2) do chinh xac doan lo; (3) AUC L1, AUC trong lo, delta anh so voi chi lo — cu vs moi, hieu ghep cap + KTC.
Luat (da noi voi nguoi dung truoc khi chay): |hieu AUC| < 0.01 va cosin gan 1 -> giu ket qua (2a); nguoc lai chay lai chuoi dev (2b).
Ghi results/tuan1_phase0/P0_2b_xoa_ngoai_phoi. Chi so lieu gop.
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from v2_lib import *

core = load_core()
Zo, Zn = np.load(CACHE / "img_v2.npz"), np.load(CACHE / "img_v2_bg.npz")
assert (Zo["img_id"] == Zn["img_id"]).all()
pos = pd.Series(np.arange(len(Zo["img_id"])), index=Zo["img_id"])
row = core.id.map(pos); hi = core.has_img.to_numpy() & row.notna().to_numpy()
y, uid, fold, batch, dev = core.y.to_numpy(), core.uid2.to_numpy(), core.fold2.to_numpy(), core.batch.to_numpy(), core.dev2.to_numpy()
m = dev & hi; yi, ui, bi = y[m], uid[m], batch[m]

def take(a):
    out = np.zeros((len(core),) + a.shape[1:], np.float32); out[hi] = a[row[hi].astype(int)].astype(np.float32); return out.reshape(len(core), -1)
def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)

ENC = (("biovil_frozen", "BioViL-T đóng băng (a)"), ("xrv", "ResNet50-xrv (b)"), ("biovil_control", "BioViL-T control (c)"))
# (1) cosin
cs = []
for enc, nm in ENC:
    ok = Zo["area"] > 0
    for r, rg in enumerate(REG):
        c = cos(Zo[f"regf_{enc}"][:, r], Zn[f"regf_{enc}"][:, r])[ok[:, r] & (Zn["area"][:, r] > 0)]
        cs.append(dict(bo_ma_hoa=nm, phan=rg, trung_vi=np.median(c), p5=np.percentile(c, 5), nho_nhat=c.min()))
    c = cos(Zo[f"glob_{enc}"], Zn[f"glob_{enc}"])
    cs.append(dict(bo_ma_hoa=nm, phan="toàn ảnh", trung_vi=np.median(c), p5=np.percentile(c, 5), nho_nhat=c.min()))
CS = pd.DataFrame(cs)
pg = Zn["pct_giu"]
geo = pd.DataFrame([dict(muc="Phần ảnh 1024 được giữ (trong bao lồi phổi nới 2%) — trung vị [p5–p95]",
                         gia_tri=f"{np.median(pg):.0f}% [{np.percentile(pg,5):.0f}–{np.percentile(pg,95):.0f}%]"),
                    dict(muc="Khung cắt chiếm (% ảnh) cũ → mới, trung vị", gia_tri=f"{np.median(Zo['pct_crop']):.0f}% → {np.median(Zn['pct_crop']):.0f}%"),
                    dict(muc="Mask 6 vùng giống hệt cũ (số ảnh)", gia_tri=f"{int((Zo['masks'] == Zn['masks']).all((1, 2)).sum())}/{len(pg)}"),
                    dict(muc="Pixel mask vùng trùng cũ (trung vị %)", gia_tri=f"{100*np.median((Zo['masks'] == Zn['masks']).mean((1, 2))):.1f}%")])
print(geo.to_string(index=False)); print(CS.round(4).to_string(index=False), flush=True)

# (2)+(3)
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(batch[hi].reshape(-1, 1))
Bx = np.zeros((len(core), len(oh.categories_[0])), np.float32); Bx[hi] = oh.transform(batch[hi].reshape(-1, 1))
pb = oof_cv(Bx, y, fold, m)[m]
lab = pd.factorize(bi)[0]
res = []
for enc, nm in ENC:
    for kind, key in (("6 vùng", "regf"), ("toàn ảnh", "glob")):
        P, row_ = {}, dict(bo_ma_hoa=nm, dac_trung=kind)
        for ver, Z in (("cu", Zo), ("moi", Zn)):
            X = take(Z[f"{key}_{enc}"]); acc = []
            for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X[m], lab, ui):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[m][tr], lab[tr])
                acc.append((mdl.predict(X[m][va]) == lab[va]).mean())
            p = oof_cv(X, y, fold, m)[m]; pbi = oof_cv(np.column_stack([Bx, X]), y, fold, m)[m]; P[ver] = (p, pbi)
            row_ |= {f"doan_lo_{ver}": np.mean(acc), f"auc_{ver}": roc_auc_score(yi, p), f"auc_trong_lo_{ver}": within_batch_auc(yi, p, bi),
                     f"delta_{ver}": roc_auc_score(yi, pbi) - roc_auc_score(yi, pb)}
        (po, pbo), (pn, pbn) = P["cu"], P["moi"]
        d, lo, h, _ = boot_ci(lambda i: roc_auc_score(yi[i], pn[i]) - roc_auc_score(yi[i], po[i]), ui)
        w, wlo, wh, _ = boot_ci(lambda i: within_batch_auc(yi[i], pn[i], bi[i]) - within_batch_auc(yi[i], po[i], bi[i]), ui)
        row_ |= dict(hieu_auc=d, hieu_auc_ci=f"[{lo:.3f}–{h:.3f}]", hieu_auc_trong_lo=w, hieu_auc_trong_lo_ci=f"[{wlo:.3f}–{wh:.3f}]",
                     tuong_quan_diem=np.corrcoef(po, pn)[0, 1])
        res.append(row_); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row_.items()}, flush=True)
RS = pd.DataFrame(res)
main = RS[(RS.bo_ma_hoa.str.contains(r"\(a\)")) & (RS.dac_trung == "6 vùng")].iloc[0]
cmain = CS[CS.bo_ma_hoa.str.contains(r"\(a\)")]
nho = abs(main.hieu_auc) < 0.01 and abs(main.hieu_auc_trong_lo) < 0.01 and cmain.trung_vi.min() > 0.95
ket = ("2a — thay đổi không đáng kể: GIỮ kết quả hiện có, ghi kiểm tra này vào báo cáo" if nho
       else "2b — thay đổi đáng kể: CHẠY LẠI chuỗi dev trên đặc trưng mới")

R = Report("tuan1_phase0/P0_2b_xoa_ngoai_phoi", "P0.2b — Xoá mọi thứ ngoài vùng phổi: so đặc trưng cũ và mới")
R.text("Lý do: kiểm tra bằng mắt 50 ảnh ngẫu nhiên sau bước che chữ thấy khoảng 18/50 ảnh còn sót chữ hoặc che sai chỗ (đường viền phổi thì đúng cả 50). "
       "Sửa: sau khi phân đoạn lại, tô đen (=0) mọi pixel nằm ngoài bao lồi của hai phổi nới rộng 2% cạnh ảnh; chữ ngoài phổi vì thế bị xoá hết bất kể bộ lọc chữ có bắt được hay không. "
       "Các bước khác giữ nguyên (cắt khung bao phổi + lề 6%, 6 vùng ILO, cùng 3 bộ mã hoá). "
       "Luật quyết định đặt trước khi chạy: nếu với bộ mã hoá chính (a, 6 vùng) |hiệu AUC| < 0,01, |hiệu AUC trong lô| < 0,01 và cosin trung vị mọi vùng > 0,95 thì giữ kết quả cũ (2a); ngược lại chạy lại (2b).")
R.table(geo, "hinh_hoc", "Hình học: phần ảnh được giữ, khung cắt, mask vùng")
R.table(CS.round(4), "cosin_cu_moi", "Độ giống đặc trưng cũ ↔ mới cho cùng một ảnh (cosin)",
        note="*Cosin = 1 nghĩa là vector đặc trưng không đổi; tính trên từng ảnh, báo trung vị, phân vị 5% và nhỏ nhất.*")
R.table(RS.round(4), "so_sanh_auc", "Cũ vs mới: đoán lô, ROC-AUC, AUC trong lô, Δ do ảnh (dev, chỉ lượt có ảnh, hồi quy logistic OOF)",
        note="*`doan_lo`: độ chính xác đoán đúng lô chụp (18 lớp) từ đặc trưng ảnh — thấp hơn là ít dấu vết lô hơn. "
             "`auc`: ROC-AUC (xác suất xếp một ca dương cao hơn một ca âm; 0,5 = đoán bừa). `auc_trong_lo`: AUC tính trong từng lô rồi lấy trung bình theo cỡ lô. "
             "`delta`: AUC(lô + ảnh) − AUC(chỉ lô). `hieu_*` = mới − cũ, KTC 95% bootstrap ghép cặp theo người. `tuong_quan_diem`: tương quan điểm dự đoán cũ–mới.*")
R.text(f"**Kết luận theo luật đặt trước:** {ket}.")
R.save()
print("KET LUAN:", ket); print("BGCHECK XONG")

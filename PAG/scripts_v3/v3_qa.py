"""QA anh ban 3 (PG_update §2): co tu dong moi anh + goi QA truc quan (50 anh ngau nhien phan tang lo x nhan + moi anh bi danh co).
- Goi QA (anh benh nhan) ghi vao PAG/qa_v3_KHONG_CHIA_SE/ — chi de nguoi cham xem, KHONG dua len GitHub / dich vu ngoai.
- Bang cham (cham_diem.csv) chi co ma QA, xao tron, khong lo nguon (ngau nhien / bi co) de cham khong thien lech.
- Bao cao gop (khong anh) ghi vao results_v3/tuan1_phase0/P0_2_xu_ly_anh_QA.
Chay lai sau khi cham: doc cham_diem.csv, tinh ti le dat tren mau ngau nhien, cap nhat zone_missing.
"""
import numpy as np, pandas as pd
from v3_lib import *
from v3_lib import CACHE3, QA_DIR, anon

core = load_core()
Z = np.load(CACHE3 / "img_v3.npz", allow_pickle=False)
ids = Z["img_id"]
meta = core[core.has_img].set_index("id").loc[ids]
tot = Z["phoi_R"] + Z["phoi_L"]
med, mad = np.median(tot), 1.4826 * np.median(np.abs(tot - np.median(tot)))
FLAGS = {
    "dien_tich_phoi_bat_thuong": np.abs(tot - med) / mad > 3,
    "khong_du_2_phoi": (Z["phoi_R"] < 0.01) | (Z["phoi_L"] < 0.01),
    "ti_le_trai_phai_ngoai_0.6_1.6": (Z["ti_le_LR"] < 0.6) | (Z["ti_le_LR"] > 1.6),
    "mask_cham_mep": Z["cham_mep"] > 0,
    "dao_am": (Z["dao_am"] > 0) | (Z["inv_ratio"] >= 1.0),         # anh dao cho mask chac ngang hoac hon anh goc
    "con_chu": Z["n_chu_con_lai"] > 0,
}
F = pd.DataFrame(FLAGS); any_flag = F.any(axis=1).to_numpy()

# ---- mau ngau nhien: 50 anh dev, phan tang lo x nhan (phan bo ti le, du lon nhat), khong dung anh test
rng = np.random.default_rng(SEED)
dev_img = (meta.split == "dev").to_numpy()
st = pd.Series(meta.batch.astype(str) + "|" + meta.y.astype(str)).where(dev_img)
cnt = st.value_counts(); q = 50 * cnt / cnt.sum(); alloc = np.floor(q).astype(int)
alloc[(q - alloc).sort_values(ascending=False).index[:50 - alloc.sum()]] += 1
rand = np.zeros(len(ids), bool)
for s, k in alloc.items():
    if k: rand[rng.choice(np.where(st == s)[0], k, replace=False)] = True
sel = np.where(rand | any_flag)[0]
order = rng.permutation(sel)
qa_map = pd.DataFrame(dict(qa_id=[f"qa_{j+1:03d}" for j in range(len(order))], img_pos=order, ngau_nhien=rand[order], bi_co=any_flag[order],
                           co=[",".join(F.columns[F.iloc[i].to_numpy()]) for i in order]))
qa_map.to_csv(CACHE3 / "qa_map.csv", index=False, encoding="utf-8-sig")   # co vi tri anh -> cuc bo, khong chia se

# ---- goi QA truc quan
QA_DIR.mkdir(exist_ok=True)
grade = QA_DIR / "cham_diem.csv"
if not grade.exists():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy import ndimage
    Tn = np.load(CACHE3 / "qa_thumbs.npz", allow_pickle=False)
    for r in qa_map.itertuples():
        i = r.img_pos; fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
        ax[0].imshow(Tn["goc"][i], cmap="gray"); mg = Tn["mask_goc"][i]
        ax[0].contour(mg == 1, levels=[0.5], colors="lime", linewidths=1); ax[0].contour(mg == 2, levels=[0.5], colors="cyan", linewidths=1)
        ax[0].set_title("Gốc + mask phổi (xanh lá = phải, xanh dương = trái)", fontsize=8)
        ax[1].imshow(Tn["V1"][i], cmap="gray")
        if Tn["to"][i].any(): ax[1].contour(Tn["to"][i], levels=[0.5], colors="red", linewidths=1)
        ax[1].set_title("V1: cắt theo phổi (viền đỏ = chỗ đã tô chữ)", fontsize=8)
        lung = ndimage.zoom(Z["masks"][i] > 0, 2, order=0); lung = ndimage.binary_dilation(lung, iterations=3)
        ax[2].imshow(np.where(lung, Tn["V1"][i], 0), cmap="gray"); ax[2].set_title("V2: xóa ngoài mask phổi", fontsize=8)
        for a in ax: a.axis("off")
        fig.suptitle(r.qa_id); fig.savefig(QA_DIR / f"{r.qa_id}.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(dict(qa_id=qa_map.qa_id, danh_gia="", ghi_chu="")).to_csv(grade, index=False, encoding="utf-8-sig")
    (QA_DIR / "HUONG_DAN.txt").write_text(
        "QA truc quan anh ban 3 (PG_update 17/10). Moi tep qa_XXX.png: trai = anh goc + mask phoi; giua = V1 (cat theo phoi, vien do = cho da to chu); phai = V2 (xoa ngoai phoi).\n"
        "Dien cot danh_gia trong cham_diem.csv bang MOT trong ba gia tri:\n"
        "  dat      : mask om dung hai phoi, khung cat du hai phoi, khong con chu doc duoc (ten/ngay/ma), anh khong dao am\n"
        "  mot_phan : loi nho - mask hut/thua mot phan phoi (vd. goc suon hoanh, dinh phoi), hoac con ky tu le khong doc ra thong tin\n"
        "  hong     : mask sai phoi ro, mat han mot phoi, khung cat cut mat phoi, con doc duoc ten/ngay, hoac dao am\n"
        "Ghi chu tu do vao cot ghi_chu. Thu tu da xao tron; khong biet anh nao la mau ngau nhien hay bi danh co (de cham khong thien lech).\n"
        "Thu muc nay chua anh benh nhan: KHONG dua len GitHub, KHONG chia se, KHONG gui ra dich vu ngoai.\n", encoding="utf-8")

# ---- doc ket qua cham (neu da cham)
G = pd.read_csv(grade, encoding="utf-8-sig").merge(qa_map, on="qa_id")
G["danh_gia"] = G.danh_gia.fillna("").astype(str).str.strip().str.lower()
graded = G.danh_gia.isin(["dat", "mot_phan", "hong"])
zone_missing = np.asarray(FLAGS["khong_du_2_phoi"]).copy()
zone_missing[G.img_pos[G.danh_gia == "hong"].to_numpy()] = True
np.save(CACHE3 / "zone_missing.npy", zone_missing)

# ---- bao cao gop
R = Report("tuan1_phase0/P0_2_xu_ly_anh_QA", "Xử lý ảnh bản 3 + QA (PG_update §2)")
R.text("Từ **ảnh gốc** (không dùng bản 224/512 đã lưu): ảnh xám, pad vuông → kiểm tra đảo âm (phân đoạn trên ảnh và ảnh đảo, so độ chắc của mask) → "
       "phân đoạn phổi bằng PSPNet torchxrayvision (không dùng mask Set B), giữ thành phần lớn nhất mỗi bên → "
       "OCR **chỉ phát hiện** (RapidOCR, mô hình DBNet, không nhận dạng) trên **ảnh gốc** đã pad, bỏ hộp chạm phổi / nghiêng >15° / cao >6% ảnh, tô (inpaint Telea) phần hộp chữ nằm ngoài hộp bao hai phổi → "
       "cắt hộp bao hai phổi + lề 5%, pad vuông, resize 512 (không CenterCrop) → **V1**; **V2** = V1 và xóa pixel ngoài mask phổi (nới 1%). "
       "Đặc trưng trích SAU khi tô chữ.")
cnt_rows = [dict(co=k, so_anh=int(v.sum()), pct=100 * v.mean(), dev=int((v & (meta.split == "dev").to_numpy()).sum()),
                 test=int((v & (meta.split == "test").to_numpy()).sum())) for k, v in FLAGS.items()]
cnt_rows.append(dict(co="ít nhất một cờ", so_anh=int(any_flag.sum()), pct=100 * any_flag.mean(),
                     dev=int((any_flag & (meta.split == "dev").to_numpy()).sum()), test=int((any_flag & (meta.split == "test").to_numpy()).sum())))
R.table(pd.DataFrame(cnt_rows), "co_tu_dong", f"Cờ QA tự động trên {len(ids)} ảnh", nd=1,
        note=f"*Diện tích phổi bất thường = tổng diện tích hai phổi (tỉ lệ ảnh gốc) lệch trung vị quá 3 MAD chuẩn hóa (trung vị {100*med:.1f}%). "
             "Không đủ 2 phổi = một bên < 1% ảnh. Mask chạm mép = mask chạm biên ảnh gốc (phổi có thể bị cắt cụt khi chụp). "
             "Đảo âm = ảnh đảo (255 − x) cho mask phổi chắc ngang hoặc hơn ảnh gốc (đã tự đảo lại nếu gấp ≥1,5 lần). Còn chữ = OCR chạy lại trên ảnh gốc sau khi tô vẫn thấy hộp chữ trong dải lề của khung cắt (ngoài hộp bao hai phổi). OCR chạy trên ảnh gốc (không phải khung cắt phóng to, nơi nó bắt nhầm xương sườn thành chữ); bỏ hộp chạm phổi, nghiêng >15° hoặc cao >6% ảnh; chỉ tô phần nằm ngoài hộp bao hai phổi (trong trường phổi không có chữ in).*")
qd = pd.DataFrame([
    dict(muc="Hộp chữ (sau lọc) chồng lên dải lề khung cắt / ảnh — trung bình [p95]", gia_tri=f"{Z['n_hop_chu'].mean():.2f} [{np.percentile(Z['n_hop_chu'],95):.0f}]"),
    dict(muc="Ảnh có tô chữ", gia_tri=f"{int((Z['pct_to'] > 0).sum())} ({100*(Z['pct_to'] > 0).mean():.1f}%)"),
    dict(muc="Diện tích tô (% khung cắt) — trung vị trong ảnh có tô [p95]", gia_tri=f"{np.median(Z['pct_to'][Z['pct_to']>0]) if (Z['pct_to']>0).any() else 0:.2f}% [{np.percentile(Z['pct_to'],95):.2f}%]"),
    dict(muc="Ảnh có hộp OCR chạm phổi (bỏ, không tô — phổi không có chữ in)", gia_tri=int((Z["hop_chu_trong_phoi"] > 0).sum())),
    dict(muc="Ảnh bị đảo lại (đảo âm theo phân đoạn)", gia_tri=int(Z["dao_am"].sum())),
    dict(muc="Khung cắt chiếm (% ảnh gốc) — trung vị [p5–p95]", gia_tri=f"{np.median(Z['pct_crop']):.0f}% [{np.percentile(Z['pct_crop'],5):.0f}–{np.percentile(Z['pct_crop'],95):.0f}%]"),
    dict(muc="Tỉ lệ trái/phải — trung vị [p5–p95]", gia_tri=f"{np.median(Z['ti_le_LR']):.2f} [{np.percentile(Z['ti_le_LR'],5):.2f}–{np.percentile(Z['ti_le_LR'],95):.2f}]"),
    dict(muc="Vùng có diện tích 0", gia_tri=int((Z["area"] == 0).sum())),
])
R.table(qd, "tom_tat_xu_ly", "Tóm tắt xử lý")
n_r = int(rand.sum()); gr = G[G.ngau_nhien & graded]
R.table(pd.DataFrame([
    dict(muc="Ảnh trong gói QA trực quan", gia_tri=len(qa_map)),
    dict(muc="  trong đó mẫu ngẫu nhiên (dev, phân tầng lô × nhãn)", gia_tri=n_r),
    dict(muc="  trong đó bị đánh cờ", gia_tri=int(any_flag.sum())),
    dict(muc="Đã chấm", gia_tri=int(graded.sum())),
    dict(muc="Mẫu ngẫu nhiên đạt", gia_tri=f"{int((gr.danh_gia=='dat').sum())}/{len(gr)}" + (f" ({100*(gr.danh_gia=='dat').mean():.0f}%)" if len(gr) else "")),
    dict(muc="Cổng QA (≥ 90% mẫu ngẫu nhiên đạt)", gia_tri=("CHƯA CHẤM" if len(gr) < n_r else ("ĐẠT" if (gr.danh_gia == "dat").mean() >= 0.9 else "KHÔNG ĐẠT"))),
    dict(muc="zone_missing (không đủ 2 phổi hoặc chấm hỏng)", gia_tri=int(zone_missing.sum())),
]), "qa_truc_quan", "QA trực quan")
if graded.any():
    R.table(G[graded].groupby(["ngau_nhien", "danh_gia"]).size().unstack(fill_value=0).reset_index(), "ket_qua_cham", "Kết quả chấm theo nguồn")
    gc = G[graded].ghi_chu.fillna("").astype(str).str.lower()
    R.table(pd.DataFrame([
        dict(loai="V1 còn ký tự lẻ R/L (không đọc ra thông tin)", so_anh=int(gc.str.contains("thừa").sum())),
        dict(loai="Sót chữ cạnh vùng đã tô, không đọc được", so_anh=int(gc.str.contains("sót chữ").sum())),
        dict(loai="Phổi chạm / tràn viền khung (V1 hoặc V2)", so_anh=int(gc.str.contains("viền").sum())),
        dict(loai="Còn chữ đọc được (tên/ngày/mã)", so_anh=int(gc.str.contains("đọc được").sum() - gc.str.contains("không đọc được").sum())),
    ]), "ghi_chu_cham", "Loại lỗi ghi nhận khi chấm (một ảnh có thể nhiều loại)",
        note="*Không ảnh nào còn chữ đọc được; lỗi chữ còn lại chỉ là ký tự lẻ hoặc mảnh chữ vỡ hạt ở góc/vai, ngoài phổi.*")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].hist(100 * tot, 50); ax[0].set_title("Tổng diện tích phổi (% ảnh gốc)")
ax[1].hist(np.clip(Z["ti_le_LR"], 0, 3), 50); ax[1].axvline(0.6, c="r"); ax[1].axvline(1.6, c="r"); ax[1].set_title("Tỉ lệ trái/phải")
ax[2].imshow(np.stack([(Z["masks"] == r + 1).mean(0) for r in range(6)]).argmax(0) * ((Z["masks"] > 0).mean(0) > 0.5), cmap="tab10", vmin=0, vmax=9)
ax[2].set_title("Vùng chiếm đa số (khung cắt)"); ax[2].axis("off")
R.fig(fig, "phan_bo_qa", "Phân bố diện tích, tỉ lệ trái/phải và bản đồ vùng trung bình (không có ảnh bệnh nhân)")
R.save()
print(pd.DataFrame(cnt_rows).to_string(index=False)); print(qd.to_string(index=False))
print("goi QA:", len(qa_map), "| ngau nhien:", n_r, "| da cham:", int(graded.sum()), "| zone_missing:", int(zone_missing.sum()))

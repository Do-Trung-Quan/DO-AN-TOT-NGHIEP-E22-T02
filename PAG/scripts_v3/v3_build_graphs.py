"""Viec 2 + 3 (tuan 2, PG_update muc 4): bang "bien -> nut" va dung do thi benh nhan PG, kem QA.

Chot (thay 01/11 + hop 04/11): khong dung `nam`; giu E1/E2; dung PG-global truoc (1 nut anh = C), roi PG 6 vung;
nut "Phoi nhiem/nghe" dung 10 cap nghe (v3_graph_lib). Du lieu: Main_data_Merged_1835_Images.xlsx (qua cache),
fold chuan duy nhat cache_v3/p05_folds.npz.

Moi fold k (0..4) x moi bo cuc (global | vung) x moi ban ma hoa:
  ngoai : fit tren dev & fold != k                      -> dung khi huan luyen that, cham fold k
  noi   : fit tren dev & fold != k & fold != (k+1)%5    -> dung khi chon sieu tham so tren fold (k+1)%5
Ra cache_v3/graphs/pg_{bo_cuc}_fold{k}_{ban}.pt (du lieu that, KHONG commit) + results_v3/tuan2_do_thi_PG/ (so gop).
Moi file gom 7831 do thi (dev 6683 + test 1148); 199 luot dev cua nguoi da co trong test khong dung.
Nhan cua test KHONG luu (y = -1). Khong luu id.

  python PAG/scripts_v3/v3_build_graphs.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_graph_lib import (CACHE3, CLIN_SLOTS, EDGE_NAME, GNAMES, GROUPS, JOB_COLS, JOB_LV, LAYOUTS,  # noqa: E402
                          LEVEL_NAME, NOT_USED, SLOT_NAME, edge_template, edges_of, encode_fold, load_base_graph)
from v3_lib import Report, run_config, MERG  # noqa: E402
from v3_base import IMG3, FOLDS  # noqa: E402
from pag_lib import ClinEncoder, ImgEncoder  # noqa: E402

OUT = CACHE3 / "graphs"
FOLDER = "tuan2_do_thi_PG"
# Cot cam (so khop CHINH XAC ten cot goc cua dac trung, tranh bao nham kieu "vitridau" chua "id")
BANNED_EXACT = {"ketqua", "matdotonthuong", "kichthuoctt", "chatluongphim", "vungtonthuong", "tonthuongkhac", "nam",
                "nam_bnn", "namdt", "hoten", "ten", "sdt", "sobh", "bnncuthe", "id", "ID", "batch", "fvclt", "fev1lt",
                "fvc", "fev1", "gaenler", "img_id", "matdo", "kichthuoc", "chatluong"}


def present_mask(B, E, layout, rows):
    n_img = 1 if layout == "global" else 6
    P = np.zeros((int(rows.sum()), 14), bool)
    P[:, :n_img] = B["has_img"][rows, None]
    for g, nm in enumerate(GNAMES):
        P[:, 6 + g] = E["present"][nm][rows]
    P[:, 12] = E["present"]["Bối cảnh"][rows]
    P[:, 13] = True
    return P


def build_one(B, layout, trm, rows, T):
    E = encode_fold(B, trm, "a", region=(layout == "vung"))
    P = present_mask(B, E, layout, rows)
    ei, et = zip(*[edges_of(T, p) for p in P])
    y = np.where(B["test"][rows], -1, B["y"][rows]).astype(np.int64)
    return E, P, dict(
        bo_cuc=layout, slot_name=SLOT_NAME, edge_type_name={int(k): v for k, v in EDGE_NAME.items()},
        edge_template=torch.as_tensor(T),
        x_img=torch.as_tensor(E["img"][rows]), area=torch.as_tensor(E["area"][rows]),
        x_clin={g: torch.as_tensor(E["clin"][g][rows]) for g in GNAMES + ["Bối cảnh"]},
        ten_dac_trung={g: E["names"][g] for g in GNAMES + ["Bối cảnh"]},
        present=torch.as_tensor(P), edge_index=[torch.as_tensor(e) for e in ei], edge_type=[torch.as_tensor(t) for t in et],
        y=torch.as_tensor(y), has_img=torch.as_tensor(B["has_img"][rows]), dev=torch.as_tensor(B["dev"][rows]),
        test=torch.as_tensor(B["test"][rows]), fold=torch.as_tensor(B["fold"][rows]),
        hang=torch.as_tensor(np.flatnonzero(rows)))


def fit_only_on_train(B, trm, layout):
    """Kiem tra ma hoa chi hoc tu hang train: XOA TRANG moi hang KHONG thuoc train (lam sang -> NaN, anh -> nhieu)
    roi fit lai; tham so ma hoa va dac trung cua hang train phai giong het.
    06/11: truoc do lam sang chi hoan vi cac hang ngoai train voi nhau -> khong doi trung vi/tan suat gop, nen
    encoder lo fit tren toan bang van qua. Xoa trang thi lam doi thong ke gop -> bat duoc ro ri."""
    rng = np.random.default_rng(0)
    raw = B["raw"]; other = ~trm
    raw2 = raw.copy().astype(object); raw2.loc[other] = np.nan
    a, b = ClinEncoder().fit(raw, trm), ClinEncoder().fit(raw2, trm)
    ok_spec = json.dumps(a.spec, default=str) == json.dumps(b.spec, default=str)
    za, _, _ = a.transform(raw.loc[trm]); zb, _, _ = b.transform(raw.loc[trm])
    ok_clin = ok_spec and all(np.array_equal(za[g], zb[g]) for g in za)
    key = "reg_a" if layout == "vung" else "glob_a"
    X = B[key]; X2 = X.copy(); X2[other] = rng.normal(size=X2[other].shape).astype(np.float32)
    tri = trm & B["has_img"]
    z1, z2 = ImgEncoder(128).fit(X, tri).transform(X[tri]), ImgEncoder(128).fit(X2, tri).transform(X2[tri])
    return bool(ok_clin), bool(np.allclose(z1, z2))


def main():
    t0 = time.time()
    B = load_base_graph()
    rows = B["dev"] | B["test"]
    dev, fold, hi = B["dev"], B["fold"], B["has_img"]
    OUT.mkdir(parents=True, exist_ok=True)
    R = Report(FOLDER, "Tuần 2 — Bảng biến → nút và dựng đồ thị bệnh nhân PG (PG-global và PG 6 vùng)")
    R.text("Chốt: không dùng `nam`; giữ cạnh E1/E2; dựng PG-global (1 nút ảnh = C, trung bình 6 vector vùng BioViL-T "
           "đóng băng trên ảnh V2) trước, rồi PG 6 vùng. Nút \"Phơi nhiễm/nghề\" dùng 10 cấp nghề theo mức bụi "
           "(`classify_job_10_levels`, cùng bộ gom của nhánh timeseries) thay cho chữ thô. Fold chuẩn: `p05_folds.npz`. "
           "Mọi phép mã hoá fit trong fold, chỉ trên phần train. Nhãn test không lưu, không nhìn.")

    # ---------------- viec 2: bang bien -> nut (ma hoa fit tren toan dev, CHI de mo ta)
    ce = ClinEncoder().fit(B["raw"], dev)
    _, names, pres = ce.transform(B["raw"])
    rows_tab = []
    for g in GNAMES + ["Bối cảnh"]:
        cs = GROUPS[g] if g in GROUPS else ["tinh"]
        miss = B["raw"].loc[dev, cs].isna().mean() * 100
        rows_tab.append(dict(nut=g, so_bien=len(cs), so_cot_ma_hoa=len(names[g]),
                             pct_luot_trong_ca_nut=round(float((~pres[g][dev]).mean() * 100), 2),
                             pct_thieu_tb_moi_bien=round(float(miss.mean()), 2),
                             bien_thieu_nhieu_nhat=", ".join(f"{c} {v:.0f}%" for c, v in miss.sort_values(ascending=False).head(3).items()),
                             bien=", ".join(cs)))
    rows_tab.append(dict(nut="Ảnh (PG-global)", so_bien=1, so_cot_ma_hoa=128, pct_luot_trong_ca_nut=round(float((~hi[dev]).mean() * 100), 2),
                         pct_thieu_tb_moi_bien=np.nan, bien_thieu_nhieu_nhat="", bien="C = trung bình 6 vector vùng BioViL-T V2 (512) → chuẩn hoá + PCA 128 trong fold"))
    rows_tab.append(dict(nut="Ảnh 6 vùng (PG 6 vùng)", so_bien=6, so_cot_ma_hoa=129, pct_luot_trong_ca_nut=round(float((~hi[dev]).mean() * 100), 2),
                         pct_thieu_tb_moi_bien=np.nan, bien_thieu_nhieu_nhat="", bien="mỗi vùng: vector vùng (512) → một phép chiếu PCA 128 dùng chung 6 vùng, + diện tích vùng"))
    R.table(pd.DataFrame(rows_tab), "bien_vao_nut", "Bảng biến → nút",
            note="*Mã hoá ở bảng này fit trên toàn dev chỉ để mô tả; khi dựng đồ thị mã hoá fit trong từng fold. "
                 "`pct_luot_trong_ca_nut`: % lượt dev mà cả nút trống (nút vắng mặt trong đồ thị). "
                 "Số: điền trung vị + cột cờ thiếu + z-score; danh mục: one-hot (tần suất ≥ 20, tối đa 15 mức, trống là một mức).*")

    lv = []
    for c in JOB_COLS:
        vc = B["raw"].loc[dev, f"{c}_lv"].fillna("trống").value_counts()
        for k, v in vc.items():
            lv.append(dict(cot=c, cap=k, y_nghia=LEVEL_NAME.get(k, "ô trống"), so_luot_dev=int(v)))
    R.table(pd.DataFrame(lv), "cap_nghe_phan_bo", "Phân bố 10 cấp nghề (dev)",
            note="*Cả 4 cột: ô trống giữ là \"trống\" (thiếu), không gom vào lv9 (sửa 06/11; bản trước gom ô trống `cviec`/`pxuong` "
                 "vào lv9 \"không đi làm\" là sai nghĩa). lv9 chỉ còn các ô ghi \"không\", \"ở nhà\", \"tự do\". Một số nghề có bụi rơi vào lv10 (vd \"đánh bóng\", \"vận hành băng tải\") — giới hạn của bộ gom theo từ khóa.*")
    top = []
    for k in LEVEL_NAME:
        s = B["raw"].loc[B["raw"].cviec_lv == k, "cviec"].astype(str).str.strip().str.lower().value_counts().head(15)
        top += [dict(cap=k, ten_cong_viec=n, so_luot=int(v)) for n, v in s.items()]
    pd.DataFrame(top).to_csv(R.d / "cap_nghe_ten_pho_bien.csv", index=False, encoding="utf-8-sig")
    R.text("_15 tên công việc phổ biến nhất mỗi cấp (để soát bằng mắt): `cap_nghe_ten_pho_bien.csv`._\n")

    # ---------------- viec 3: dung do thi
    qa_nc, qa_chk, files = [], [], []
    for k in range(5):
        tr = dev & (fold != k) & (fold >= 0)
        itr = tr & (fold != (k + 1) % 5)
        for ban, trm in (("ngoai", tr), ("noi", itr)):
            for layout in LAYOUTS:
                T = edge_template(layout)
                E, P, G = build_one(B, layout, trm, rows, T)
                G.update(fold_bao_cao=k, ban_ma_hoa=ban, so_hang_fit=int(trm.sum()),
                         fold_validation_noi_bo=(k + 1) % 5 if ban == "noi" else None)
                path = OUT / f"pg_{layout}_fold{k}_{ban}.pt"
                torch.save(G, path)
                files.append(path)
                n_nodes = P.sum(1)
                n_edges = np.array([e.shape[1] for e in G["edge_index"]])
                for has in (True, False):
                    m = B["has_img"][rows] == has
                    for (nn, ne), cnt in pd.Series(list(zip(n_nodes[m], n_edges[m]))).value_counts().items():
                        qa_nc.append(dict(bo_cuc=layout, fold=k, ban=ban, co_phim=has, so_nut=int(nn), so_canh=int(ne), so_do_thi=int(cnt)))
                feat = [n for g in G["ten_dac_trung"].values() for n in g]
                bad = [n for n in feat if n.split("=")[0].split("__")[0] in BANNED_EXACT]
                ok_clin, ok_img = fit_only_on_train(B, trm, layout) if k == 0 else (None, None)
                qa_chk.append(dict(bo_cuc=layout, fold=k, ban=ban, so_do_thi=int(rows.sum()), so_dac_trung_lam_sang=len(feat),
                                   dac_trung_bi_cam=len(bad), ten_bi_cam=", ".join(bad[:5]),
                                   nhan_test_da_an=bool((G["y"][G["test"]] == -1).all()),
                                   ma_hoa_lam_sang_chi_tu_train=ok_clin, ma_hoa_anh_chi_tu_train=ok_img,
                                   dung_luong_MB=round(path.stat().st_size / 2 ** 20, 1)))
                print(f"fold {k} {ban:5s} {layout:6s} xong  {time.time() - t0:.0f}s", flush=True)

    # ---------------- QA
    so = pd.DataFrame([dict(tap="dev", co_phim=int((dev & hi).sum()), khong_phim=int((dev & ~hi).sum())),
                       dict(tap="test (chỉ dựng + đếm)", co_phim=int((B["test"] & hi).sum()), khong_phim=int((B["test"] & ~hi).sum())),
                       dict(tap="không dựng (dev của người đã có trong test)", co_phim=int((~dev & ~B["test"] & hi).sum()),
                            khong_phim=int((~dev & ~B["test"] & ~hi).sum()))])
    so["tong"] = so.co_phim + so.khong_phim
    R.table(so, "so_do_thi", "Số đồ thị", note=f"*Tổng số đồ thị trong mỗi file: {int(rows.sum())} (dev + test).*")

    nc = pd.DataFrame(qa_nc).groupby(["bo_cuc", "co_phim", "so_nut", "so_canh"], as_index=False).so_do_thi.sum()
    nc["so_do_thi"] = nc.so_do_thi // 10          # 5 fold x 2 ban ma hoa giong nhau ve cau truc
    R.table(nc, "nut_canh", "Phân bố số nút / số cạnh (cấu trúc giống nhau ở mọi fold và mọi bản mã hoá)",
            note="*Kỳ vọng: PG-global có phim 9 nút / 29 cạnh (E3 15 + E4 6 + E5 8), không phim 8 / 22 (E3 15 + E5 7); "
                 "PG 6 vùng có phim 14 / 71 (E1 4 + E2 3 + E3 15 + E4 36 + E5 13), không phim 8 / 22. "
                 "Tắt cạnh vùng (`edge_template('vung', noi_vung=False)`, ablation) thì có phim 14 / 64.*")
    exp = {("global", True): (9, 29), ("global", False): (8, 22), ("vung", True): (14, 71), ("vung", False): (8, 22)}
    lech = [(r.bo_cuc, r.co_phim, r.so_nut, r.so_canh, r.so_do_thi) for r in nc.itertuples() if exp[(r.bo_cuc, r.co_phim)] != (r.so_nut, r.so_canh)]

    T_off = edge_template("vung", noi_vung=False)
    assert edges_of(T_off, np.ones(14, bool))[0].shape[1] == 64

    P0 = torch.load(OUT / "pg_vung_fold0_ngoai.pt", weights_only=False)["present"].numpy()
    hi_r = B["has_img"][rows]
    th = pd.DataFrame([dict(nut=SLOT_NAME[s], pct_vang_mat_co_phim=round(float((~P0[hi_r, s]).mean() * 100), 2),
                            pct_vang_mat_khong_phim=round(float((~P0[~hi_r, s]).mean() * 100), 2)) for s in range(14)])
    R.table(th, "vang_mat_theo_nut", "% đồ thị vắng mặt từng nút (PG 6 vùng, fold 0)",
            note="*Nút lâm sàng vắng mặt khi mọi biến của nút đều trống. Giá trị thiếu bên trong nút được điền trong fold + cột cờ thiếu.*")

    chk = pd.DataFrame(qa_chk)
    R.table(chk, "kiem_tra", "Kiểm tra từng file đồ thị",
            note="*`dac_trung_bi_cam` = số đặc trưng có tên khớp danh sách cấm (đọc phim/nhãn, PII, mã lô, `nam`, hô hấp ký % dự đoán). "
                 "`ma_hoa_*_chi_tu_train` (fold 0): xoá trắng mọi hàng không thuộc train (lâm sàng → trống, ảnh → nhiễu) rồi fit lại — tham số mã hoá và đặc trưng "
                 "hàng train phải giống hệt. `nhan_test_da_an`: nhãn test lưu là −1.*")

    ok = (not lech and (chk.dac_trung_bi_cam == 0).all() and chk.nhan_test_da_an.all()
          and chk.ma_hoa_lam_sang_chi_tu_train.dropna().all() and chk.ma_hoa_anh_chi_tu_train.dropna().all())
    R.text(f"## Kết luận QA: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}\n")
    if lech:
        R.text("Số nút/cạnh lệch kỳ vọng: " + "; ".join(map(str, lech)))
    R.text(f"File đồ thị (dữ liệu thật, trong `cache_v3/graphs/`, không commit): {len(files)} file, "
           f"tổng {sum(p.stat().st_size for p in files) / 2 ** 20:.0f} MB.")
    R.save()
    run_config(FOLDER, __file__, data=[IMG3, FOLDS, CACHE3.parent / "cache_v2" / "core.pkl", MERG / "cache_pg" / "clin_raw.pkl"],
               bo_cuc=list(LAYOUTS), ban_ma_hoa={"ngoai": "dev & fold!=k", "noi": "dev & fold!=k & fold!=(k+1)%5"},
               anh="BioViL-T dong bang V2: global = trung binh 6 vector vung; vung = 6 vector; ImgEncoder PCA 128 trong fold",
               nghe="10 cap classify_job_10_levels cho " + ", ".join(JOB_COLS), noi_vung=True, khong_dung=["nam"])
    print(f"\nQA: {'DAT' if ok else 'KHONG DAT'} | {time.time() - t0:.0f}s | bao cao {R.d / 'BAO_CAO.md'}")


if __name__ == "__main__":
    main()

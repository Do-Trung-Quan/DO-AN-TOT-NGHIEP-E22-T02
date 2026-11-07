# Tuần 2 — Bảng biến → nút và dựng đồ thị bệnh nhân PG (PG-global và PG 6 vùng)

Chốt: không dùng `nam`; giữ cạnh E1/E2; dựng PG-global (1 nút ảnh = C, trung bình 6 vector vùng BioViL-T đóng băng trên ảnh V2) trước, rồi PG 6 vùng. Nút "Phơi nhiễm/nghề" dùng 10 cấp nghề theo mức bụi (`classify_job_10_levels`, cùng bộ gom của nhánh timeseries) thay cho chữ thô. Fold chuẩn: `p05_folds.npz`. Mọi phép mã hoá fit trong fold, chỉ trên phần train. Nhãn test không lưu, không nhìn.

### Bảng biến → nút

| nut | so_bien | so_cot_ma_hoa | pct_luot_trong_ca_nut | pct_thieu_tb_moi_bien | bien_thieu_nhieu_nhat | bien |
|---|---|---|---|---|---|---|
| Nhân khẩu & thể trạng | 10 | 15 | 0.000 | 0.000 | gioitinh 0%, namsinh 0%, cao 0% | gioitinh, namsinh, cao, can, bmi, A6, hatd, hatt, mach, theluc |
| Phơi nhiễm/nghề | 21 | 89 | 0.000 | 19.000 | tgian2 99%, cviec2_lv 99%, tgian1 88% | tuoinghe, nampx, cv5nam, tgian1, tgian2, A7, A9a, A9b, A10, A11, A12, F4mu, F4ung, F4KhauTrang, F4gang, F4BHLD, F4Kinh, cviec_lv, pxuong_lv, cviec1_lv, cviec2_lv |
| Hút thuốc | 2 | 4 | 0.000 | 28.550 | slthuoc 57%, hutthuoc 0% | hutthuoc, slthuoc |
| Triệu chứng & tiền sử | 32 | 101 | 0.000 | 53.020 | dhkhac 99%, n2 98%, n3 97% | tiensuhh, benhhh, tsnoi, tsngoai, ho, tdho, tsho, n1, khacdom, loaidom, tdkhacdo, khotho, mdkhotho, tdkhotho, daunguc, vitridau, daulan, tcdau, n2, tgdau, ytodau, n3, ytotang, chaymui, khan, khokhe, dhkhac, metmoi, sutcan, socansut, tgsut, bnn |
| Hô hấp ký | 3 | 3 | 0.000 | 0.000 | fvctt 0%, fev1tt 0%, fev1_fvc 0% | fvctt, fev1tt, fev1_fvc |
| Khám thực thể | 52 | 63 | 0.000 | 88.280 | goduc4 100%, goduc3 100%, goduc2 100% | longnguc, khoangls, rungthan, rungt1, rungt2, rungt3, rungt4, rungt5, rungt6, rungg1, rungg2, rungg3, rungg4, rungg5, rungg6, go, goduc1, goduc2, goduc3, goduc4, goduc5, goduc6, riraopn, ran, ranam, ranno, ranrit, ranngay, vtam1, vtam2, vtam3, vtam4, vtam5, vtam6, vtno1, vtno2, vtno3, vtno4, vtno5, vtno6, vtrit1, vtrit2, vtrit3, vtrit4, vtrit5, vtrit6, vtngay1, vtngay2, vtngay3, vtngay4, vtngay5, vtngay6 |
| Bối cảnh | 1 | 5 | 0.000 | 0.000 | tinh 0% | tinh |
| Ảnh (PG-global) | 1 | 128 | 77.550 | nan |  | C = trung bình 6 vector vùng BioViL-T V2 (512) → chuẩn hoá + PCA 128 trong fold |
| Ảnh 6 vùng (PG 6 vùng) | 6 | 129 | 77.550 | nan |  | mỗi vùng: vector vùng (512) → một phép chiếu PCA 128 dùng chung 6 vùng, + diện tích vùng |

_Tệp: `bien_vao_nut.csv`_

*Mã hoá ở bảng này fit trên toàn dev chỉ để mô tả; khi dựng đồ thị mã hoá fit trong từng fold. `pct_luot_trong_ca_nut`: % lượt dev mà cả nút trống (nút vắng mặt trong đồ thị). Số: điền trung vị + cột cờ thiếu + z-score; danh mục: one-hot (tần suất ≥ 20, tối đa 15 mức, trống là một mức).*

### Phân bố 10 cấp nghề (dev)

| cot | cap | y_nghia | so_luot_dev |
|---|---|---|---|
| cviec | lv1 | bụi nặng (khai thác, đá, nghiền, luyện...) | 1883 |
| cviec | trống | ô trống | 703 |
| cviec | lv3 | sửa chữa, cơ khí, hàn, điện | 702 |
| cviec | lv6 | vận hành máy, lái xe | 653 |
| cviec | lv5 | công nhân sản xuất chung | 644 |
| cviec | lv4 | cân, đóng gói, phân loại | 578 |
| cviec | lv10 | không xác định | 574 |
| cviec | lv2 | gia công, khoan, mài, xây | 527 |
| cviec | lv8 | văn phòng, quản lý, dịch vụ | 279 |
| cviec | lv7 | kiểm nghiệm, xét nghiệm | 140 |
| pxuong | lv1 | bụi nặng (khai thác, đá, nghiền, luyện...) | 1715 |
| pxuong | lv10 | không xác định | 1296 |
| pxuong | trống | ô trống | 861 |
| pxuong | lv4 | cân, đóng gói, phân loại | 833 |
| pxuong | lv3 | sửa chữa, cơ khí, hàn, điện | 681 |
| pxuong | lv5 | công nhân sản xuất chung | 646 |
| pxuong | lv2 | gia công, khoan, mài, xây | 238 |
| pxuong | lv8 | văn phòng, quản lý, dịch vụ | 155 |
| pxuong | lv6 | vận hành máy, lái xe | 145 |
| pxuong | lv7 | kiểm nghiệm, xét nghiệm | 113 |
| cviec1 | trống | ô trống | 5908 |
| cviec1 | lv5 | công nhân sản xuất chung | 163 |
| cviec1 | lv10 | không xác định | 131 |
| cviec1 | lv1 | bụi nặng (khai thác, đá, nghiền, luyện...) | 129 |
| cviec1 | lv3 | sửa chữa, cơ khí, hàn, điện | 104 |
| cviec1 | lv8 | văn phòng, quản lý, dịch vụ | 72 |
| cviec1 | lv2 | gia công, khoan, mài, xây | 71 |
| cviec1 | lv6 | vận hành máy, lái xe | 47 |
| cviec1 | lv9 | ghi "không / ở nhà / tự do" | 37 |
| cviec1 | lv4 | cân, đóng gói, phân loại | 21 |
| cviec2 | trống | ô trống | 6643 |
| cviec2 | lv1 | bụi nặng (khai thác, đá, nghiền, luyện...) | 13 |
| cviec2 | lv5 | công nhân sản xuất chung | 10 |
| cviec2 | lv10 | không xác định | 6 |
| cviec2 | lv3 | sửa chữa, cơ khí, hàn, điện | 4 |
| cviec2 | lv2 | gia công, khoan, mài, xây | 3 |
| cviec2 | lv6 | vận hành máy, lái xe | 2 |
| cviec2 | lv8 | văn phòng, quản lý, dịch vụ | 1 |
| cviec2 | lv4 | cân, đóng gói, phân loại | 1 |

_Tệp: `cap_nghe_phan_bo.csv`_

*Cả 4 cột: ô trống giữ là "trống" (thiếu), không gom vào lv9 (sửa 06/11; bản trước gom ô trống `cviec`/`pxuong` vào lv9 "không đi làm" là sai nghĩa). lv9 chỉ còn các ô ghi "không", "ở nhà", "tự do". Một số nghề có bụi rơi vào lv10 (vd "đánh bóng", "vận hành băng tải") — giới hạn của bộ gom theo từ khóa.*

_15 tên công việc phổ biến nhất mỗi cấp (để soát bằng mắt): `cap_nghe_ten_pho_bien.csv`._


### Số đồ thị

| tap | co_phim | khong_phim | tong |
|---|---|---|---|
| dev | 1500 | 5183 | 6683 |
| test (chỉ dựng + đếm) | 264 | 884 | 1148 |
| không dựng (dev của người đã có trong test) | 71 | 128 | 199 |

_Tệp: `so_do_thi.csv`_

*Tổng số đồ thị trong mỗi file: 7831 (dev + test).*

### Phân bố số nút / số cạnh (cấu trúc giống nhau ở mọi fold và mọi bản mã hoá)

| bo_cuc | co_phim | so_nut | so_canh | so_do_thi |
|---|---|---|---|---|
| global | False | 8 | 22 | 6067 |
| global | True | 9 | 29 | 1764 |
| vung | False | 8 | 22 | 6067 |
| vung | True | 14 | 71 | 1764 |

_Tệp: `nut_canh.csv`_

*Kỳ vọng: PG-global có phim 9 nút / 29 cạnh (E3 15 + E4 6 + E5 8), không phim 8 / 22 (E3 15 + E5 7); PG 6 vùng có phim 14 / 71 (E1 4 + E2 3 + E3 15 + E4 36 + E5 13), không phim 8 / 22. Tắt cạnh vùng (`edge_template('vung', noi_vung=False)`, ablation) thì có phim 14 / 64.*

### % đồ thị vắng mặt từng nút (PG 6 vùng, fold 0)

| nut | pct_vang_mat_co_phim | pct_vang_mat_khong_phim |
|---|---|---|
| RU | 0.000 | 100.000 |
| RM | 0.000 | 100.000 |
| RL | 0.000 | 100.000 |
| LU | 0.000 | 100.000 |
| LM | 0.000 | 100.000 |
| LL | 0.000 | 100.000 |
| Nhân khẩu & thể trạng | 0.000 | 0.000 |
| Phơi nhiễm/nghề | 0.000 | 0.000 |
| Hút thuốc | 0.000 | 0.000 |
| Triệu chứng & tiền sử | 0.000 | 0.000 |
| Hô hấp ký | 0.000 | 0.000 |
| Khám thực thể | 0.000 | 0.000 |
| Bối cảnh | 0.000 | 0.000 |
| Bệnh nhân | 0.000 | 0.000 |

_Tệp: `vang_mat_theo_nut.csv`_

*Nút lâm sàng vắng mặt khi mọi biến của nút đều trống. Giá trị thiếu bên trong nút được điền trong fold + cột cờ thiếu.*

### Kiểm tra từng file đồ thị

| bo_cuc | fold | ban | so_do_thi | so_dac_trung_lam_sang | dac_trung_bi_cam | ten_bi_cam | nhan_test_da_an | ma_hoa_lam_sang_chi_tu_train | ma_hoa_anh_chi_tu_train | dung_luong_MB |
|---|---|---|---|---|---|---|---|---|---|---|
| global | 0 | ngoai | 7831 | 266 | 0 |  | True | True | True | 19.800 |
| vung | 0 | ngoai | 7831 | 266 | 0 |  | True | True | True | 40.700 |
| global | 0 | noi | 7831 | 260 | 0 |  | True | True | True | 19.600 |
| vung | 0 | noi | 7831 | 260 | 0 |  | True | True | True | 40.500 |
| global | 1 | ngoai | 7831 | 270 | 0 |  | True | None | None | 19.900 |
| vung | 1 | ngoai | 7831 | 270 | 0 |  | True | None | None | 40.800 |
| global | 1 | noi | 7831 | 260 | 0 |  | True | None | None | 19.600 |
| vung | 1 | noi | 7831 | 260 | 0 |  | True | None | None | 40.500 |
| global | 2 | ngoai | 7831 | 267 | 0 |  | True | None | None | 19.800 |
| vung | 2 | ngoai | 7831 | 267 | 0 |  | True | None | None | 40.700 |
| global | 2 | noi | 7831 | 258 | 0 |  | True | None | None | 19.500 |
| vung | 2 | noi | 7831 | 258 | 0 |  | True | None | None | 40.400 |
| global | 3 | ngoai | 7831 | 269 | 0 |  | True | None | None | 19.900 |
| vung | 3 | ngoai | 7831 | 269 | 0 |  | True | None | None | 40.700 |
| global | 3 | noi | 7831 | 258 | 0 |  | True | None | None | 19.500 |
| vung | 3 | noi | 7831 | 258 | 0 |  | True | None | None | 40.400 |
| global | 4 | ngoai | 7831 | 266 | 0 |  | True | None | None | 19.800 |
| vung | 4 | ngoai | 7831 | 266 | 0 |  | True | None | None | 40.700 |
| global | 4 | noi | 7831 | 257 | 0 |  | True | None | None | 19.500 |
| vung | 4 | noi | 7831 | 257 | 0 |  | True | None | None | 40.400 |

_Tệp: `kiem_tra.csv`_

*`dac_trung_bi_cam` = số đặc trưng có tên khớp danh sách cấm (đọc phim/nhãn, PII, mã lô, `nam`, hô hấp ký % dự đoán). `ma_hoa_*_chi_tu_train` (fold 0): xoá trắng mọi hàng không thuộc train (lâm sàng → trống, ảnh → nhiễu) rồi fit lại — tham số mã hoá và đặc trưng hàng train phải giống hệt. `nhan_test_da_an`: nhãn test lưu là −1.*

## Kết luận QA: ĐẠT


File đồ thị (dữ liệu thật, trong `cache_v3/graphs/`, không commit): 20 file, tổng 602 MB.

# Tuần 5 — Nhánh embedding + phân loại cổ điển + truy hồi ca tương tự (PAG_V2 §7)

**E1** = vector của node bệnh nhân (CLS) ở lớp cuối PG; ở fold k vector lấy từ mô hình PG huấn luyện trên 4 fold còn lại (1 seed, cùng cấu hình đã chọn). **Thô** = đặc trưng lâm sàng đã mã hoá + vector ảnh toàn cục BioViL-T (PCA 128 trong fold) + cờ có phim — mốc so sánh. Siêu tham số của LR/SVM/kNN chọn bằng CV nội bộ 3 phần theo người trong 4 fold train. E2 (tự giám sát GraphCL) và E3 (graph kernel Weisfeiler–Lehman) **không chạy**: mọi đồ thị bệnh nhân có cùng một khung 14 node, khác nhau chỉ ở đặc trưng node, nên graph kernel suy biến về so khớp đặc trưng (≈ kNN trên vector thô); E2 là tuỳ chọn.

### Phân loại từ vector (ROC-AUC dev OOF, KTC 95% bootstrap theo người 300 lần)

| vector | bo_phan_loai | nhom | roc_auc | ci |
|---|---|---|---|---|
| — | đầu PG (3 seed) | có ảnh | 0.764 | [0.734–0.791] |
| E1 | LR | có ảnh | 0.748 | [0.716–0.773] |
| E1 | SVM | có ảnh | 0.731 | [0.702–0.759] |
| E1 | kNN | có ảnh | 0.753 | [0.719–0.779] |
| tho | LR | có ảnh | 0.747 | [0.717–0.777] |
| tho | SVM | có ảnh | 0.669 | [0.635–0.699] |
| tho | kNN | có ảnh | 0.656 | [0.623–0.684] |
| — | đầu PG (3 seed) | không ảnh | 0.692 | [0.666–0.715] |
| E1 | LR | không ảnh | 0.684 | [0.659–0.710] |
| E1 | SVM | không ảnh | 0.668 | [0.639–0.696] |
| E1 | kNN | không ảnh | 0.683 | [0.656–0.709] |
| tho | LR | không ảnh | 0.713 | [0.690–0.736] |
| tho | SVM | không ảnh | 0.691 | [0.665–0.713] |
| tho | kNN | không ảnh | 0.663 | [0.634–0.685] |

_Tệp: `phan_loai.csv`_

*ROC-AUC = xác suất xếp một ca dương cao hơn một ca âm (0,5 = đoán bừa). Dùng điểm thô của bộ phân loại (AUC không cần hiệu chỉnh). PDF dự báo E1 + LR/SVM thường không hơn chính đầu phân loại của PG.*

### Truy hồi kNN: k người gần nhất trong fold train của mỗi lượt khám ở fold validation

| vector | k | nhom_truy_van | nhan_truy_van | ti_le_lang_gieng_duong | ti_le_duong_nen_train | ti_le_duong_nen_nhom | cung_tinh | cung_tinh_ky_vong | cung_trang_thai_phim | cung_trang_thai_phim_ky_vong | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 5 | có ảnh | 0 | 0.207 | 0.119 | 0.245 | 0.686 | 0.262 | 0.983 | 0.224 | 1133.000 |
| E1 | 5 | có ảnh | 1 | 0.442 | 0.119 | 0.244 | 0.680 | 0.264 | 0.997 | 0.224 | 367.000 |
| E1 | 5 | không ảnh | 0 | 0.080 | 0.119 | 0.082 | 0.860 | 0.248 | 0.998 | 0.776 | 4756.000 |
| E1 | 5 | không ảnh | 1 | 0.137 | 0.119 | 0.082 | 0.885 | 0.247 | 0.999 | 0.775 | 427.000 |
| E1 | 10 | có ảnh | 0 | 0.204 | 0.119 | 0.245 | 0.662 | 0.262 | 0.978 | 0.224 | 1133.000 |
| E1 | 10 | có ảnh | 1 | 0.453 | 0.119 | 0.244 | 0.646 | 0.264 | 0.995 | 0.224 | 367.000 |
| E1 | 10 | không ảnh | 0 | 0.080 | 0.119 | 0.082 | 0.829 | 0.248 | 0.997 | 0.776 | 4756.000 |
| E1 | 10 | không ảnh | 1 | 0.142 | 0.119 | 0.082 | 0.865 | 0.247 | 0.999 | 0.775 | 427.000 |
| E1 | 20 | có ảnh | 0 | 0.206 | 0.119 | 0.245 | 0.635 | 0.262 | 0.968 | 0.224 | 1133.000 |
| E1 | 20 | có ảnh | 1 | 0.465 | 0.119 | 0.244 | 0.620 | 0.264 | 0.991 | 0.224 | 367.000 |
| E1 | 20 | không ảnh | 0 | 0.080 | 0.119 | 0.082 | 0.789 | 0.248 | 0.995 | 0.776 | 4756.000 |
| E1 | 20 | không ảnh | 1 | 0.142 | 0.119 | 0.082 | 0.828 | 0.247 | 0.998 | 0.775 | 427.000 |
| tho | 5 | có ảnh | 0 | 0.096 | 0.119 | 0.245 | 0.709 | 0.262 | 0.392 | 0.224 | 1133.000 |
| tho | 5 | có ảnh | 1 | 0.160 | 0.119 | 0.244 | 0.755 | 0.264 | 0.359 | 0.224 | 367.000 |
| tho | 5 | không ảnh | 0 | 0.078 | 0.119 | 0.082 | 0.785 | 0.248 | 0.997 | 0.776 | 4756.000 |
| tho | 5 | không ảnh | 1 | 0.126 | 0.119 | 0.082 | 0.812 | 0.247 | 0.998 | 0.775 | 427.000 |
| tho | 10 | có ảnh | 0 | 0.101 | 0.119 | 0.245 | 0.683 | 0.262 | 0.370 | 0.224 | 1133.000 |
| tho | 10 | có ảnh | 1 | 0.150 | 0.119 | 0.244 | 0.719 | 0.264 | 0.343 | 0.224 | 367.000 |
| tho | 10 | không ảnh | 0 | 0.079 | 0.119 | 0.082 | 0.754 | 0.248 | 0.995 | 0.776 | 4756.000 |
| tho | 10 | không ảnh | 1 | 0.122 | 0.119 | 0.082 | 0.793 | 0.247 | 0.996 | 0.775 | 427.000 |
| tho | 20 | có ảnh | 0 | 0.105 | 0.119 | 0.245 | 0.659 | 0.262 | 0.363 | 0.224 | 1133.000 |
| tho | 20 | có ảnh | 1 | 0.149 | 0.119 | 0.244 | 0.695 | 0.264 | 0.343 | 0.224 | 367.000 |
| tho | 20 | không ảnh | 0 | 0.080 | 0.119 | 0.082 | 0.726 | 0.248 | 0.989 | 0.776 | 4756.000 |
| tho | 20 | không ảnh | 1 | 0.118 | 0.119 | 0.082 | 0.760 | 0.247 | 0.994 | 0.775 | 427.000 |

_Tệp: `truy_hoi_knn.csv`_

*`ti_le_lang_gieng_duong` = Precision@k cho truy vấn dương (nhan_truy_van=1): tỉ lệ láng giềng cũng dương; với truy vấn âm (=0) thì 1 − giá trị này là tỉ lệ láng giềng cùng nhãn. So với `ti_le_duong_nen_train` (tỉ lệ dương toàn train) và `ti_le_duong_nen_nhom` (tỉ lệ dương trong train cùng nhóm phim). `cung_tinh` = tỉ lệ láng giềng cùng tỉnh với truy vấn, so với `cung_tinh_ky_vong` (nếu chọn láng giềng ngẫu nhiên). `cung_trang_thai_phim` = tỉ lệ láng giềng cùng có/không phim với truy vấn, so với kỳ vọng ngẫu nhiên. Cao hơn nhiều so với kỳ vọng nghĩa là vector bị “dồn” theo tỉnh / theo phim.*

### Siêu tham số chọn trong từng fold (CV nội bộ)

| fold | vector | bo_phan_loai | C | n_neighbors |
|---|---|---|---|---|
| 0 | E1 | LR | 0.010 | nan |
| 0 | E1 | SVM | 0.300 | nan |
| 0 | E1 | kNN | nan | 100.000 |
| 0 | tho | LR | 0.010 | nan |
| 0 | tho | SVM | 0.300 | nan |
| 0 | tho | kNN | nan | 100.000 |
| 1 | E1 | LR | 0.010 | nan |
| 1 | E1 | SVM | 0.300 | nan |
| 1 | E1 | kNN | nan | 100.000 |
| 1 | tho | LR | 0.010 | nan |
| 1 | tho | SVM | 0.300 | nan |
| 1 | tho | kNN | nan | 100.000 |
| 2 | E1 | LR | 0.010 | nan |
| 2 | E1 | SVM | 0.300 | nan |
| 2 | E1 | kNN | nan | 100.000 |
| 2 | tho | LR | 0.010 | nan |
| 2 | tho | SVM | 0.300 | nan |
| 2 | tho | kNN | nan | 100.000 |
| 3 | E1 | LR | 0.010 | nan |
| 3 | E1 | SVM | 0.300 | nan |
| 3 | E1 | kNN | nan | 100.000 |
| 3 | tho | LR | 0.010 | nan |
| 3 | tho | SVM | 0.300 | nan |
| 3 | tho | kNN | nan | 100.000 |
| 4 | E1 | LR | 0.010 | nan |
| 4 | E1 | SVM | 0.300 | nan |
| 4 | E1 | kNN | nan | 100.000 |
| 4 | tho | LR | 0.010 | nan |
| 4 | tho | SVM | 0.300 | nan |
| 4 | tho | kNN | nan | 100.000 |

_Tệp: `sieu_tham_so_chon.csv`_

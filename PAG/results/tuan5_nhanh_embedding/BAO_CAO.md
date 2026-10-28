# Tuần 5 — Nhánh embedding + phân loại cổ điển + truy hồi ca tương tự (PAG_V2 §7)

**E1** = vector của node bệnh nhân (CLS) ở lớp cuối PG; ở fold k vector lấy từ mô hình PG huấn luyện trên 4 fold còn lại (1 seed, cùng cấu hình đã chọn). **Thô** = đặc trưng lâm sàng đã mã hoá + vector ảnh toàn cục BioViL-T (PCA 128 trong fold) + cờ có phim — mốc so sánh. Siêu tham số của LR/SVM/kNN chọn bằng CV nội bộ 3 phần theo người trong 4 fold train. E2 (tự giám sát GraphCL) và E3 (graph kernel Weisfeiler–Lehman) **không chạy**: mọi đồ thị bệnh nhân có cùng một khung 14 node, khác nhau chỉ ở đặc trưng node, nên graph kernel suy biến về so khớp đặc trưng (≈ kNN trên vector thô); E2 là tuỳ chọn.

### Phân loại từ vector (ROC-AUC dev OOF, KTC 95% bootstrap theo người 300 lần)

| vector | bo_phan_loai | nhom | roc_auc | ci |
|---|---|---|---|---|
| — | đầu PG (3 seed) | có ảnh | 0.765 | [0.735–0.790] |
| E1 | LR | có ảnh | 0.745 | [0.712–0.775] |
| E1 | SVM | có ảnh | 0.739 | [0.707–0.765] |
| E1 | kNN | có ảnh | 0.750 | [0.718–0.779] |
| tho | LR | có ảnh | 0.732 | [0.704–0.760] |
| tho | SVM | có ảnh | 0.668 | [0.635–0.699] |
| tho | kNN | có ảnh | 0.655 | [0.622–0.687] |
| — | đầu PG (3 seed) | không ảnh | 0.692 | [0.666–0.718] |
| E1 | LR | không ảnh | 0.689 | [0.660–0.714] |
| E1 | SVM | không ảnh | 0.682 | [0.654–0.707] |
| E1 | kNN | không ảnh | 0.676 | [0.647–0.700] |
| tho | LR | không ảnh | 0.713 | [0.691–0.737] |
| tho | SVM | không ảnh | 0.691 | [0.665–0.714] |
| tho | kNN | không ảnh | 0.662 | [0.633–0.684] |

_Tệp: `phan_loai.csv`_

*ROC-AUC = xác suất xếp một ca dương cao hơn một ca âm (0,5 = đoán bừa). Dùng điểm thô của bộ phân loại (AUC không cần hiệu chỉnh). PDF dự báo E1 + LR/SVM thường không hơn chính đầu phân loại của PG.*

### Truy hồi kNN: k người gần nhất trong fold train của mỗi lượt khám ở fold validation

| vector | k | nhom_truy_van | nhan_truy_van | ti_le_lang_gieng_duong | ti_le_duong_nen_train | ti_le_duong_nen_nhom | cung_tinh | cung_tinh_ky_vong | cung_trang_thai_phim | cung_trang_thai_phim_ky_vong | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 5 | có ảnh | 0 | 0.196 | 0.119 | 0.245 | 0.687 | 0.262 | 0.936 | 0.224 | 1133.000 |
| E1 | 5 | có ảnh | 1 | 0.431 | 0.119 | 0.244 | 0.682 | 0.264 | 0.972 | 0.224 | 367.000 |
| E1 | 5 | không ảnh | 0 | 0.081 | 0.119 | 0.082 | 0.852 | 0.248 | 0.993 | 0.776 | 4756.000 |
| E1 | 5 | không ảnh | 1 | 0.152 | 0.119 | 0.082 | 0.847 | 0.247 | 0.998 | 0.775 | 427.000 |
| E1 | 10 | có ảnh | 0 | 0.199 | 0.119 | 0.245 | 0.663 | 0.262 | 0.921 | 0.224 | 1133.000 |
| E1 | 10 | có ảnh | 1 | 0.439 | 0.119 | 0.244 | 0.658 | 0.264 | 0.967 | 0.224 | 367.000 |
| E1 | 10 | không ảnh | 0 | 0.081 | 0.119 | 0.082 | 0.825 | 0.248 | 0.991 | 0.776 | 4756.000 |
| E1 | 10 | không ảnh | 1 | 0.156 | 0.119 | 0.082 | 0.825 | 0.247 | 0.995 | 0.775 | 427.000 |
| E1 | 20 | có ảnh | 0 | 0.200 | 0.119 | 0.245 | 0.636 | 0.262 | 0.899 | 0.224 | 1133.000 |
| E1 | 20 | có ảnh | 1 | 0.449 | 0.119 | 0.244 | 0.638 | 0.264 | 0.958 | 0.224 | 367.000 |
| E1 | 20 | không ảnh | 0 | 0.082 | 0.119 | 0.082 | 0.788 | 0.248 | 0.988 | 0.776 | 4756.000 |
| E1 | 20 | không ảnh | 1 | 0.153 | 0.119 | 0.082 | 0.791 | 0.247 | 0.992 | 0.775 | 427.000 |
| tho | 5 | có ảnh | 0 | 0.095 | 0.119 | 0.245 | 0.705 | 0.262 | 0.379 | 0.224 | 1133.000 |
| tho | 5 | có ảnh | 1 | 0.150 | 0.119 | 0.244 | 0.759 | 0.264 | 0.356 | 0.224 | 367.000 |
| tho | 5 | không ảnh | 0 | 0.078 | 0.119 | 0.082 | 0.785 | 0.248 | 0.997 | 0.776 | 4756.000 |
| tho | 5 | không ảnh | 1 | 0.126 | 0.119 | 0.082 | 0.812 | 0.247 | 0.998 | 0.775 | 427.000 |
| tho | 10 | có ảnh | 0 | 0.102 | 0.119 | 0.245 | 0.682 | 0.262 | 0.367 | 0.224 | 1133.000 |
| tho | 10 | có ảnh | 1 | 0.146 | 0.119 | 0.244 | 0.725 | 0.264 | 0.338 | 0.224 | 367.000 |
| tho | 10 | không ảnh | 0 | 0.079 | 0.119 | 0.082 | 0.754 | 0.248 | 0.995 | 0.776 | 4756.000 |
| tho | 10 | không ảnh | 1 | 0.122 | 0.119 | 0.082 | 0.793 | 0.247 | 0.996 | 0.775 | 427.000 |
| tho | 20 | có ảnh | 0 | 0.107 | 0.119 | 0.245 | 0.659 | 0.262 | 0.363 | 0.224 | 1133.000 |
| tho | 20 | có ảnh | 1 | 0.146 | 0.119 | 0.244 | 0.694 | 0.264 | 0.335 | 0.224 | 367.000 |
| tho | 20 | không ảnh | 0 | 0.080 | 0.119 | 0.082 | 0.726 | 0.248 | 0.989 | 0.776 | 4756.000 |
| tho | 20 | không ảnh | 1 | 0.118 | 0.119 | 0.082 | 0.760 | 0.247 | 0.995 | 0.775 | 427.000 |

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

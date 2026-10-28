# Tuần 3 — PG (đồ thị trong từng bệnh nhân) và hai mô hình Transformer bảng

PG: 14 node / bệnh nhân, Graph Transformer có thiên lệch theo loại cạnh, modality dropout. T0 = FT-Transformer chỉ lâm sàng; T2 = FT-Transformer trên đúng các node của PG nhưng KHÔNG có cấu trúc cạnh (mọi token nhìn nhau). Mỗi mô hình dò 30 cấu hình (cùng không gian với bản 1) trên validation nội bộ, 3 seed ở vòng ngoài, hiệu chỉnh Platt trong fold.

### PG / T0 / T2 — dev OOF đã hiệu chỉnh

| mo_hinh | nhom | n | duong | roc_auc | roc_auc_ci | pr_auc | brier | brier_nen | ece | do_doc_hc | chan_hc | nhay_tai_dh80 | nhay_tai_dh90 | auc_trong_lo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PG | toàn bộ | 6683 | 794 | 0.747 | [0.729–0.766] | 0.331 | 0.093 | 0.105 | 0.012 | 0.963 | -0.126 | 0.539 | 0.388 | nan |
| PG | có ảnh | 1500 | 367 | 0.754 | [0.728–0.781] | 0.482 | 0.157 | 0.185 | 0.019 | 0.888 | -0.088 | 0.572 | 0.351 | 0.695 |
| PG | không ảnh | 5183 | 427 | 0.679 | [0.652–0.703] | 0.143 | 0.074 | 0.076 | 0.011 | 0.889 | -0.345 | 0.422 | 0.218 | nan |
| PG | không ảnh — Đồng Nai | 1205 | 110 | 0.618 | [0.563–0.665] | 0.139 | 0.082 | 0.083 | 0.020 | 0.652 | -0.758 | 0.391 | 0.155 | nan |
| PG | không ảnh — tỉnh khác | 3978 | 317 | 0.698 | [0.672–0.726] | 0.145 | 0.072 | 0.073 | 0.016 | 0.942 | -0.274 | 0.442 | 0.227 | nan |
| T0 | toàn bộ | 6683 | 794 | 0.707 | [0.686–0.726] | 0.276 | 0.097 | 0.105 | 0.012 | 0.936 | -0.103 | 0.467 | 0.336 | nan |
| T0 | có ảnh | 1500 | 367 | 0.652 | [0.617–0.683] | 0.387 | 0.175 | 0.185 | 0.038 | 0.882 | -0.116 | 0.401 | 0.226 | 0.589 |
| T0 | không ảnh | 5183 | 427 | 0.639 | [0.611–0.666] | 0.126 | 0.074 | 0.076 | 0.010 | 0.853 | -0.326 | 0.344 | 0.183 | nan |
| T0 | không ảnh — Đồng Nai | 1205 | 110 | 0.641 | [0.588–0.693] | 0.162 | 0.081 | 0.083 | 0.021 | 0.971 | 0.145 | 0.345 | 0.218 | nan |
| T0 | không ảnh — tỉnh khác | 3978 | 317 | 0.640 | [0.612–0.672] | 0.120 | 0.072 | 0.073 | 0.011 | 0.837 | -0.422 | 0.331 | 0.180 | nan |
| T2 | toàn bộ | 6683 | 794 | 0.737 | [0.717–0.757] | 0.352 | 0.092 | 0.105 | 0.016 | 1.044 | 0.125 | 0.535 | 0.413 | nan |
| T2 | có ảnh | 1500 | 367 | 0.761 | [0.732–0.790] | 0.522 | 0.154 | 0.185 | 0.042 | 0.997 | 0.233 | 0.605 | 0.390 | 0.703 |
| T2 | không ảnh | 5183 | 427 | 0.664 | [0.636–0.690] | 0.146 | 0.074 | 0.076 | 0.013 | 0.864 | -0.361 | 0.382 | 0.234 | nan |
| T2 | không ảnh — Đồng Nai | 1205 | 110 | 0.623 | [0.566–0.675] | 0.150 | 0.082 | 0.083 | 0.011 | 0.714 | -0.571 | 0.382 | 0.227 | nan |
| T2 | không ảnh — tỉnh khác | 3978 | 317 | 0.678 | [0.647–0.706] | 0.148 | 0.071 | 0.073 | 0.017 | 0.910 | -0.305 | 0.388 | 0.249 | nan |

_Tệp: `chi_so.csv`_

*Giải thích chỉ số: `roc_auc` = ROC-AUC (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0.5 = đoán bừa, 1 = hoàn hảo), kèm KTC 95% (khoảng tin cậy, bootstrap theo người 1000 lần). `pr_auc` = PR-AUC (diện tích dưới đường precision–recall; mốc đoán bừa = tỉ lệ dương). `brier` = sai số bình phương trung bình của xác suất (càng thấp càng tốt), so với `brier_nen` = Brier của việc luôn đoán tỉ lệ nền. `ece` = sai lệch hiệu chỉnh trung bình (xác suất dự đoán lệch tần suất thật bao nhiêu, 10 khoảng). `do_doc_hc` = độ dốc hiệu chỉnh (1 là lý tưởng; <1 là dự đoán quá cực đoan), `chan_hc` = chặn hiệu chỉnh (0 là lý tưởng). `auc_trong_lo` = AUC tính riêng trong từng lô chụp rồi lấy trung bình theo cỡ lô (loại phần mô hình chỉ “nhận ra lô”). `nhay_tai_dh80/90` = độ nhạy (tỉ lệ ca bệnh bắt được) khi chọn ngưỡng sao cho độ đặc hiệu = 0,80 / 0,90 (chỉ 20% / 10% người âm bị báo nhầm) — chỉ số cho sàng lọc.*

### Phân tầng theo tỉnh và giới (ROC-AUC; chỉ ô có ≥30 lượt và đủ hai nhãn)

| mo_hinh | theo | gia_tri | nhom | n | duong | roc_auc |
|---|---|---|---|---|---|---|
| PG | tỉnh | Hai Duong | có ảnh | 352 | 76 | 0.706 |
| PG | tỉnh | Hai Duong | không ảnh | 943 | 49 | 0.619 |
| PG | tỉnh | Binh Dinh | có ảnh | 512 | 131 | 0.845 |
| PG | tỉnh | Binh Dinh | không ảnh | 538 | 3 | 0.706 |
| PG | tỉnh | Thai Nguyen | có ảnh | 633 | 159 | 0.677 |
| PG | tỉnh | Thai Nguyen | không ảnh | 1957 | 177 | 0.644 |
| PG | tỉnh | Phu Yen | không ảnh | 540 | 88 | 0.643 |
| PG | tỉnh | Dong Nai | không ảnh | 1205 | 110 | 0.618 |
| PG | giới | Nam | có ảnh | 1177 | 313 | 0.749 |
| PG | giới | Nam | không ảnh | 4306 | 381 | 0.671 |
| PG | giới | Nu | có ảnh | 323 | 54 | 0.751 |
| PG | giới | Nu | không ảnh | 877 | 46 | 0.706 |
| L0 | tỉnh | Hai Duong | có ảnh | 352 | 76 | 0.592 |
| L0 | tỉnh | Hai Duong | không ảnh | 943 | 49 | 0.661 |
| L0 | tỉnh | Binh Dinh | có ảnh | 512 | 131 | 0.705 |
| L0 | tỉnh | Binh Dinh | không ảnh | 538 | 3 | 0.667 |
| L0 | tỉnh | Thai Nguyen | có ảnh | 633 | 159 | 0.578 |
| L0 | tỉnh | Thai Nguyen | không ảnh | 1957 | 177 | 0.674 |
| L0 | tỉnh | Phu Yen | không ảnh | 540 | 88 | 0.660 |
| L0 | tỉnh | Dong Nai | không ảnh | 1205 | 110 | 0.621 |
| L0 | giới | Nam | có ảnh | 1177 | 313 | 0.616 |
| L0 | giới | Nam | không ảnh | 4306 | 381 | 0.699 |
| L0 | giới | Nu | có ảnh | 323 | 54 | 0.641 |
| L0 | giới | Nu | không ảnh | 877 | 46 | 0.729 |
| L1 | tỉnh | Hai Duong | có ảnh | 352 | 76 | 0.701 |
| L1 | tỉnh | Binh Dinh | có ảnh | 512 | 131 | 0.836 |
| L1 | tỉnh | Thai Nguyen | có ảnh | 633 | 159 | 0.668 |
| L1 | giới | Nam | có ảnh | 1177 | 313 | 0.736 |
| L1 | giới | Nu | có ảnh | 323 | 54 | 0.746 |
| X2 | tỉnh | Hai Duong | có ảnh | 352 | 76 | 0.683 |
| X2 | tỉnh | Hai Duong | không ảnh | 943 | 49 | 0.629 |
| X2 | tỉnh | Binh Dinh | có ảnh | 512 | 131 | 0.813 |
| X2 | tỉnh | Binh Dinh | không ảnh | 538 | 3 | 0.522 |
| X2 | tỉnh | Thai Nguyen | có ảnh | 633 | 159 | 0.631 |
| X2 | tỉnh | Thai Nguyen | không ảnh | 1957 | 177 | 0.694 |
| X2 | tỉnh | Phu Yen | không ảnh | 540 | 88 | 0.703 |
| X2 | tỉnh | Dong Nai | không ảnh | 1205 | 110 | 0.607 |
| X2 | giới | Nam | có ảnh | 1177 | 313 | 0.702 |
| X2 | giới | Nam | không ảnh | 4306 | 381 | 0.709 |
| X2 | giới | Nu | có ảnh | 323 | 54 | 0.734 |
| X2 | giới | Nu | không ảnh | 877 | 46 | 0.722 |
| T2 | tỉnh | Hai Duong | có ảnh | 352 | 76 | 0.731 |
| T2 | tỉnh | Hai Duong | không ảnh | 943 | 49 | 0.640 |
| T2 | tỉnh | Binh Dinh | có ảnh | 512 | 131 | 0.816 |
| T2 | tỉnh | Binh Dinh | không ảnh | 538 | 3 | 0.660 |
| T2 | tỉnh | Thai Nguyen | có ảnh | 633 | 159 | 0.717 |
| T2 | tỉnh | Thai Nguyen | không ảnh | 1957 | 177 | 0.619 |
| T2 | tỉnh | Phu Yen | không ảnh | 540 | 88 | 0.630 |
| T2 | tỉnh | Dong Nai | không ảnh | 1205 | 110 | 0.623 |
| T2 | giới | Nam | có ảnh | 1177 | 313 | 0.766 |
| T2 | giới | Nam | không ảnh | 4306 | 381 | 0.658 |
| T2 | giới | Nu | có ảnh | 323 | 54 | 0.708 |
| T2 | giới | Nu | không ảnh | 877 | 46 | 0.678 |

_Tệp: `phan_tang.csv`_

### Cấu hình chọn theo fold — PG

| fold | cau_hinh | dim | so_lop | dropout | lr | weight_decay | p_modality_dropout | epoch |
|---|---|---|---|---|---|---|---|---|
| 0 | 22 | 128 | 2 | 0.300 | 0.001 | 0.010 | 0.500 | 17 |
| 1 | 27 | 128 | 3 | 0.300 | 0.001 | 0.000 | 0.200 | 14 |
| 2 | 6 | 64 | 3 | 0.100 | 0.001 | 0.000 | 0.200 | 13 |
| 3 | 4 | 64 | 2 | 0.300 | 0.000 | 0.010 | 0.200 | 17 |
| 4 | 1 | 64 | 2 | 0.100 | 0.001 | 0.000 | 0.200 | 8 |

_Tệp: `cau_hinh_chon_PG.csv`_

### Cấu hình chọn theo fold — T0

| fold | cau_hinh | dim | so_lop | dropout | lr | weight_decay | p_modality_dropout | epoch |
|---|---|---|---|---|---|---|---|---|
| 0 | 27 | 128 | 3 | 0.300 | 0.001 | 0.000 | 0.200 | 28 |
| 1 | 27 | 128 | 3 | 0.300 | 0.001 | 0.000 | 0.200 | 25 |
| 2 | 11 | 64 | 3 | 0.300 | 0.001 | 0.000 | 0.200 | 32 |
| 3 | 26 | 128 | 3 | 0.100 | 0.001 | 0.010 | 0.500 | 13 |
| 4 | 8 | 64 | 3 | 0.300 | 0.000 | 0.000 | 0.300 | 27 |

_Tệp: `cau_hinh_chon_T0.csv`_

### Cấu hình chọn theo fold — T2

| fold | cau_hinh | dim | so_lop | dropout | lr | weight_decay | p_modality_dropout | epoch |
|---|---|---|---|---|---|---|---|---|
| 0 | 14 | 128 | 2 | 0.100 | 0.000 | 0.010 | 0.200 | 12 |
| 1 | 11 | 64 | 3 | 0.300 | 0.001 | 0.000 | 0.200 | 18 |
| 2 | 6 | 64 | 3 | 0.100 | 0.001 | 0.000 | 0.200 | 10 |
| 3 | 1 | 64 | 2 | 0.100 | 0.001 | 0.000 | 0.200 | 11 |
| 4 | 10 | 64 | 3 | 0.300 | 0.000 | 0.010 | 0.300 | 36 |

_Tệp: `cau_hinh_chon_T2.csv`_

### PG khi che toàn bộ node ảnh ở lúc dự đoán (người có phim)

| nhom | AUC_day_du | AUC_khi_bo_anh |
|---|---|---|
| có ảnh | 0.754 | 0.662 |

_Tệp: `bo_anh.csv`_

*Chênh lệch = phần đóng góp của ảnh trong PG đã huấn luyện.*

### Chú ý trung bình CLS → 6 vùng (người có phim)

| RU | RM | RL | LU | LM | LL |
|---|---|---|---|---|---|
| 0.079 | 0.091 | 0.091 | 0.091 | 0.115 | 0.084 |

_Tệp: `chu_y_vung.csv`_

*Không đánh giá được H4 (chú ý có trúng vùng tổn thương không) vì dữ liệu không có cột `vungtonthuong`.*

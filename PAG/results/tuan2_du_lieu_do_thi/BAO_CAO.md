# Tuần 2 — Biến → node, cạnh và các mô hình đối chứng (bản 2)

Dữ liệu: dev2 = 6683 lượt khám (5 fold theo NGƯỜI sau khi gộp mã người trùng); test khoá đến cuối. Cấu trúc dự phòng (P0.6 kích hoạt vì khám theo vùng quá thưa): 6 node vùng phổi + 5 node nhóm lâm sàng + 1 node khám thực thể + 1 node bối cảnh (tỉnh) + 1 node CLS = 14 node; không có cạnh E6 (khám ↔ đúng vùng). Bỏ biến `nam` (năm công nhận BNN → rò nhãn); giữ `bnn` làm đặc trưng và có ablation PG-nobnn.

### Biến → node (mã hoá học trên toàn dev chỉ để mô tả; khi huấn luyện mã hoá fit trong từng fold)

| node | so_bien | so_cot_ma_hoa | pct_thieu_toan_bo_nhom | pct_thieu_tb_moi_bien | bien |
|---|---|---|---|---|---|
| Nhân khẩu & thể trạng | 10 | 15 | 0.0 | 0.0 | gioitinh, namsinh, cao, can, bmi, A6, hatd, hatt, mach, theluc |
| Phơi nhiễm/nghề | 21 | 96 | 0.0 | 19.0 | cviec, pxuong, tuoinghe, nampx, cv5nam, cviec1, tgian1, cviec2, tgian2, A7, A9a, A9b, A10, A11, A12, F4mu, F4ung, F4KhauTrang, F4gang, F4BHLD, F4Kinh |
| Hút thuốc | 2 | 4 | 0.0 | 28.6 | hutthuoc, slthuoc |
| Triệu chứng & tiền sử | 32 | 101 | 0.0 | 53.0 | tiensuhh, benhhh, tsnoi, tsngoai, ho, tdho, tsho, n1, khacdom, loaidom, tdkhacdo, khotho, mdkhotho, tdkhotho, daunguc, vitridau, daulan, tcdau, n2, tgdau, ytodau, n3, ytotang, chaymui, khan, khokhe, dhkhac, metmoi, sutcan, socansut, tgsut, bnn |
| Hô hấp ký | 3 | 3 | 0.0 | 0.0 | fvctt, fev1tt, fev1_fvc |
| Khám thực thể | 52 | 63 | 0.0 | 88.3 | longnguc, khoangls, rungthan, rungt1, rungt2, rungt3, rungt4, rungt5, rungt6, rungg1, rungg2, rungg3, rungg4, rungg5, rungg6, go, goduc1, goduc2, goduc3, goduc4, goduc5, goduc6, riraopn, ran, ranam, ranno, ranrit, ranngay, vtam1, vtam2, vtam3, vtam4, vtam5, vtam6, vtno1, vtno2, vtno3, vtno4, vtno5, vtno6, vtrit1, vtrit2, vtrit3, vtrit4, vtrit5, vtrit6, vtngay1, vtngay2, vtngay3, vtngay4, vtngay5, vtngay6 |
| Bối cảnh | 1 | 5 | 0.0 | 0.0 | tinh |

_Tệp: `bien_vao_node.csv`_

*`pct_thieu_toan_bo_nhom`: % lượt mà cả nhóm đều trống (node đó gần như vô tin). `pct_thieu_tb_moi_bien`: % trống trung bình từng biến.*

### Số láng giềng k của đồ thị bệnh nhân (kNN cosin) chọn theo validation nội bộ từng fold

| fold | mo_hinh | k_chon | val_noi_bo_k5 | val_noi_bo_k10 | val_noi_bo_k20 |
|---|---|---|---|---|---|
| 0 | L3b | 20 | 0.685 | 0.685 | 0.686 |
| 0 | B3b | 20 | 0.676 | 0.675 | 0.680 |
| 0 | GAT | 10 | 0.678 | 0.681 | 0.675 |
| 1 | L3b | 5 | 0.727 | 0.726 | 0.727 |
| 1 | B3b | 20 | 0.703 | 0.703 | 0.704 |
| 1 | GAT | 20 | 0.709 | 0.716 | 0.717 |
| 2 | L3b | 10 | 0.699 | 0.700 | 0.699 |
| 2 | B3b | 20 | 0.702 | 0.700 | 0.706 |
| 2 | GAT | 20 | 0.706 | 0.707 | 0.707 |
| 3 | L3b | 10 | 0.704 | 0.705 | 0.704 |
| 3 | B3b | 10 | 0.707 | 0.711 | 0.710 |
| 3 | GAT | 5 | 0.720 | 0.716 | 0.719 |
| 4 | L3b | 20 | 0.713 | 0.714 | 0.714 |
| 4 | B3b | 5 | 0.711 | 0.704 | 0.706 |
| 4 | GAT | 10 | 0.709 | 0.713 | 0.708 |

_Tệp: `k_chon.csv`_

### Mô hình đối chứng — dev OOF, xác suất đã hiệu chỉnh trong fold

| mo_hinh | nhom | n | duong | roc_auc | roc_auc_ci | pr_auc | brier | brier_nen | ece | do_doc_hc | chan_hc | nhay_tai_dh80 | nhay_tai_dh90 | auc_trong_lo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | toàn bộ | 6683 | 794 | 0.736 | [0.720–0.754] | 0.283 | 0.096 | 0.105 | 0.009 | 0.958 | -0.081 | 0.511 | 0.317 | nan |
| L0 | có ảnh | 1500 | 367 | 0.627 | [0.592–0.661] | 0.368 | 0.178 | 0.185 | 0.034 | 0.691 | -0.342 | 0.379 | 0.248 | 0.554 |
| L0 | không ảnh | 5183 | 427 | 0.708 | [0.684–0.731] | 0.177 | 0.072 | 0.076 | 0.010 | 1.037 | 0.081 | 0.436 | 0.260 | nan |
| L0 | không ảnh — Đồng Nai | 1205 | 110 | 0.621 | [0.565–0.673] | 0.159 | 0.081 | 0.083 | 0.017 | 0.856 | -0.308 | 0.336 | 0.227 | nan |
| L0 | không ảnh — tỉnh khác | 3978 | 317 | 0.731 | [0.702–0.760] | 0.184 | 0.070 | 0.073 | 0.012 | 1.070 | 0.149 | 0.483 | 0.284 | nan |
| L1 | toàn bộ | 1500 | 367 | 0.741 | [0.713–0.769] | 0.495 | 0.158 | 0.185 | 0.017 | 1.007 | -0.012 | 0.507 | 0.341 | nan |
| L1 | có ảnh | 1500 | 367 | 0.741 | [0.713–0.769] | 0.495 | 0.158 | 0.185 | 0.017 | 1.007 | -0.012 | 0.507 | 0.341 | 0.689 |
| Z1 | toàn bộ | 1500 | 367 | 0.690 | [0.660–0.721] | 0.422 | 0.169 | 0.185 | 0.038 | 0.799 | -0.216 | 0.450 | 0.300 | nan |
| Z1 | có ảnh | 1500 | 367 | 0.690 | [0.660–0.721] | 0.422 | 0.169 | 0.185 | 0.038 | 0.799 | -0.216 | 0.450 | 0.300 | 0.625 |
| L1-khám | toàn bộ | 6683 | 794 | 0.626 | [0.605–0.647] | 0.187 | 0.101 | 0.105 | 0.022 | 0.880 | -0.227 | 0.461 | 0.189 | nan |
| L1-khám | có ảnh | 1500 | 367 | 0.481 | [0.446–0.516] | 0.243 | 0.187 | 0.185 | 0.043 | 0.040 | -1.081 | 0.049 | 0.049 | 0.455 |
| L1-khám | không ảnh | 5183 | 427 | 0.485 | [0.456–0.515] | 0.080 | 0.076 | 0.076 | 0.018 | -0.133 | -2.734 | 0.019 | 0.019 | nan |
| L1-khám | không ảnh — Đồng Nai | 1205 | 110 | 0.500 | [0.442–0.567] | 0.095 | 0.083 | 0.083 | 0.019 | -0.274 | -2.964 | 0.209 | 0.036 | nan |
| L1-khám | không ảnh — tỉnh khác | 3978 | 317 | 0.481 | [0.445–0.512] | 0.076 | 0.074 | 0.073 | 0.018 | -0.124 | -2.748 | 0.013 | 0.013 | nan |
| L2 | toàn bộ | 6683 | 794 | 0.751 | [0.735–0.768] | 0.327 | 0.093 | 0.105 | 0.011 | 1.004 | -0.027 | 0.534 | 0.390 | nan |
| L2 | có ảnh | 1500 | 367 | 0.711 | [0.680–0.741] | 0.457 | 0.165 | 0.185 | 0.026 | 1.017 | -0.054 | 0.477 | 0.322 | 0.644 |
| L2 | không ảnh | 5183 | 427 | 0.707 | [0.683–0.729] | 0.169 | 0.073 | 0.076 | 0.009 | 1.071 | 0.145 | 0.422 | 0.239 | nan |
| L2 | không ảnh — Đồng Nai | 1205 | 110 | 0.612 | [0.554–0.663] | 0.144 | 0.082 | 0.083 | 0.021 | 0.790 | -0.481 | 0.309 | 0.164 | nan |
| L2 | không ảnh — tỉnh khác | 3978 | 317 | 0.733 | [0.704–0.761] | 0.177 | 0.070 | 0.073 | 0.012 | 1.120 | 0.255 | 0.457 | 0.268 | nan |
| X0 | toàn bộ | 6683 | 794 | 0.742 | [0.726–0.760] | 0.292 | 0.095 | 0.105 | 0.015 | 0.986 | -0.108 | 0.516 | 0.348 | nan |
| X0 | có ảnh | 1500 | 367 | 0.648 | [0.615–0.680] | 0.386 | 0.175 | 0.185 | 0.040 | 0.867 | -0.212 | 0.376 | 0.251 | 0.579 |
| X0 | không ảnh | 5183 | 427 | 0.709 | [0.685–0.732] | 0.179 | 0.072 | 0.076 | 0.007 | 1.024 | -0.037 | 0.464 | 0.283 | nan |
| X0 | không ảnh — Đồng Nai | 1205 | 110 | 0.602 | [0.544–0.661] | 0.155 | 0.082 | 0.083 | 0.019 | 0.618 | -0.902 | 0.327 | 0.191 | nan |
| X0 | không ảnh — tỉnh khác | 3978 | 317 | 0.739 | [0.712–0.763] | 0.190 | 0.069 | 0.073 | 0.012 | 1.117 | 0.160 | 0.521 | 0.312 | nan |
| X2 | toàn bộ | 6683 | 794 | 0.759 | [0.742–0.776] | 0.329 | 0.093 | 0.105 | 0.013 | 0.979 | -0.146 | 0.553 | 0.394 | nan |
| X2 | có ảnh | 1500 | 367 | 0.713 | [0.684–0.744] | 0.450 | 0.165 | 0.185 | 0.032 | 0.906 | -0.210 | 0.480 | 0.308 | 0.654 |
| X2 | không ảnh | 5183 | 427 | 0.714 | [0.689–0.738] | 0.185 | 0.072 | 0.076 | 0.010 | 1.046 | -0.005 | 0.468 | 0.276 | nan |
| X2 | không ảnh — Đồng Nai | 1205 | 110 | 0.607 | [0.547–0.665] | 0.151 | 0.082 | 0.083 | 0.026 | 0.632 | -0.881 | 0.327 | 0.173 | nan |
| X2 | không ảnh — tỉnh khác | 3978 | 317 | 0.743 | [0.717–0.768] | 0.197 | 0.069 | 0.073 | 0.007 | 1.141 | 0.196 | 0.517 | 0.315 | nan |
| L3b | toàn bộ | 6683 | 794 | 0.751 | [0.735–0.767] | 0.327 | 0.093 | 0.105 | 0.011 | 1.004 | -0.027 | 0.530 | 0.388 | nan |
| L3b | có ảnh | 1500 | 367 | 0.711 | [0.679–0.741] | 0.457 | 0.165 | 0.185 | 0.028 | 1.017 | -0.053 | 0.463 | 0.316 | 0.643 |
| L3b | không ảnh | 5183 | 427 | 0.707 | [0.684–0.729] | 0.168 | 0.073 | 0.076 | 0.012 | 1.068 | 0.141 | 0.426 | 0.244 | nan |
| L3b | không ảnh — Đồng Nai | 1205 | 110 | 0.613 | [0.556–0.665] | 0.144 | 0.082 | 0.083 | 0.019 | 0.783 | -0.496 | 0.300 | 0.164 | nan |
| L3b | không ảnh — tỉnh khác | 3978 | 317 | 0.733 | [0.704–0.761] | 0.177 | 0.070 | 0.073 | 0.013 | 1.120 | 0.257 | 0.473 | 0.268 | nan |
| B3b | toàn bộ | 6683 | 794 | 0.738 | [0.721–0.756] | 0.303 | 0.095 | 0.105 | 0.014 | 0.960 | -0.133 | 0.534 | 0.382 | nan |
| B3b | có ảnh | 1500 | 367 | 0.701 | [0.671–0.730] | 0.413 | 0.169 | 0.185 | 0.034 | 0.923 | -0.152 | 0.455 | 0.240 | 0.630 |
| B3b | không ảnh | 5183 | 427 | 0.680 | [0.654–0.706] | 0.160 | 0.073 | 0.076 | 0.014 | 0.962 | -0.138 | 0.407 | 0.255 | nan |
| B3b | không ảnh — Đồng Nai | 1205 | 110 | 0.588 | [0.533–0.638] | 0.126 | 0.083 | 0.083 | 0.029 | 0.598 | -0.947 | 0.309 | 0.191 | nan |
| B3b | không ảnh — tỉnh khác | 3978 | 317 | 0.705 | [0.674–0.733] | 0.171 | 0.070 | 0.073 | 0.011 | 1.030 | 0.022 | 0.448 | 0.281 | nan |
| GAT | toàn bộ | 6683 | 794 | 0.743 | [0.725–0.761] | 0.312 | 0.094 | 0.105 | 0.011 | 0.971 | -0.095 | 0.544 | 0.385 | nan |
| GAT | có ảnh | 1500 | 367 | 0.704 | [0.674–0.734] | 0.429 | 0.167 | 0.185 | 0.025 | 0.915 | -0.148 | 0.460 | 0.267 | 0.629 |
| GAT | không ảnh | 5183 | 427 | 0.691 | [0.665–0.718] | 0.167 | 0.073 | 0.076 | 0.008 | 1.022 | 0.019 | 0.426 | 0.272 | nan |
| GAT | không ảnh — Đồng Nai | 1205 | 110 | 0.603 | [0.549–0.658] | 0.136 | 0.082 | 0.083 | 0.023 | 0.664 | -0.761 | 0.318 | 0.191 | nan |
| GAT | không ảnh — tỉnh khác | 3978 | 317 | 0.716 | [0.688–0.747] | 0.180 | 0.070 | 0.073 | 0.010 | 1.098 | 0.182 | 0.461 | 0.300 | nan |

_Tệp: `chi_so_doi_chung.csv`_

*Giải thích chỉ số: `roc_auc` = ROC-AUC (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0.5 = đoán bừa, 1 = hoàn hảo), kèm KTC 95% (khoảng tin cậy, bootstrap theo người 1000 lần). `pr_auc` = PR-AUC (diện tích dưới đường precision–recall; mốc đoán bừa = tỉ lệ dương). `brier` = sai số bình phương trung bình của xác suất (càng thấp càng tốt), so với `brier_nen` = Brier của việc luôn đoán tỉ lệ nền. `ece` = sai lệch hiệu chỉnh trung bình (xác suất dự đoán lệch tần suất thật bao nhiêu, 10 khoảng). `do_doc_hc` = độ dốc hiệu chỉnh (1 là lý tưởng; <1 là dự đoán quá cực đoan), `chan_hc` = chặn hiệu chỉnh (0 là lý tưởng). `auc_trong_lo` = AUC tính riêng trong từng lô chụp rồi lấy trung bình theo cỡ lô (loại phần mô hình chỉ “nhận ra lô”). `nhay_tai_dh80/90` = độ nhạy (tỉ lệ ca bệnh bắt được) khi chọn ngưỡng sao cho độ đặc hiệu = 0,80 / 0,90 (chỉ 20% / 10% người âm bị báo nhầm) — chỉ số cho sàng lọc.*

*L0 = hồi quy logistic chỉ lâm sàng; L1 = LR chỉ ảnh toàn cục; Z1 = LR trên 6 vùng; L1-khám = LR chỉ biến khám thực thể; L2 = LR lâm sàng + ảnh; X0/X2 = XGBoost tương ứng L0/L2; L3b/B3b = LR/MLP + điểm rủi ro của láng giềng có ảnh; GAT = mạng chú ý đồ thị trên đồ thị bệnh nhân. Đồ thị bệnh nhân dựng trên toàn bộ 8030 lượt (chỉ đặc trưng, không nhãn test) — transductive như giao thức Fusion.*

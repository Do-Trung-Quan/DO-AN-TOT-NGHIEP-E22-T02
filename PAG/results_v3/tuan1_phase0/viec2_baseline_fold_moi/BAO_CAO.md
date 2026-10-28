# Việc (2) — Chạy lại toàn bộ baseline trên fold mới

Lời nhắn PGS 27/10: chạy lại L0, C (thay L1), L2, X0/X2, T0/T2, GAT/B3b với k chọn trên dev, trên fold mới (chuẩn duy nhất: `cache_v3/p05_folds.npz` — dev2/fold2/uid2). Dev = 6683 lượt khám (1500 có ảnh), 5 fold theo người; test khoá, không chấm.

Khác bản 2: đặc trưng ảnh là **C** = trung bình 6 vector vùng BioViL-T đóng băng trên ảnh V2 (che chữ mới + xoá ngoài phổi; chốt ở việc 1). C dùng làm ảnh trong L2/X2/B3b/GAT; T2 dùng 6 token vùng qua một phép chiếu PCA 128 chung cho cả 6 vùng.

Giao thức: LR (hồi quy logistic, cân bằng lớp) tinh chỉnh C (hệ số điều chuẩn, C nhỏ = phạt mạnh) bằng CV nội bộ 5 fold theo người trong phần train, tiêu chí ROC-AUC gộp, lưới 1e-6…10; XGBoost tham số cố định như bản 2, trung bình 3 seed; B3b/GAT chọn k ∈ {5, 10, 20} (số láng giềng đồ thị kNN) trên validation nội bộ (fold (k+1)%5 trong 4 fold train); T0/T2 dò 30 cấu hình trên cùng validation nội bộ, 3 seed. Hiệu chỉnh xác suất Platt trong fold, riêng cho lượt có ảnh / không ảnh.


### Nhóm toàn bộ — dev OOF, xác suất đã hiệu chỉnh

| mo_hinh | n | duong | roc_auc | roc_auc_ci | pr_auc | brier | brier_nen | ece | do_doc_hc | nhay_tai_dh80 |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 6683 | 794 | 0.736 | [0.719–0.753] | 0.286 | 0.096 | 0.105 | 0.009 | 0.928 | 0.505 |
| L2 | 6683 | 794 | 0.761 | [0.744–0.778] | 0.353 | 0.091 | 0.105 | 0.014 | 0.933 | 0.563 |
| X0 | 6683 | 794 | 0.742 | [0.726–0.760] | 0.292 | 0.095 | 0.105 | 0.015 | 0.986 | 0.516 |
| X2 | 6683 | 794 | 0.761 | [0.744–0.778] | 0.328 | 0.093 | 0.105 | 0.013 | 0.963 | 0.560 |
| B3b | 6683 | 794 | 0.749 | [0.732–0.766] | 0.326 | 0.094 | 0.105 | 0.012 | 0.879 | 0.543 |
| GAT | 6683 | 794 | 0.753 | [0.735–0.769] | 0.323 | 0.094 | 0.105 | 0.013 | 0.876 | 0.552 |
| T0 | 6683 | 794 | 0.707 | [0.687–0.726] | 0.270 | 0.097 | 0.105 | 0.008 | 0.897 | 0.481 |
| T2 | 6683 | 794 | 0.731 | [0.712–0.752] | 0.309 | 0.095 | 0.105 | 0.011 | 0.885 | 0.539 |

_Tệp: `chi_so_toan_bo.csv`_

### Nhóm có ảnh — dev OOF, xác suất đã hiệu chỉnh

| mo_hinh | n | duong | roc_auc | roc_auc_ci | pr_auc | brier | brier_nen | ece | do_doc_hc | nhay_tai_dh80 | auc_trong_lo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 1500 | 367 | 0.639 | [0.605–0.673] | 0.378 | 0.176 | 0.185 | 0.030 | 0.725 | 0.406 | 0.568 |
| C | 1500 | 367 | 0.752 | [0.724–0.778] | 0.479 | 0.159 | 0.185 | 0.028 | 0.945 | 0.518 | 0.729 |
| L2 | 1500 | 367 | 0.752 | [0.724–0.780] | 0.504 | 0.157 | 0.185 | 0.035 | 0.865 | 0.567 | 0.701 |
| X0 | 1500 | 367 | 0.648 | [0.615–0.680] | 0.386 | 0.175 | 0.185 | 0.040 | 0.867 | 0.376 | 0.579 |
| X2 | 1500 | 367 | 0.723 | [0.697–0.750] | 0.446 | 0.164 | 0.185 | 0.036 | 0.856 | 0.482 | 0.681 |
| B3b | 1500 | 367 | 0.728 | [0.701–0.755] | 0.457 | 0.164 | 0.185 | 0.039 | 0.807 | 0.458 | 0.696 |
| GAT | 1500 | 367 | 0.726 | [0.700–0.754] | 0.448 | 0.165 | 0.185 | 0.045 | 0.773 | 0.471 | 0.687 |
| T0 | 1500 | 367 | 0.656 | [0.622–0.688] | 0.372 | 0.175 | 0.185 | 0.030 | 0.809 | 0.387 | 0.600 |
| T2 | 1500 | 367 | 0.725 | [0.694–0.753] | 0.435 | 0.167 | 0.185 | 0.036 | 0.749 | 0.515 | 0.656 |

_Tệp: `chi_so_co_anh.csv`_

*Giải thích chỉ số: `roc_auc` = ROC-AUC (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0.5 = đoán bừa, 1 = hoàn hảo), kèm KTC 95% (khoảng tin cậy, bootstrap theo người 1000 lần). `pr_auc` = PR-AUC (diện tích dưới đường precision–recall; mốc đoán bừa = tỉ lệ dương). `brier` = sai số bình phương trung bình của xác suất (càng thấp càng tốt), so với `brier_nen` = Brier của việc luôn đoán tỉ lệ nền. `ece` = sai lệch hiệu chỉnh trung bình (xác suất dự đoán lệch tần suất thật bao nhiêu, 10 khoảng). `do_doc_hc` = độ dốc hiệu chỉnh (1 là lý tưởng; <1 là dự đoán quá cực đoan). `auc_trong_lo` = AUC tính riêng trong từng lô chụp rồi lấy trung bình theo cỡ lô (loại phần mô hình chỉ “nhận ra lô”). `nhay_tai_dh80` = độ nhạy (tỉ lệ ca bệnh bắt được) khi chọn ngưỡng sao cho độ đặc hiệu = 0,80 (20% người âm bị báo nhầm).*

### Nhóm không ảnh — dev OOF, xác suất đã hiệu chỉnh

| mo_hinh | n | duong | roc_auc | roc_auc_ci | pr_auc | brier | brier_nen | ece | do_doc_hc | nhay_tai_dh80 |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 5183 | 427 | 0.702 | [0.676–0.726] | 0.174 | 0.072 | 0.076 | 0.009 | 1.000 | 0.438 |
| L2 | 5183 | 427 | 0.703 | [0.678–0.725] | 0.171 | 0.073 | 0.076 | 0.013 | 0.989 | 0.440 |
| X0 | 5183 | 427 | 0.709 | [0.685–0.732] | 0.179 | 0.072 | 0.076 | 0.007 | 1.024 | 0.464 |
| X2 | 5183 | 427 | 0.710 | [0.685–0.733] | 0.185 | 0.072 | 0.076 | 0.010 | 1.026 | 0.443 |
| B3b | 5183 | 427 | 0.680 | [0.655–0.705] | 0.160 | 0.073 | 0.076 | 0.017 | 0.826 | 0.400 |
| GAT | 5183 | 427 | 0.690 | [0.664–0.716] | 0.160 | 0.073 | 0.076 | 0.011 | 0.899 | 0.440 |
| T0 | 5183 | 427 | 0.637 | [0.609–0.665] | 0.125 | 0.075 | 0.076 | 0.011 | 0.788 | 0.337 |
| T2 | 5183 | 427 | 0.662 | [0.633–0.688] | 0.145 | 0.074 | 0.076 | 0.006 | 0.889 | 0.396 |

_Tệp: `chi_so_khong_anh.csv`_

### So sánh cặp trên nhóm có ảnh (chênh ROC-AUC gộp và AUC trong lô, KTC 95% bootstrap theo người)

| so_sanh | chenh_auc_gop | ci_gop | chenh_auc_trong_lo | ci_trong_lo |
|---|---|---|---|---|
| L2 − C | 0.001 | [-0.019; +0.019] | -0.028 | [-0.051; -0.007] |
| X2 − C | -0.028 | [-0.050; -0.005] | -0.048 | [-0.077; -0.018] |
| B3b − C | -0.024 | [-0.038; -0.009] | -0.032 | [-0.054; -0.012] |
| GAT − C | -0.025 | [-0.040; -0.010] | -0.042 | [-0.061; -0.022] |
| T2 − C | -0.026 | [-0.053; -0.000] | -0.072 | [-0.109; -0.032] |
| L2 − L0 | 0.113 | [+0.087; +0.138] | 0.132 | [+0.098; +0.166] |

_Tệp: `so_sanh_cap_co_anh.csv`_

*Âm = mô hình đứng trước kém hơn. AUC trong lô loại phần mô hình chỉ “nhận ra lô chụp”.*

### So với bản 2 — nhóm có ảnh (bản 2: ảnh cũ, cùng fold; L1 bản 2 ↔ C bản 3)

| mo_hinh | ban2 | auc_co_anh_ban2 | auc_co_anh_ban3 | trong_lo_ban2 | trong_lo_ban3 |
|---|---|---|---|---|---|
| L0 | L0 | 0.627 | 0.639 | 0.554 | 0.568 |
| C | L1 | 0.754 | 0.752 | 0.710 | 0.729 |
| L2 | L2 | 0.725 | 0.752 | 0.659 | 0.701 |
| X0 | X0 | 0.648 | 0.648 | 0.579 | 0.579 |
| X2 | X2 | 0.726 | 0.723 | 0.666 | 0.681 |
| B3b | B3b | 0.702 | 0.728 | 0.638 | 0.696 |
| GAT | GAT | 0.716 | 0.726 | 0.652 | 0.687 |
| T0 | T0 | 0.652 | 0.656 | 0.589 | 0.600 |
| T2 | T2 | 0.748 | 0.725 | 0.689 | 0.656 |

_Tệp: `so_voi_ban_2.csv`_

### Tham số chọn theo fold (C của LR; k của B3b/GAT)

| fold | B3b | C | GAT | L0 | L2 |
|---|---|---|---|---|---|
| 0 | 5.0000 | 0.0100 | 20.0000 | 0.0032 | 0.0032 |
| 1 | 5.0000 | 0.0316 | 5.0000 | 0.0032 | 0.0032 |
| 2 | 10.0000 | 0.0316 | 10.0000 | 0.0100 | 0.0032 |
| 3 | 10.0000 | 0.0316 | 10.0000 | 0.0032 | 0.0032 |
| 4 | 20.0000 | 0.1000 | 10.0000 | 0.0100 | 0.0100 |

_Tệp: `tham_so_chon.csv`_

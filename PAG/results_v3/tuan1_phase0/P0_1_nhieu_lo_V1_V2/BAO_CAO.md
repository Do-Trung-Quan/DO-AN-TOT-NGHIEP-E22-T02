# P0.1 (bản 3) — Nhiễu theo lô chụp trên ảnh đã xử lý V1, V2

Tập: dev gốc (chưa ghép người), có ảnh: 1571 lượt, 393 dương, 1493 người, 18 lô. Mô hình L1 = hồi quy logistic (chuẩn hóa, C=0,1, cân lớp) chỉ trên vector ảnh, 5 fold theo người. KTC = khoảng tin cậy 95% bootstrap theo người (1000 lần). Tập test không mở. zone_missing (thay 6 vùng bằng vector toàn ảnh): 0 ảnh.

### Chỉ số 1 — Probe đoán lô (18 lớp) từ embedding

| dac_trung | probe_doan_lo | probe_doan_bua |
|---|---|---|
| Cũ — BioViL-T đóng băng, toàn ảnh (CenterCrop, chưa che chữ) | 0.450 | 0.108 |
| V1 — BioViL-T đóng băng, toàn ảnh | 0.424 | 0.108 |
| V1 — BioViL-T đóng băng, 6 vùng | 0.384 | 0.108 |
| V2 — BioViL-T đóng băng, toàn ảnh | 0.375 | 0.108 |
| V2 — BioViL-T đóng băng, 6 vùng | 0.370 | 0.108 |
| V1 — CheXFound, toàn ảnh | 0.577 | 0.108 |
| V1 — CheXFound, 6 vùng | 0.506 | 0.108 |
| V2 — CheXFound, toàn ảnh | 0.513 | 0.108 |
| V2 — CheXFound, 6 vùng | 0.466 | 0.108 |

_Tệp: `1_probe_doan_lo.csv`_

*Độ chính xác của hồi quy logistic đoán ảnh thuộc lô nào. `probe_doan_bua` = luôn đoán lô đông nhất. Càng cao hơn mức đoán bừa thì embedding càng mang dấu vết lô (máy chụp, thời điểm, chữ in…).*

### Chỉ số 2–4 — AUC trong lô, Δ(lô + ảnh so với lô), LOBO

| dac_trung | n | duong | L1_auc | L1_ci | auc_trong_lo | auc_trong_lo_ci | auc_chi_lo | delta | delta_ci | lobo_auc | lobo_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cũ — BioViL-T đóng băng, toàn ảnh (CenterCrop, chưa che chữ) | 1571 | 393 | 0.759 | [0.732; 0.785] | 0.716 | [0.682; 0.747] | 0.705 | 0.078 | [+0.052; +0.106] | 0.764 | [0.737; 0.790] |
| V1 — BioViL-T đóng băng, toàn ảnh | 1571 | 393 | 0.744 | [0.715; 0.770] | 0.707 | [0.671; 0.741] | 0.705 | 0.069 | [+0.043; +0.095] | 0.747 | [0.720; 0.773] |
| V1 — BioViL-T đóng băng, 6 vùng | 1571 | 393 | 0.732 | [0.704; 0.760] | 0.698 | [0.662; 0.733] | 0.705 | 0.058 | [+0.030; +0.090] | 0.724 | [0.696; 0.751] |
| V2 — BioViL-T đóng băng, toàn ảnh | 1571 | 393 | 0.770 | [0.743; 0.795] | 0.748 | [0.717; 0.779] | 0.705 | 0.097 | [+0.073; +0.124] | 0.757 | [0.731; 0.782] |
| V2 — BioViL-T đóng băng, 6 vùng | 1571 | 393 | 0.715 | [0.687; 0.745] | 0.685 | [0.649; 0.719] | 0.705 | 0.038 | [+0.008; +0.068] | 0.704 | [0.675; 0.732] |
| V1 — CheXFound, toàn ảnh | 1571 | 393 | 0.736 | [0.708; 0.763] | 0.697 | [0.660; 0.729] | 0.705 | 0.056 | [+0.026; +0.089] | 0.702 | [0.673; 0.731] |
| V1 — CheXFound, 6 vùng | 1571 | 393 | 0.730 | [0.702; 0.758] | 0.696 | [0.657; 0.732] | 0.705 | 0.038 | [+0.006; +0.074] | 0.729 | [0.702; 0.756] |
| V2 — CheXFound, toàn ảnh | 1571 | 393 | 0.761 | [0.734; 0.787] | 0.725 | [0.688; 0.759] | 0.705 | 0.087 | [+0.057; +0.119] | 0.728 | [0.698; 0.756] |
| V2 — CheXFound, 6 vùng | 1571 | 393 | 0.742 | [0.714; 0.769] | 0.717 | [0.681; 0.753] | 0.705 | 0.057 | [+0.025; +0.091] | 0.729 | [0.701; 0.757] |

_Tệp: `2_ket_qua_4_chi_so.csv`_

*`L1_auc`: ROC-AUC gộp (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0,5 = đoán bừa). `auc_trong_lo`: AUC tính riêng trong từng lô có đủ hai nhãn, trung bình theo cỡ lô — loại phần mô hình chỉ "nhận ra lô". `auc_chi_lo`: AUC khi chỉ biết lô (one-hot). `delta`: AUC(lô + ảnh) − AUC(chỉ lô); KTC nằm trọn trên 0 nghĩa là ảnh thêm thông tin ngoài lô. `lobo_auc`: leave-one-batch-out — 18 lần, mỗi lần huấn luyện trên 17 lô và dự đoán lô còn lại; AUC tính trên dự đoán gộp. Mô hình không bao giờ thấy lô nó dự đoán.*

### So với kết quả trên đặc trưng cũ

| nguon | lobo_auc | delta |
|---|---|---|
| Máy khác (PGS trích trong PG_update) | 0.742 | +0.078 [+0.052; +0.106] |
| Máy này — đặc trưng cũ, cùng giao thức bản 3 (hàng đầu bảng trên) | 0.764 | +0.078 [+0.052; +0.106] |

_Tệp: `3_so_voi_cu.csv`_

*Cả hai dùng dev chưa ghép người và fold gốc.*

### Phân tích độ nhạy — loại 55 người có lượt bnn=Có nhưng ketqua=2 (P0.4; không sửa nhãn)

| dac_trung | n | duong | L1_auc | auc_trong_lo | auc_trong_lo_ci | delta | delta_ci | lobo_auc | lobo_ci |
|---|---|---|---|---|---|---|---|---|---|
| Cũ — BioViL-T đóng băng, toàn ảnh (CenterCrop, chưa che chữ) | 1523 | 393 | 0.764 | 0.722 | [0.687; 0.751] | 0.087 | [+0.058; +0.113] | 0.770 | [0.742; 0.795] |
| V1 — BioViL-T đóng băng, toàn ảnh | 1523 | 393 | 0.748 | 0.714 | [0.679; 0.743] | 0.078 | [+0.050; +0.103] | 0.752 | [0.725; 0.776] |
| V1 — BioViL-T đóng băng, 6 vùng | 1523 | 393 | 0.734 | 0.702 | [0.664; 0.739] | 0.065 | [+0.032; +0.095] | 0.726 | [0.695; 0.754] |
| V2 — BioViL-T đóng băng, toàn ảnh | 1523 | 393 | 0.774 | 0.753 | [0.719; 0.783] | 0.103 | [+0.076; +0.127] | 0.761 | [0.733; 0.787] |
| V2 — BioViL-T đóng băng, 6 vùng | 1523 | 393 | 0.724 | 0.696 | [0.661; 0.729] | 0.051 | [+0.021; +0.081] | 0.711 | [0.682; 0.740] |
| V1 — CheXFound, toàn ảnh | 1523 | 393 | 0.738 | 0.704 | [0.667; 0.738] | 0.062 | [+0.031; +0.093] | 0.710 | [0.680; 0.737] |
| V1 — CheXFound, 6 vùng | 1523 | 393 | 0.732 | 0.700 | [0.660; 0.737] | 0.045 | [+0.009; +0.080] | 0.735 | [0.705; 0.761] |
| V2 — CheXFound, toàn ảnh | 1523 | 393 | 0.760 | 0.730 | [0.696; 0.763] | 0.093 | [+0.061; +0.123] | 0.729 | [0.699; 0.758] |
| V2 — CheXFound, 6 vùng | 1523 | 393 | 0.744 | 0.719 | [0.678; 0.755] | 0.064 | [+0.026; +0.099] | 0.735 | [0.705; 0.763] |

_Tệp: `4_do_nhay_loai_55_nguoi.csv`_

### Quyết định theo luật PG_update §2 (AUC trong lô của L1: ≥0,70 và KTC Δ trọn trên 0 → tiếp tục; 0,60–0,70 → báo PGS; <0,60 → dừng)

| dac_trung | auc_trong_lo | can_duoi_KTC_delta | quyet_dinh |
|---|---|---|---|
| Cũ — BioViL-T đóng băng, toàn ảnh (CenterCrop, chưa che chữ) | 0.716 | 0.052 | TIẾP TỤC — AUC trong lô là chỉ số chính |
| V1 — BioViL-T đóng băng, toàn ảnh | 0.707 | 0.043 | TIẾP TỤC — AUC trong lô là chỉ số chính |
| V1 — BioViL-T đóng băng, 6 vùng | 0.698 | 0.030 | BÁO CÁO PGS trước khi làm tiếp |
| V2 — BioViL-T đóng băng, toàn ảnh | 0.748 | 0.073 | TIẾP TỤC — AUC trong lô là chỉ số chính |
| V2 — BioViL-T đóng băng, 6 vùng | 0.685 | 0.008 | BÁO CÁO PGS trước khi làm tiếp |
| V1 — CheXFound, toàn ảnh | 0.697 | 0.026 | BÁO CÁO PGS trước khi làm tiếp |
| V1 — CheXFound, 6 vùng | 0.696 | 0.006 | BÁO CÁO PGS trước khi làm tiếp |
| V2 — CheXFound, toàn ảnh | 0.725 | 0.057 | TIẾP TỤC — AUC trong lô là chỉ số chính |
| V2 — CheXFound, 6 vùng | 0.717 | 0.025 | TIẾP TỤC — AUC trong lô là chỉ số chính |

_Tệp: `5_quyet_dinh.csv`_

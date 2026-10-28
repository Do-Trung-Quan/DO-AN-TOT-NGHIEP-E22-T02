# P0.1 bước 5 — Nhiễu lô chụp SAU che chữ + cắt theo phổi

### Đoán lô (18 lớp) từ đặc trưng ảnh đã che chữ + cắt

| dac_trung | do_chinh_xac_doan_lo | muc_doan_bua |
|---|---|---|
| BioViL-T đóng băng — 6 vùng | 0.408 | 0.111 |
| BioViL-T đóng băng — toàn ảnh | 0.420 | 0.111 |
| BioViL-T control — 6 vùng | 0.372 | 0.111 |
| BioViL-T control — toàn ảnh | 0.406 | 0.111 |
| ResNet50-xrv — 6 vùng | 0.404 | 0.111 |
| ResNet50-xrv — toàn ảnh | 0.404 | 0.111 |
| CheXFound — 6 vùng | 0.494 | 0.111 |
| CheXFound — toàn ảnh | 0.555 | 0.111 |

_Tệp: `doan_lo_tu_anh.csv`_

*Độ chính xác đoán lô càng gần `muc_doan_bua` (luôn đoán lô đông nhất) thì đặc trưng càng ít mang dấu vết của lô.*

### AUC gộp, AUC trong lô, Δ do ảnh so với chỉ biết lô, và fold để nguyên lô

| dac_trung | L1_auc | L1_auc_ci | auc_trong_lo | auc_trong_lo_ci | auc_chi_lo | delta_anh | delta_ci | auc_de_nguyen_lo |
|---|---|---|---|---|---|---|---|---|
| BioViL-T đóng băng — 6 vùng | 0.738 | [0.710–0.768] | 0.695 | [0.658–0.734] | 0.707 | 0.055 | [0.024–0.087] | 0.740 |
| BioViL-T đóng băng — toàn ảnh | 0.767 | [0.740–0.793] | 0.729 | [0.696–0.763] | 0.707 | 0.086 | [0.058–0.114] | 0.761 |
| BioViL-T control — 6 vùng | 0.721 | [0.693–0.749] | 0.694 | [0.661–0.725] | 0.707 | 0.049 | [0.019–0.081] | 0.688 |
| BioViL-T control — toàn ảnh | 0.759 | [0.733–0.785] | 0.725 | [0.692–0.754] | 0.707 | 0.085 | [0.060–0.110] | 0.742 |
| ResNet50-xrv — 6 vùng | 0.700 | [0.671–0.731] | 0.676 | [0.639–0.710] | 0.707 | 0.023 | [-0.011–0.059] | 0.685 |
| ResNet50-xrv — toàn ảnh | 0.753 | [0.725–0.779] | 0.732 | [0.696–0.765] | 0.707 | 0.077 | [0.046–0.108] | 0.737 |
| CheXFound — 6 vùng | 0.768 | [0.742–0.792] | 0.726 | [0.682–0.766] | 0.707 | 0.074 | [0.041–0.106] | 0.753 |
| CheXFound — toàn ảnh | 0.746 | [0.719–0.775] | 0.710 | [0.676–0.746] | 0.707 | 0.070 | [0.038–0.102] | 0.732 |

_Tệp: `ket_qua.csv`_

*`L1_auc`: ROC-AUC (xác suất mô hình xếp một ca dương cao hơn một ca âm; 0.5 = đoán bừa) của hồi quy logistic chỉ trên ảnh. `auc_trong_lo`: AUC tính riêng trong từng lô chụp rồi lấy trung bình theo cỡ lô; nó loại phần mô hình chỉ “nhận ra lô”. `delta_anh`: AUC(lô + ảnh) − AUC(chỉ lô); dương nghĩa là ảnh thêm thông tin ngoài lô. `auc_de_nguyen_lo`: mỗi lô nằm trọn trong một fold; mô hình phải dự đoán cho lô nó chưa thấy. CI = khoảng tin cậy 95% bootstrap theo người.*

### Để so sánh: kết quả bước 1–4 (đặc trưng cũ, chưa che chữ)

| dac_trung_truoc_che (toàn ảnh, CenterCrop) | auc_trong_lo_truoc | delta_truoc |
|---|---|---|
| BioViL-T frozen (toàn ảnh) | 0.722 | 0.077 |
| BioViL-T control (toàn ảnh) | 0.729 | 0.088 |
| ResNet50-xrv (toàn ảnh, bản 1) | 0.701 | 0.061 |

_Tệp: `truoc_che_de_so_sanh.csv`_

### Quyết định cuối P0.1 theo luật §2 (≥0.70 và Δ>0 → đi tiếp; 0.60–0.70 → có điều kiện; <0.60 → dừng)

| dac_trung | auc_trong_lo | delta_ci_duoi | quyet_dinh |
|---|---|---|---|
| BioViL-T đóng băng — 6 vùng | 0.695 | 0.024 | ĐI TIẾP CÓ ĐIỀU KIỆN — che chữ + cắt bắt buộc |
| BioViL-T đóng băng — toàn ảnh | 0.729 | 0.058 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| BioViL-T control — 6 vùng | 0.694 | 0.019 | ĐI TIẾP CÓ ĐIỀU KIỆN — che chữ + cắt bắt buộc |
| BioViL-T control — toàn ảnh | 0.725 | 0.060 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| ResNet50-xrv — 6 vùng | 0.676 | -0.011 | ĐI TIẾP CÓ ĐIỀU KIỆN — che chữ + cắt bắt buộc |
| ResNet50-xrv — toàn ảnh | 0.732 | 0.046 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| CheXFound — 6 vùng | 0.726 | 0.041 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| CheXFound — toàn ảnh | 0.710 | 0.038 | ĐI TIẾP — AUC trong lô thành chỉ số chính |

_Tệp: `quyet_dinh.csv`_

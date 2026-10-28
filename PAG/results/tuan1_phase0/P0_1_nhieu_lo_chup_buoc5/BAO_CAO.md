# P0.1 bước 5 — Nhiễu lô chụp SAU che chữ + cắt theo phổi

### Đoán lô (18 lớp) từ đặc trưng ảnh đã che chữ + cắt

| dac_trung | do_chinh_xac_doan_lo | muc_doan_bua |
|---|---|---|
| BioViL-T đóng băng — 6 vùng | 0.407 | 0.111 |
| BioViL-T đóng băng — toàn ảnh | 0.439 | 0.111 |
| BioViL-T control — 6 vùng | 0.392 | 0.111 |
| BioViL-T control — toàn ảnh | 0.399 | 0.111 |
| ResNet50-xrv — 6 vùng | 0.402 | 0.111 |
| ResNet50-xrv — toàn ảnh | 0.431 | 0.111 |

_Tệp: `doan_lo_tu_anh.csv`_

*Độ chính xác đoán lô càng gần `muc_doan_bua` (luôn đoán lô đông nhất) thì đặc trưng càng ít mang dấu vết của lô.*

### AUC gộp, AUC trong lô, Δ do ảnh so với chỉ biết lô, và fold để nguyên lô

| dac_trung | L1_auc | L1_auc_ci | auc_trong_lo | auc_trong_lo_ci | auc_chi_lo | delta_anh | delta_ci | auc_de_nguyen_lo |
|---|---|---|---|---|---|---|---|---|
| BioViL-T đóng băng — 6 vùng | 0.730 | [0.703–0.757] | 0.680 | [0.645–0.720] | 0.707 | 0.048 | [0.017–0.079] | 0.725 |
| BioViL-T đóng băng — toàn ảnh | 0.750 | [0.725–0.779] | 0.706 | [0.671–0.745] | 0.707 | 0.071 | [0.044–0.099] | 0.749 |
| BioViL-T control — 6 vùng | 0.723 | [0.698–0.752] | 0.669 | [0.634–0.709] | 0.707 | 0.046 | [0.020–0.078] | 0.697 |
| BioViL-T control — toàn ảnh | 0.760 | [0.735–0.787] | 0.725 | [0.691–0.757] | 0.707 | 0.091 | [0.067–0.115] | 0.739 |
| ResNet50-xrv — 6 vùng | 0.693 | [0.663–0.723] | 0.681 | [0.645–0.715] | 0.707 | 0.026 | [-0.010–0.062] | 0.690 |
| ResNet50-xrv — toàn ảnh | 0.750 | [0.723–0.778] | 0.726 | [0.689–0.760] | 0.707 | 0.078 | [0.047–0.110] | 0.739 |

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
| BioViL-T đóng băng — 6 vùng | 0.680 | 0.017 | ĐI TIẾP CÓ ĐIỀU KIỆN — che chữ + cắt bắt buộc |
| BioViL-T đóng băng — toàn ảnh | 0.706 | 0.044 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| BioViL-T control — 6 vùng | 0.669 | 0.020 | ĐI TIẾP CÓ ĐIỀU KIỆN — che chữ + cắt bắt buộc |
| BioViL-T control — toàn ảnh | 0.725 | 0.067 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| ResNet50-xrv — 6 vùng | 0.681 | -0.010 | ĐI TIẾP CÓ ĐIỀU KIỆN — che chữ + cắt bắt buộc |
| ResNet50-xrv — toàn ảnh | 0.726 | 0.047 | ĐI TIẾP — AUC trong lô thành chỉ số chính |

_Tệp: `quyet_dinh.csv`_

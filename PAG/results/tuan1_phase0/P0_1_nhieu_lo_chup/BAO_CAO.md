# P0.1 — Kiểm tra nhiễu theo lô chụp (bước 1–4, đặc trưng cũ chưa che chữ)

### Phân bố nhãn theo 18 lô (dev, có ảnh; mã lô ẩn danh)

| lo | ketqua=2 | ketqua=1 |
|---|---|---|
| lô 01 | 143 | 23 |
| lô 02 | 62 | 12 |
| lô 03 | 76 | 57 |
| lô 04 | 24 | 39 |
| lô 05 | 114 | 14 |
| lô 06 | 23 | 9 |
| lô 07 | 108 | 28 |
| lô 08 | 116 | 28 |
| lô 09 | 65 | 40 |
| lô 10 | 58 | 40 |
| lô 11 | 14 | 9 |
| lô 12 | 65 | 19 |
| lô 13 | 46 | 21 |
| lô 14 | 0 | 4 |
| lô 15 | 57 | 2 |
| lô 16 | 131 | 7 |
| lô 17 | 31 | 10 |
| lô 18 | 0 | 5 |

_Tệp: `phan_bo_nhan_theo_lo.csv`_

### Bước 1 — Đoán lô (18 lớp) từ embedding ảnh

| dac_trung | so_lo | do_chinh_xac | muc_doan_bua |
|---|---|---|---|
| BioViL-T frozen (toàn ảnh) | 18 | 0.478 | 0.111 |
| BioViL-T control (toàn ảnh) | 18 | 0.441 | 0.111 |
| ResNet50-xrv (toàn ảnh, bản 1) | 18 | 0.484 | 0.111 |

_Tệp: `buoc1_doan_lo_tu_anh.csv`_

*`muc_doan_bua` là độ chính xác khi luôn đoán lô đông nhất. Độ chính xác cao hơn nhiều mức này nghĩa là embedding mang dấu vết của lô.*

### Bước 2–4 — AUC gộp, AUC trong lô, Δ do ảnh so với chỉ lô, và fold để nguyên lô

| dac_trung | L1_auc | L1_auc_ci | auc_trong_lo | auc_trong_lo_ci | auc_chi_lo | auc_lo_cong_anh | delta_anh | delta_ci | auc_de_nguyen_lo | auc_trong_lo_de_nguyen_lo |
|---|---|---|---|---|---|---|---|---|---|---|
| BioViL-T frozen (toàn ảnh) | 0.763 | [0.737–0.788] | 0.722 | [0.686–0.753] | 0.707 | 0.784 | 0.077 | [0.050–0.102] | 0.761 | 0.738 |
| BioViL-T control (toàn ảnh) | 0.768 | [0.742–0.794] | 0.729 | [0.696–0.758] | 0.707 | 0.796 | 0.088 | [0.064–0.113] | 0.753 | 0.730 |
| ResNet50-xrv (toàn ảnh, bản 1) | 0.751 | [0.725–0.778] | 0.701 | [0.662–0.735] | 0.707 | 0.768 | 0.061 | [0.032–0.089] | 0.744 | 0.706 |

_Tệp: `buoc2_4_ket_qua.csv`_

### Quyết định tạm (trước che chữ) theo luật §2

| dac_trung | auc_trong_lo | delta_ci_duoi | quyet_dinh |
|---|---|---|---|
| BioViL-T frozen (toàn ảnh) | 0.722 | 0.050 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| BioViL-T control (toàn ảnh) | 0.729 | 0.064 | ĐI TIẾP — AUC trong lô thành chỉ số chính |
| ResNet50-xrv (toàn ảnh, bản 1) | 0.701 | 0.032 | ĐI TIẾP — AUC trong lô thành chỉ số chính |

_Tệp: `quyet_dinh_buoc_1_4.csv`_

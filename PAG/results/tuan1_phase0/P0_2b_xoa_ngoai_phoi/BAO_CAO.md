# P0.2b — Xoá mọi thứ ngoài vùng phổi: so đặc trưng cũ và mới

Lý do: kiểm tra bằng mắt 50 ảnh ngẫu nhiên sau bước che chữ thấy khoảng 18/50 ảnh còn sót chữ hoặc che sai chỗ (đường viền phổi thì đúng cả 50). Sửa: sau khi phân đoạn lại, tô đen (=0) mọi pixel nằm ngoài bao lồi của hai phổi nới rộng 2% cạnh ảnh; chữ ngoài phổi vì thế bị xoá hết bất kể bộ lọc chữ có bắt được hay không. Các bước khác giữ nguyên (cắt khung bao phổi + lề 6%, 6 vùng ILO, cùng 3 bộ mã hoá). Luật quyết định đặt trước khi chạy: nếu với bộ mã hoá chính (a, 6 vùng) |hiệu AUC| < 0,01, |hiệu AUC trong lô| < 0,01 và cosin trung vị mọi vùng > 0,95 thì giữ kết quả cũ (2a); ngược lại chạy lại (2b).

### Hình học: phần ảnh được giữ, khung cắt, mask vùng

| muc | gia_tri |
|---|---|
| Phần ảnh 1024 được giữ (trong bao lồi phổi nới 2%) — trung vị [p5–p95] | 40% [30–51%] |
| Khung cắt chiếm (% ảnh) cũ → mới, trung vị | 59% → 59% |
| Mask 6 vùng giống hệt cũ (số ảnh) | 1835/1835 |
| Pixel mask vùng trùng cũ (trung vị %) | 100.0% |

_Tệp: `hinh_hoc.csv`_

### Độ giống đặc trưng cũ ↔ mới cho cùng một ảnh (cosin)

| bo_ma_hoa | phan | trung_vi | p5 | nho_nhat |
|---|---|---|---|---|
| BioViL-T đóng băng (a) | RU | 0.964 | 0.911 | 0.781 |
| BioViL-T đóng băng (a) | RM | 0.992 | 0.979 | 0.930 |
| BioViL-T đóng băng (a) | RL | 0.991 | 0.972 | 0.898 |
| BioViL-T đóng băng (a) | LU | 0.971 | 0.932 | 0.740 |
| BioViL-T đóng băng (a) | LM | 0.995 | 0.987 | 0.959 |
| BioViL-T đóng băng (a) | LL | 0.990 | 0.950 | 0.730 |
| BioViL-T đóng băng (a) | toàn ảnh | 0.992 | 0.986 | 0.969 |
| ResNet50-xrv (b) | RU | 0.907 | 0.802 | 0.526 |
| ResNet50-xrv (b) | RM | 0.959 | 0.890 | 0.674 |
| ResNet50-xrv (b) | RL | 0.959 | 0.862 | 0.689 |
| ResNet50-xrv (b) | LU | 0.941 | 0.870 | 0.653 |
| ResNet50-xrv (b) | LM | 0.991 | 0.974 | 0.866 |
| ResNet50-xrv (b) | LL | 0.989 | 0.964 | 0.891 |
| ResNet50-xrv (b) | toàn ảnh | 0.950 | 0.914 | 0.839 |
| BioViL-T control (c) | RU | 0.992 | 0.976 | 0.871 |
| BioViL-T control (c) | RM | 0.998 | 0.988 | 0.826 |
| BioViL-T control (c) | RL | 0.992 | 0.928 | 0.658 |
| BioViL-T control (c) | LU | 0.991 | 0.969 | 0.737 |
| BioViL-T control (c) | LM | 0.998 | 0.989 | 0.941 |
| BioViL-T control (c) | LL | 0.996 | 0.983 | 0.736 |
| BioViL-T control (c) | toàn ảnh | 0.995 | 0.979 | 0.933 |

_Tệp: `cosin_cu_moi.csv`_

*Cosin = 1 nghĩa là vector đặc trưng không đổi; tính trên từng ảnh, báo trung vị, phân vị 5% và nhỏ nhất.*

### Cũ vs mới: đoán lô, ROC-AUC, AUC trong lô, Δ do ảnh (dev, chỉ lượt có ảnh, hồi quy logistic OOF)

| bo_ma_hoa | dac_trung | doan_lo_cu | auc_cu | auc_trong_lo_cu | delta_cu | doan_lo_moi | auc_moi | auc_trong_lo_moi | delta_moi | hieu_auc | hieu_auc_ci | hieu_auc_trong_lo | hieu_auc_trong_lo_ci | tuong_quan_diem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BioViL-T đóng băng (a) | 6 vùng | 0.407 | 0.730 | 0.680 | 0.048 | 0.408 | 0.738 | 0.696 | 0.055 | 0.008 | [-0.010–0.024] | 0.015 | [-0.014–0.042] | 0.841 |
| BioViL-T đóng băng (a) | toàn ảnh | 0.439 | 0.750 | 0.706 | 0.071 | 0.420 | 0.767 | 0.729 | 0.086 | 0.016 | [0.003–0.030] | 0.024 | [0.001–0.044] | 0.916 |
| ResNet50-xrv (b) | 6 vùng | 0.402 | 0.693 | 0.681 | 0.026 | 0.404 | 0.701 | 0.676 | 0.022 | 0.007 | [-0.015–0.031] | -0.006 | [-0.032–0.023] | 0.703 |
| ResNet50-xrv (b) | toàn ảnh | 0.431 | 0.750 | 0.726 | 0.078 | 0.404 | 0.753 | 0.732 | 0.077 | 0.003 | [-0.017–0.022] | 0.006 | [-0.021–0.033] | 0.829 |
| BioViL-T control (c) | 6 vùng | 0.392 | 0.723 | 0.669 | 0.046 | 0.372 | 0.721 | 0.694 | 0.049 | -0.002 | [-0.025–0.021] | 0.025 | [-0.014–0.062] | 0.761 |
| BioViL-T control (c) | toàn ảnh | 0.399 | 0.760 | 0.725 | 0.091 | 0.406 | 0.759 | 0.725 | 0.085 | -0.001 | [-0.018–0.015] | -0.000 | [-0.024–0.020] | 0.890 |

_Tệp: `so_sanh_auc.csv`_

*`doan_lo`: độ chính xác đoán đúng lô chụp (18 lớp) từ đặc trưng ảnh — thấp hơn là ít dấu vết lô hơn. `auc`: ROC-AUC (xác suất xếp một ca dương cao hơn một ca âm; 0,5 = đoán bừa). `auc_trong_lo`: AUC tính trong từng lô rồi lấy trung bình theo cỡ lô. `delta`: AUC(lô + ảnh) − AUC(chỉ lô). `hieu_*` = mới − cũ, KTC 95% bootstrap ghép cặp theo người. `tuong_quan_diem`: tương quan điểm dự đoán cũ–mới.*

**Kết luận theo luật đặt trước:** 2b — thay đổi đáng kể: CHẠY LẠI chuỗi dev trên đặc trưng mới.

# Việc (1) lần 2 — Vùng so với toàn ảnh: tinh chỉnh C bằng inner CV và thêm stacking (BioViL-T đóng băng, V2)

Tập: dev mới sau P0.5, có ảnh: 1500 lượt, 367 dương, 1345 người, 18 lô (16 lô đủ hai nhãn). Fold ngoài = 5 fold mới (P0.5). **C của hồi quy logistic** (độ mạnh điều chuẩn; C nhỏ = mô hình bị kìm nhiều hơn) chọn riêng cho từng biểu diễn và từng fold ngoài bằng **inner CV**: 5 fold con theo người trên phần huấn luyện, lưới C = 1e-06, 3.16228e-06, 1e-05, 3.16228e-05, 0.0001, 0.000316228, 0.001, 0.00316228, 0.01, 0.0316228, 0.1, 0.316228, 1, 3.16228, 10, tiêu chí ROC-AUC gộp; mô hình có cột lô (dùng tính Δ) và mô hình chỉ lô cũng được chọn C như vậy. PCA fit riêng từng vùng trong fold.

**S — Stacking:** trong mỗi fold ngoài, 6 hồi quy logistic, mỗi cái chỉ thấy một vùng (C chọn bằng inner CV); điểm của chúng trên phần huấn luyện lấy **out-of-fold nested** (mỗi điểm do mô hình chưa thấy lượt đó, theo 5 fold con); rồi một hồi quy logistic (C chọn bằng inner CV) học trên 6 điểm. Để tính Δ, hồi quy cuối nhận thêm cột lô. KTC = khoảng tin cậy 95% bootstrap theo người (1000 lần). Tập test không mở.

### AUC trong lô và Δ theo cách biểu diễn (C đã tinh chỉnh)

| cach | so_chieu | C_chon_5_fold | so_fold_C_o_mep_luoi | L1_auc | L1_ci | auc_trong_lo | auc_trong_lo_ci | delta | delta_ci |
|---|---|---|---|---|---|---|---|---|---|
| A — Toàn ảnh (img_embedding = trung bình mọi patch) | 512 | 0.01, 0.1, 0.0316228, 0.1, 0.316228 | 0 | 0.749 | [0.722; 0.776] | 0.730 | [0.696; 0.762] | 0.083 | [+0.056; +0.112] |
| C — Trung bình 6 vector vùng | 512 | 0.01, 0.0316228, 0.0316228, 0.0316228, 0.1 | 0 | 0.758 | [0.731; 0.784] | 0.740 | [0.704; 0.771] | 0.093 | [+0.066; +0.120] |
| D — 6 vùng nối lại | 3072 | 0.000316228, 0.001, 0.001, 0.00316228, 0.001 | 0 | 0.739 | [0.711; 0.765] | 0.717 | [0.678; 0.750] | 0.058 | [+0.024; +0.091] |
| E — 6 vùng + PCA 32 chiều/vùng | 192 | 3.16228e-06, 0.000316228, 1e-06, 0.000316228, 0.000316228 | 1 | 0.714 | [0.685; 0.747] | 0.690 | [0.651; 0.728] | 0.078 | [+0.047; +0.108] |
| F — 6 vùng + PCA 64 chiều/vùng | 384 | 0.0001, 3.16228e-06, 1e-05, 1e-05, 3.16228e-06 | 0 | 0.738 | [0.711; 0.767] | 0.705 | [0.664; 0.745] | 0.066 | [+0.035; +0.097] |
| S — Stacking: 6 LR theo vùng → LR trên 6 điểm | 6 điểm | xem bảng 3 | 0 | 0.762 | [0.735; 0.787] | 0.740 | [0.702; 0.774] | 0.089 | [+0.064; +0.114] |

_Tệp: `1_ket_qua.csv`_

*`auc_trong_lo`: ROC-AUC (xác suất xếp ca dương cao hơn ca âm; 0,5 = đoán bừa) tính riêng trong từng lô đủ hai nhãn, trung bình theo cỡ lô. `delta`: AUC(lô + ảnh) − AUC(chỉ lô). `L1_auc`: AUC gộp. `C_chon_5_fold`: C được inner CV chọn ở 5 fold ngoài.*

### So sánh ghép cặp AUC trong lô (cùng người, cùng mẫu bootstrap)

| so_sanh | y_nghia | hieu_auc_trong_lo | ktc | P_hieu_duong |
|---|---|---|---|---|
| C − A | Chỉ lấy vùng phổi, gộp trung bình | 0.010 | [-0.003; +0.023] | 0.950 |
| S − C | Stacking 6 vùng so với trung bình 6 vùng (so sánh thầy yêu cầu) | 0.000 | [-0.024; +0.024] | 0.508 |
| D − C | 6 vùng nối lại so với trung bình 6 vùng | -0.023 | [-0.044; -0.002] | 0.014 |
| E − C | PCA 32 so với trung bình 6 vùng | -0.050 | [-0.084; -0.016] | 0.003 |
| F − C | PCA 64 so với trung bình 6 vùng | -0.034 | [-0.072; +0.000] | 0.027 |
| E − D | PCA 32 so với 6 vùng nối lại | -0.027 | [-0.058; +0.005] | 0.066 |
| F − D | PCA 64 so với 6 vùng nối lại | -0.012 | [-0.043; +0.018] | 0.222 |
| S − A | Stacking so với toàn ảnh | 0.010 | [-0.015; +0.035] | 0.787 |

_Tệp: `2_so_sanh_ghep_cap.csv`_

*`hieu_auc_trong_lo`: AUC trong lô cách trước − cách sau. KTC chứa 0 = chưa phân biệt được. `P_hieu_duong`: tỉ lệ lần bootstrap hiệu > 0.*

### C được chọn trong stacking (theo fold ngoài)

| fold | C_vung1 | C_vung2 | C_vung3 | C_vung4 | C_vung5 | C_vung6 | C_meta | C_meta_co_lo |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.003 | 0.100 | 0.032 | 0.032 | 0.010 | 0.100 | 0.032 | 0.010 |
| 1 | 0.010 | 0.010 | 0.010 | 0.032 | 0.032 | 0.032 | 0.010 | 0.032 |
| 2 | 0.003 | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 | 0.100 | 0.010 |
| 3 | 0.100 | 0.032 | 0.010 | 0.032 | 0.032 | 0.032 | 0.032 | 0.010 |
| 4 | 0.003 | 0.032 | 0.010 | 0.032 | 0.032 | 0.032 | 0.003 | 0.032 |

_Tệp: `3_C_stacking.csv`_

Mốc Δ (mô hình chỉ lô) giữ C = 0,1 cố định, AUC gộp = 0.707. Lý do: AUC inner CV của nó gần như không đổi theo C (thứ tự 18 lô không phụ thuộc C), nên inner CV chọn C gần như ngẫu nhiên; khi chọn C rất nhỏ, điểm mỗi fold dồn sát 0,5 với độ lệch khác nhau, AUC gộp của mốc tụt (thử: 0,707 → 0,674) và Δ bị phóng đại giả khoảng 0,03. Lưới C mở rộng xuống 1e-6 vì ở lần chạy thử với lưới 1e-4..10, F chọn đúng mép dưới ở cả 5 fold.

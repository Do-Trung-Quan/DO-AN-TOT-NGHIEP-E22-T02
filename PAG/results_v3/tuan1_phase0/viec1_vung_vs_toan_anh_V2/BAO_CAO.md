# Việc (1) — Đặc trưng vùng thấp hơn toàn ảnh: do mất thông tin hay do quá nhiều chiều? (BioViL-T đóng băng, V2)

Tập: **dev mới sau P0.5** (ghép người, chia lại fold), có ảnh: 1500 lượt, 367 dương, 1345 người, 18 lô (16 lô đủ hai nhãn dùng cho AUC trong lô). Mô hình L1 = hồi quy logistic (chuẩn hóa, C=0,1, cân lớp), 5 fold mới theo người. PCA fit riêng từng vùng, **chỉ trên phần huấn luyện của mỗi fold**. KTC = khoảng tin cậy 95% bootstrap theo người (1000 lần). Tập test không mở.

Đã kiểm tra: vector toàn ảnh `img_embedding` của BioViL-T đúng bằng trung bình mọi `patch_embeddings` (chưa qua lớp chiếu), tức cùng không gian với vector vùng — nên A, C, D khác nhau chỉ ở cách gộp patch. C (trung bình 6 vùng) là phép chiếu tuyến tính của D (6 vùng nối lại): mọi thông tin trong C đều có trong D.

### AUC trong lô và Δ theo cách biểu diễn

| cach | so_chieu | L1_auc | L1_ci | auc_trong_lo | auc_trong_lo_ci | delta | delta_ci |
|---|---|---|---|---|---|---|---|
| A — Toàn ảnh (img_embedding = trung bình mọi patch) | 512 | 0.759 | [0.731; 0.786] | 0.736 | [0.701; 0.768] | 0.086 | [+0.058; +0.115] |
| C — Trung bình 6 vector vùng | 512 | 0.760 | [0.733; 0.786] | 0.738 | [0.702; 0.770] | 0.089 | [+0.062; +0.118] |
| D — 6 vùng nối lại | 3072 | 0.704 | [0.675; 0.734] | 0.660 | [0.613; 0.706] | 0.030 | [-0.004; +0.065] |
| E — 6 vùng + PCA 32 chiều/vùng | 192 | 0.728 | [0.701; 0.758] | 0.691 | [0.651; 0.731] | 0.054 | [+0.022; +0.085] |
| F — 6 vùng + PCA 64 chiều/vùng | 384 | 0.688 | [0.658; 0.719] | 0.643 | [0.599; 0.686] | 0.006 | [-0.029; +0.039] |

_Tệp: `1_ket_qua.csv`_

*`auc_trong_lo`: ROC-AUC (xác suất xếp ca dương cao hơn ca âm; 0,5 = đoán bừa) tính riêng trong từng lô đủ hai nhãn, trung bình theo cỡ lô. `delta`: AUC(lô + ảnh) − AUC(chỉ lô); KTC trọn trên 0 = ảnh thêm thông tin ngoài lô. `L1_auc`: AUC gộp.*

### So sánh ghép cặp AUC trong lô (cùng người, cùng mẫu bootstrap)

| so_sanh | y_nghia | hieu_auc_trong_lo | ktc | P_hieu_duong |
|---|---|---|---|---|
| C − A | Chỉ lấy vùng phổi, gộp trung bình (cùng không gian, cùng số chiều) | 0.002 | [-0.007; +0.013] | 0.703 |
| D − C | Giữ riêng 6 vùng thay vì gộp (C là phép chiếu tuyến tính của D → D chứa đủ thông tin của C) | -0.079 | [-0.108; -0.044] | 0.000 |
| E − D | Giảm chiều PCA 32 so với 6 vùng nối lại | 0.031 | [+0.003; +0.058] | 0.989 |
| F − D | Giảm chiều PCA 64 so với 6 vùng nối lại | -0.017 | [-0.044; +0.008] | 0.083 |
| E − C | PCA 32 so với trung bình 6 vùng | -0.047 | [-0.077; -0.017] | 0.000 |
| F − C | PCA 64 so với trung bình 6 vùng | -0.095 | [-0.128; -0.060] | 0.000 |
| D − A | 6 vùng nối lại so với toàn ảnh (khoảng cách cần giải thích) | -0.076 | [-0.108; -0.040] | 0.000 |

_Tệp: `2_so_sanh_ghep_cap.csv`_

*`hieu_auc_trong_lo`: AUC trong lô cách trước − cách sau. KTC chứa 0 = chưa phân biệt được. `P_hieu_duong`: tỉ lệ lần bootstrap hiệu > 0.*

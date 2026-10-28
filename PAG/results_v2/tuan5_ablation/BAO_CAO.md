# Tuần 5 — Ablation: mỗi lần bỏ/đổi một thành phần của PG

### Ablation — dev OOF đã hiệu chỉnh

| bien_the | mo_ta | auc_có ảnh | auc_trong_lo | auc_không ảnh | delta_có ảnh | ci_có ảnh | delta_không ảnh | ci_không ảnh |
|---|---|---|---|---|---|---|---|---|
| PG | đầy đủ | 0.756 | 0.704 | 0.674 | nan | nan | nan | nan |
| PG-drop | PG bình thường, bỏ node ảnh lúc đánh giá | 0.658 | 0.608 | 0.678 | -0.098 | [-0.128–-0.071] | 0.004 | [-0.001–0.008] |
| PG-noedge | bỏ hết cạnh (chỉ tự nối + CLS) | 0.747 | 0.682 | 0.640 | -0.010 | [-0.022–0.003] | -0.034 | [-0.049–-0.018] |
| PG-complete | đồ thị đầy đủ, một loại cạnh | 0.770 | 0.715 | 0.676 | 0.013 | [0.000–0.026] | 0.003 | [-0.008–0.014] |
| PG-shuffle | xáo loại cạnh giữa các vùng (phá giải phẫu) | 0.755 | 0.703 | 0.672 | -0.002 | [-0.004–0.001] | -0.002 | [-0.004–0.001] |
| PG-global | 1 node ảnh toàn cục thay 6 vùng | 0.723 | 0.671 | 0.667 | -0.033 | [-0.055–-0.012] | -0.007 | [-0.022–0.009] |
| PG-noMD | không modality dropout | 0.727 | 0.665 | 0.682 | -0.029 | [-0.052–-0.006] | 0.009 | [-0.009–0.026] |
| PG-noclin | chỉ ảnh + khám (bỏ 5 nhóm lâm sàng, người có phim) | 0.762 | 0.721 | nan | 0.006 | [-0.013–0.024] | nan | nan |
| PG-noctx | bỏ node tỉnh | 0.760 | 0.710 | 0.641 | 0.004 | [-0.005–0.014] | -0.033 | [-0.043–-0.022] |
| PG-RGCN | R-GCN thay Graph Transformer | 0.755 | 0.703 | 0.638 | -0.002 | [-0.014–0.010] | -0.036 | [-0.051–-0.019] |
| PG-nobnn1 | chỉ bỏ biến bnn | 0.758 | 0.704 | 0.659 | 0.002 | [-0.009–0.013] | -0.015 | [-0.026–-0.003] |
| PG-nobnn | bỏ bnn + 4 biến tiền sử bệnh (tiensuhh, benhhh, tsnoi, tsngoai) | 0.746 | 0.690 | 0.671 | -0.011 | [-0.026–0.006] | -0.002 | [-0.020–0.016] |
| PG-enc-b | ảnh bằng ResNet50-xrv | 0.770 | 0.723 | 0.654 | 0.014 | [-0.008–0.036] | -0.020 | [-0.036–-0.004] |
| PG-enc-c | ảnh bằng BioViL-T control | 0.722 | 0.668 | 0.645 | -0.035 | [-0.055–-0.014] | -0.028 | [-0.045–-0.011] |
| PG-enc-d | ảnh bằng CheXFound | 0.774 | 0.720 | 0.651 | 0.017 | [-0.005–0.040] | -0.023 | [-0.039–-0.006] |

_Tệp: `ablation.csv`_

*`delta_*` = AUC(biến thể) − AUC(PG) ghép cặp, kèm KTC 95%; âm nghĩa là bỏ thành phần đó làm mô hình kém đi. PG-E6shuffle / PG-noE6 / PG-1exam trùng PG trong cấu trúc dự phòng nên không chạy.*

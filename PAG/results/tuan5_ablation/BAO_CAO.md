# Tuần 5 — Ablation: mỗi lần bỏ/đổi một thành phần của PG

### Ablation — dev OOF đã hiệu chỉnh

| bien_the | mo_ta | auc_có ảnh | auc_trong_lo | auc_không ảnh | delta_có ảnh | ci_có ảnh | delta_không ảnh | ci_không ảnh |
|---|---|---|---|---|---|---|---|---|
| PG | đầy đủ | 0.754 | 0.695 | 0.679 | nan | nan | nan | nan |
| PG-drop | PG bình thường, bỏ node ảnh lúc đánh giá | 0.662 | 0.581 | 0.683 | -0.092 | [-0.122–-0.063] | 0.004 | [-0.001–0.008] |
| PG-noedge | bỏ hết cạnh (chỉ tự nối + CLS) | 0.733 | 0.679 | 0.624 | -0.022 | [-0.037–-0.005] | -0.055 | [-0.073–-0.037] |
| PG-complete | đồ thị đầy đủ, một loại cạnh | 0.744 | 0.677 | 0.673 | -0.010 | [-0.025–0.004] | -0.006 | [-0.018–0.005] |
| PG-shuffle | xáo loại cạnh giữa các vùng (phá giải phẫu) | 0.751 | 0.687 | 0.680 | -0.003 | [-0.009–0.003] | 0.001 | [-0.006–0.008] |
| PG-global | 1 node ảnh toàn cục thay 6 vùng | 0.721 | 0.656 | 0.656 | -0.033 | [-0.053–-0.013] | -0.023 | [-0.036–-0.010] |
| PG-noMD | không modality dropout | 0.766 | 0.711 | 0.698 | 0.012 | [-0.001–0.026] | 0.019 | [0.004–0.034] |
| PG-noclin | chỉ ảnh + khám (bỏ 5 nhóm lâm sàng, người có phim) | 0.762 | 0.726 | nan | 0.007 | [-0.016–0.031] | nan | nan |
| PG-noctx | bỏ node tỉnh | 0.753 | 0.687 | 0.622 | -0.002 | [-0.015–0.012] | -0.057 | [-0.075–-0.040] |
| PG-RGCN | R-GCN thay Graph Transformer | 0.760 | 0.684 | 0.670 | 0.006 | [-0.007–0.019] | -0.009 | [-0.022–0.003] |
| PG-nobnn1 | chỉ bỏ biến bnn | 0.742 | 0.677 | 0.649 | -0.012 | [-0.025–0.002] | -0.030 | [-0.046–-0.015] |
| PG-nobnn | bỏ bnn + 4 biến tiền sử bệnh (tiensuhh, benhhh, tsnoi, tsngoai) | 0.735 | 0.662 | 0.670 | -0.019 | [-0.032–-0.006] | -0.009 | [-0.019–0.002] |
| PG-enc-b | ảnh bằng ResNet50-xrv | 0.763 | 0.715 | 0.684 | 0.009 | [-0.013–0.032] | 0.005 | [-0.008–0.018] |
| PG-enc-c | ảnh bằng BioViL-T control | 0.742 | 0.664 | 0.661 | -0.012 | [-0.034–0.010] | -0.018 | [-0.032–-0.005] |

_Tệp: `ablation.csv`_

*`delta_*` = AUC(biến thể) − AUC(PG) ghép cặp, kèm KTC 95%; âm nghĩa là bỏ thành phần đó làm mô hình kém đi. PG-E6shuffle / PG-noE6 / PG-1exam trùng PG trong cấu trúc dự phòng nên không chạy.*

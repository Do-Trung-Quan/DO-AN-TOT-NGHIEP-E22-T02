# Tuần 4 — Kiểm định giả thuyết khai báo trước (bước 5)

Khai báo trước: `khai_bao_truoc_v3.json` (mã băm `04060a166b23a75e`). Bootstrap 1000 lần theo người, Holm một phía α = 0,025 trên 5 so sánh chính. Δ = chỉ số(PG) − chỉ số(đối chứng), ghép cặp trên cùng lượt khám. `P_delta_duong` = tỉ lệ lần bootstrap PG hơn (KHÔNG phải p-value). `p_mot_phia` = tỉ lệ lần bootstrap Δ ≤ biên.

### Các so sánh chính

| ma | so_sanh | nhom | chi_so | loai | bien | n | chi_so_a | chi_so_b | delta | ktc | P_delta_duong | p_mot_phia | dat_theo_KTC | dat_sau_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1a | PG vs C | có ảnh | AUC trong lo | khong_kem | -0.020 | 1500 | 0.716 | 0.729 | -0.012 | [-0.047; +0.022] | 0.239 | 0.302 | False | False |
| H1b | PG vs L0 | không ảnh | ROC-AUC gop | khong_kem | -0.020 | 5183 | 0.655 | 0.679 | -0.024 | [-0.043; -0.005] | 0.002 | 0.655 | False | False |
| H2 | PG vs PG-shuffle | có ảnh | AUC trong lo | hon | 0.000 | 1500 | 0.716 | 0.718 | -0.001 | [-0.005; +0.002] | 0.211 | 0.789 | False | False |
| H3a | PG vs T2 | có ảnh | AUC trong lo | khong_kem | -0.020 | 1500 | 0.716 | 0.645 | 0.072 | [+0.035; +0.106] | 1.000 | 0.000 | True | True |
| H3b | PG vs T2 | không ảnh | ROC-AUC gop | khong_kem | -0.020 | 5183 | 0.655 | 0.627 | 0.028 | [+0.007; +0.050] | 0.997 | 0.000 | True | True |

_Tệp: `gia_thuyet_chinh.csv`_

## Kết luận

- **H1: KHÔNG ĐẠT** (DAT khi ca H1a va H1b deu dat sau Holm)
- **H2: KHÔNG ĐẠT** (DAT khi H2 dat sau Holm)
- **H3: ĐẠT** (DAT khi ca H3a va H3b deu dat sau Holm)

**Điểm duyệt 2:** H1 và H2 đều KHÔNG ĐẠT → theo khai báo trước phải DỪNG phát triển mô hình và báo PGS.

### So sánh khám phá (không hiệu chỉnh đa so sánh, chỉ tham khảo)

| so_sanh | nhom | chi_so | n | chi_so_a | chi_so_b | delta | ktc | P_delta_duong |
|---|---|---|---|---|---|---|---|---|
| PG-global vs C | có ảnh | AUC trong lo | 1500 | 0.656 | 0.729 | -0.072 | [-0.108; -0.035] | 0.000 |
| PG-global vs L0 | không ảnh | ROC-AUC gop | 5183 | 0.642 | 0.679 | -0.037 | [-0.055; -0.019] | 0.000 |
| PG-global vs T2 | có ảnh | AUC trong lo | 1500 | 0.656 | 0.645 | 0.012 | [-0.018; +0.041] | 0.790 |
| PG-global vs T2 | không ảnh | ROC-AUC gop | 5183 | 0.642 | 0.627 | 0.015 | [-0.004; +0.034] | 0.944 |
| PG vs PG-global | có ảnh | AUC trong lo | 1500 | 0.716 | 0.656 | 0.060 | [+0.027; +0.090] | 1.000 |
| PG vs PG-global | không ảnh | ROC-AUC gop | 5183 | 0.655 | 0.642 | 0.013 | [-0.003; +0.029] | 0.950 |
| PG vs L2 | có ảnh | AUC trong lo | 1500 | 0.716 | 0.717 | -0.001 | [-0.030; +0.029] | 0.495 |
| PG vs L2 | không ảnh | ROC-AUC gop | 5183 | 0.655 | 0.682 | -0.027 | [-0.046; -0.008] | 0.002 |
| PG vs X2 | không ảnh | ROC-AUC gop | 5183 | 0.655 | 0.699 | -0.044 | [-0.063; -0.025] | 0.000 |
| PG vs GAT | không ảnh | ROC-AUC gop | 5183 | 0.655 | 0.673 | -0.018 | [-0.041; +0.008] | 0.091 |
| PG vs B3b | không ảnh | ROC-AUC gop | 5183 | 0.655 | 0.664 | -0.009 | [-0.029; +0.011] | 0.201 |
| PG vs L0 | không ảnh — Đồng Nai | ROC-AUC gop | 1205 | 0.623 | 0.634 | -0.011 | [-0.050; +0.032] | 0.299 |
| PG vs L0 | không ảnh — tỉnh khác | ROC-AUC gop | 3978 | 0.667 | 0.692 | -0.025 | [-0.046; -0.004] | 0.008 |

_Tệp: `kham_pha.csv`_

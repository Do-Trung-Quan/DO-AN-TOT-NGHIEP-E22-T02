# P0.1 — V2 / nguồn frozen
Config `2a4ad58122d08e57` · fold: fusion_dataset (file fold hien co) · ảnh zone_missing (dev): 0 · 1571 dòng dev có ảnh (393 dương, 1493 bệnh nhân) · 18 lô · bootstrap 2000 lần theo bệnh nhân.

## 1. Probe lô
Accuracy 0.349 [0.326; 0.372]; lớp đa số 0.108; nhãn hoán vị trung bình 0.08809675366008911. Accuracy cân bằng 0.271; AUC OvR macro 0.835.

Khám phá, trong cùng một tỉnh: accuracy 0.419 so với lớp đa số 0.299.

## 2. AUC trong lô của L1
AUC gộp 0.742. AUC trong lô (chính): 0.715 [0.682; 0.747], dùng 16 lô, loại 2 lô (20190827, unknown). Nhạy cảm (≥10 dương và ≥10 âm): 0.696 [0.660; 0.733], dùng 14 lô.

| Lô | n | dương | âm | AUC |
|---|---|---|---|---|
| 20181219 | 170 | 25 | 145 | 0.614 |
| 20181222 | 75 | 12 | 63 | 0.648 |
| 20181223 | 141 | 62 | 79 | 0.731 |
| 20181224 | 63 | 39 | 24 | 0.751 |
| 20181227 | 128 | 14 | 114 | 0.763 |
| 20190103 | 38 | 12 | 26 | 0.843 |
| 20190114 | 144 | 30 | 114 | 0.725 |
| 20190115 | 155 | 29 | 126 | 0.705 |
| 20190605 | 115 | 46 | 69 | 0.719 |
| 20190606 | 100 | 41 | 59 | 0.612 |
| 20190607 | 27 | 11 | 16 | 0.511 |
| 20190620 | 85 | 20 | 65 | 0.632 |
| 20190621 | 71 | 22 | 49 | 0.758 |
| 20190827 | 5 | 5 | 0 | nan |
| 20191113 | 62 | 3 | 59 | 0.881 |
| 20191115 | 143 | 7 | 136 | 0.827 |
| 20191116 | 44 | 10 | 34 | 0.721 |
| unknown | 5 | 5 | 0 | nan |

## 3. Giá trị gia tăng của ảnh
AUC gộp LR(lô) 0.705; LR(lô+ảnh) 0.776. ΔAUC gộp +0.0710 [+0.0471; +0.0966], P(Δ>0)=100%. ΔAUC trong lô +0.1931 [+0.1597; +0.2263], P(Δ>0)=100%.

## 4. Leave-batch-out (LOBO)
AUC gộp 0.741 [0.714; 0.767]; AUC trong lô 0.726 [0.695; 0.757]; chênh so với in-fold +0.0113 [-0.0023; +0.0240], P(Δ>0)=95%.

## Đối chiếu bảng quyết định của thầy
AUC trong lô 0.715; cận dưới KTC của ΔAUC gộp +0.0471 → **tiếp tục**. chi doi chieu voi bang cua thay; quyet dinh thuoc ve PGS.
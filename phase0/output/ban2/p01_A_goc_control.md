# P0.1 — A_goc / nguồn control
Config `774c692d84e7fd34` · 1571 dòng dev có ảnh (393 dương, 1493 bệnh nhân) · 18 lô · bootstrap 2000 lần theo bệnh nhân.

## 1. Probe lô
Accuracy 0.430 [0.406; 0.454]; lớp đa số 0.108; nhãn hoán vị trung bình 0.08230426479949078. Accuracy cân bằng 0.347; AUC OvR macro 0.894.

Khám phá, trong cùng một tỉnh: accuracy 0.499 so với lớp đa số 0.299.

## 2. AUC trong lô của L1
AUC gộp 0.760. AUC trong lô (chính): 0.719 [0.686; 0.751], dùng 16 lô, loại 2 lô (20190827, unknown). Nhạy cảm (≥10 dương và ≥10 âm): 0.702 [0.665; 0.738], dùng 14 lô.

| Lô | n | dương | âm | AUC |
|---|---|---|---|---|
| 20181219 | 170 | 25 | 145 | 0.577 |
| 20181222 | 75 | 12 | 63 | 0.742 |
| 20181223 | 141 | 62 | 79 | 0.745 |
| 20181224 | 63 | 39 | 24 | 0.792 |
| 20181227 | 128 | 14 | 114 | 0.766 |
| 20190103 | 38 | 12 | 26 | 0.869 |
| 20190114 | 144 | 30 | 114 | 0.679 |
| 20190115 | 155 | 29 | 126 | 0.732 |
| 20190605 | 115 | 46 | 69 | 0.702 |
| 20190606 | 100 | 41 | 59 | 0.679 |
| 20190607 | 27 | 11 | 16 | 0.534 |
| 20190620 | 85 | 20 | 65 | 0.688 |
| 20190621 | 71 | 22 | 49 | 0.672 |
| 20190827 | 5 | 5 | 0 | nan |
| 20191113 | 62 | 3 | 59 | 0.927 |
| 20191115 | 143 | 7 | 136 | 0.796 |
| 20191116 | 44 | 10 | 34 | 0.721 |
| unknown | 5 | 5 | 0 | nan |

## 3. Giá trị gia tăng của ảnh
AUC gộp LR(lô) 0.705; LR(lô+ảnh) 0.790. ΔAUC gộp +0.0849 [+0.0601; +0.1099], P(Δ>0)=100%. ΔAUC trong lô +0.2106 [+0.1753; +0.2447], P(Δ>0)=100%.

## 4. Leave-batch-out
AUC trong lô 0.722 [0.690; 0.754]; chênh so với in-fold +0.0028 [-0.0086; +0.0142], P(Δ>0)=69%.

## Đối chiếu bảng quyết định của thầy
AUC trong lô 0.719; cận dưới KTC của ΔAUC gộp +0.0601 → **tiep_tuc**. chi doi chieu voi bang cua thay; quyet dinh thuoc ve PGS.
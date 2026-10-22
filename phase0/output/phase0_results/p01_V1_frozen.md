# P0.1 — V1 / nguồn frozen
Config `2a4ad58122d08e57` · fold: fusion_dataset (file fold hien co) · ảnh zone_missing (dev): 0 · 1571 dòng dev có ảnh (393 dương, 1493 bệnh nhân) · 18 lô · bootstrap 2000 lần theo bệnh nhân.

## 1. Probe lô
Accuracy 0.409 [0.384; 0.431]; lớp đa số 0.108; nhãn hoán vị trung bình 0.08281349458943349. Accuracy cân bằng 0.333; AUC OvR macro 0.864.

Khám phá, trong cùng một tỉnh: accuracy 0.479 so với lớp đa số 0.299.

## 2. AUC trong lô của L1
AUC gộp 0.749. AUC trong lô (chính): 0.711 [0.676; 0.742], dùng 16 lô, loại 2 lô (20190827, unknown). Nhạy cảm (≥10 dương và ≥10 âm): 0.691 [0.651; 0.728], dùng 14 lô.

| Lô | n | dương | âm | AUC |
|---|---|---|---|---|
| 20181219 | 170 | 25 | 145 | 0.636 |
| 20181222 | 75 | 12 | 63 | 0.661 |
| 20181223 | 141 | 62 | 79 | 0.741 |
| 20181224 | 63 | 39 | 24 | 0.770 |
| 20181227 | 128 | 14 | 114 | 0.769 |
| 20190103 | 38 | 12 | 26 | 0.910 |
| 20190114 | 144 | 30 | 114 | 0.615 |
| 20190115 | 155 | 29 | 126 | 0.652 |
| 20190605 | 115 | 46 | 69 | 0.666 |
| 20190606 | 100 | 41 | 59 | 0.676 |
| 20190607 | 27 | 11 | 16 | 0.534 |
| 20190620 | 85 | 20 | 65 | 0.658 |
| 20190621 | 71 | 22 | 49 | 0.771 |
| 20190827 | 5 | 5 | 0 | nan |
| 20191113 | 62 | 3 | 59 | 0.842 |
| 20191115 | 143 | 7 | 136 | 0.840 |
| 20191116 | 44 | 10 | 34 | 0.782 |
| unknown | 5 | 5 | 0 | nan |

## 3. Giá trị gia tăng của ảnh
AUC gộp LR(lô) 0.705; LR(lô+ảnh) 0.785. ΔAUC gộp +0.0798 [+0.0528; +0.1060], P(Δ>0)=100%. ΔAUC trong lô +0.2070 [+0.1713; +0.2391], P(Δ>0)=100%.

## 4. Leave-batch-out (LOBO)
AUC gộp 0.748 [0.721; 0.774]; AUC trong lô 0.732 [0.699; 0.762]; chênh so với in-fold +0.0216 [+0.0084; +0.0360], P(Δ>0)=100%.

## Đối chiếu bảng quyết định của thầy
AUC trong lô 0.711; cận dưới KTC của ΔAUC gộp +0.0528 → **tiếp tục**. chi doi chieu voi bang cua thay; quyet dinh thuoc ve PGS.
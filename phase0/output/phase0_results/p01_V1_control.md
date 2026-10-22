# P0.1 — V1 / nguồn control
Config `2a4ad58122d08e57` · fold: fusion_dataset (file fold hien co) · ảnh zone_missing (dev): 0 · 1571 dòng dev có ảnh (393 dương, 1493 bệnh nhân) · 18 lô · bootstrap 2000 lần theo bệnh nhân.

## 1. Probe lô
Accuracy 0.377 [0.354; 0.400]; lớp đa số 0.108; nhãn hoán vị trung bình 0.08510502864417568. Accuracy cân bằng 0.307; AUC OvR macro 0.860.

Khám phá, trong cùng một tỉnh: accuracy 0.455 so với lớp đa số 0.299.

## 2. AUC trong lô của L1
AUC gộp 0.750. AUC trong lô (chính): 0.712 [0.677; 0.744], dùng 16 lô, loại 2 lô (20190827, unknown). Nhạy cảm (≥10 dương và ≥10 âm): 0.702 [0.663; 0.732], dùng 14 lô.

| Lô | n | dương | âm | AUC |
|---|---|---|---|---|
| 20181219 | 170 | 25 | 145 | 0.581 |
| 20181222 | 75 | 12 | 63 | 0.771 |
| 20181223 | 141 | 62 | 79 | 0.703 |
| 20181224 | 63 | 39 | 24 | 0.785 |
| 20181227 | 128 | 14 | 114 | 0.734 |
| 20190103 | 38 | 12 | 26 | 0.837 |
| 20190114 | 144 | 30 | 114 | 0.698 |
| 20190115 | 155 | 29 | 126 | 0.688 |
| 20190605 | 115 | 46 | 69 | 0.767 |
| 20190606 | 100 | 41 | 59 | 0.660 |
| 20190607 | 27 | 11 | 16 | 0.653 |
| 20190620 | 85 | 20 | 65 | 0.675 |
| 20190621 | 71 | 22 | 49 | 0.702 |
| 20190827 | 5 | 5 | 0 | nan |
| 20191113 | 62 | 3 | 59 | 0.887 |
| 20191115 | 143 | 7 | 136 | 0.737 |
| 20191116 | 44 | 10 | 34 | 0.774 |
| unknown | 5 | 5 | 0 | nan |

## 3. Giá trị gia tăng của ảnh
AUC gộp LR(lô) 0.705; LR(lô+ảnh) 0.790. ΔAUC gộp +0.0843 [+0.0603; +0.1082], P(Δ>0)=100%. ΔAUC trong lô +0.2010 [+0.1651; +0.2332], P(Δ>0)=100%.

## 4. Leave-batch-out (LOBO)
AUC gộp 0.729 [0.702; 0.756]; AUC trong lô 0.707 [0.674; 0.739]; chênh so với in-fold -0.0048 [-0.0171; +0.0092], P(Δ>0)=25%.

## Đối chiếu bảng quyết định của thầy
AUC trong lô 0.712; cận dưới KTC của ΔAUC gộp +0.0603 → **tiếp tục**. chi doi chieu voi bang cua thay; quyet dinh thuoc ve PGS.
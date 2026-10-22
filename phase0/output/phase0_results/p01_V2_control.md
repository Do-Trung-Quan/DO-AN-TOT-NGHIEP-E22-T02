# P0.1 — V2 / nguồn control
Config `2a4ad58122d08e57` · fold: fusion_dataset (file fold hien co) · ảnh zone_missing (dev): 0 · 1571 dòng dev có ảnh (393 dương, 1493 bệnh nhân) · 18 lô · bootstrap 2000 lần theo bệnh nhân.

## 1. Probe lô
Accuracy 0.349 [0.325; 0.371]; lớp đa số 0.108; nhãn hoán vị trung bình 0.08733290897517505. Accuracy cân bằng 0.254; AUC OvR macro 0.820.

Khám phá, trong cùng một tỉnh: accuracy 0.437 so với lớp đa số 0.299.

## 2. AUC trong lô của L1
AUC gộp 0.741. AUC trong lô (chính): 0.697 [0.663; 0.731], dùng 16 lô, loại 2 lô (20190827, unknown). Nhạy cảm (≥10 dương và ≥10 âm): 0.686 [0.648; 0.724], dùng 14 lô.

| Lô | n | dương | âm | AUC |
|---|---|---|---|---|
| 20181219 | 170 | 25 | 145 | 0.639 |
| 20181222 | 75 | 12 | 63 | 0.741 |
| 20181223 | 141 | 62 | 79 | 0.753 |
| 20181224 | 63 | 39 | 24 | 0.717 |
| 20181227 | 128 | 14 | 114 | 0.702 |
| 20190103 | 38 | 12 | 26 | 0.792 |
| 20190114 | 144 | 30 | 114 | 0.668 |
| 20190115 | 155 | 29 | 126 | 0.629 |
| 20190605 | 115 | 46 | 69 | 0.721 |
| 20190606 | 100 | 41 | 59 | 0.645 |
| 20190607 | 27 | 11 | 16 | 0.648 |
| 20190620 | 85 | 20 | 65 | 0.669 |
| 20190621 | 71 | 22 | 49 | 0.723 |
| 20190827 | 5 | 5 | 0 | nan |
| 20191113 | 62 | 3 | 59 | 0.797 |
| 20191115 | 143 | 7 | 136 | 0.756 |
| 20191116 | 44 | 10 | 34 | 0.641 |
| unknown | 5 | 5 | 0 | nan |

## 3. Giá trị gia tăng của ảnh
AUC gộp LR(lô) 0.705; LR(lô+ảnh) 0.773. ΔAUC gộp +0.0681 [+0.0449; +0.0920], P(Δ>0)=100%. ΔAUC trong lô +0.1782 [+0.1430; +0.2134], P(Δ>0)=100%.

## 4. Leave-batch-out (LOBO)
AUC gộp 0.734 [0.705; 0.763]; AUC trong lô 0.699 [0.666; 0.732]; chênh so với in-fold +0.0020 [-0.0106; +0.0149], P(Δ>0)=62%.

## Đối chiếu bảng quyết định của thầy
AUC trong lô 0.697; cận dưới KTC của ΔAUC gộp +0.0449 → **báo cáo PGS trước khi làm tiếp**. chi doi chieu voi bang cua thay; quyet dinh thuoc ve PGS.
# P0.1 — A_goc / nguồn frozen
Config `774c692d84e7fd34` · 1571 dòng dev có ảnh (393 dương, 1493 bệnh nhân) · 18 lô · bootstrap 2000 lần theo bệnh nhân.

## 1. Probe lô
Accuracy 0.468 [0.443; 0.491]; lớp đa số 0.108; nhãn hoán vị trung bình 0.08230426479949078. Accuracy cân bằng 0.399; AUC OvR macro 0.899.

Khám phá, trong cùng một tỉnh: accuracy 0.532 so với lớp đa số 0.299.

## 2. AUC trong lô của L1
AUC gộp 0.759. AUC trong lô (chính): 0.716 [0.682; 0.748], dùng 16 lô, loại 2 lô (20190827, unknown). Nhạy cảm (≥10 dương và ≥10 âm): 0.698 [0.661; 0.733], dùng 14 lô.

| Lô | n | dương | âm | AUC |
|---|---|---|---|---|
| 20181219 | 170 | 25 | 145 | 0.617 |
| 20181222 | 75 | 12 | 63 | 0.716 |
| 20181223 | 141 | 62 | 79 | 0.721 |
| 20181224 | 63 | 39 | 24 | 0.799 |
| 20181227 | 128 | 14 | 114 | 0.708 |
| 20190103 | 38 | 12 | 26 | 0.920 |
| 20190114 | 144 | 30 | 114 | 0.715 |
| 20190115 | 155 | 29 | 126 | 0.669 |
| 20190605 | 115 | 46 | 69 | 0.658 |
| 20190606 | 100 | 41 | 59 | 0.670 |
| 20190607 | 27 | 11 | 16 | 0.545 |
| 20190620 | 85 | 20 | 65 | 0.660 |
| 20190621 | 71 | 22 | 49 | 0.782 |
| 20190827 | 5 | 5 | 0 | nan |
| 20191113 | 62 | 3 | 59 | 0.898 |
| 20191115 | 143 | 7 | 136 | 0.814 |
| 20191116 | 44 | 10 | 34 | 0.774 |
| unknown | 5 | 5 | 0 | nan |

## 3. Giá trị gia tăng của ảnh
AUC gộp LR(lô) 0.705; LR(lô+ảnh) 0.783. ΔAUC gộp +0.0778 [+0.0512; +0.1048], P(Δ>0)=100%. ΔAUC trong lô +0.2028 [+0.1677; +0.2370], P(Δ>0)=100%.

## 4. Leave-batch-out
AUC trong lô 0.738 [0.706; 0.767]; chênh so với in-fold +0.0217 [+0.0067; +0.0378], P(Δ>0)=100%.

## Đối chiếu bảng quyết định của thầy
AUC trong lô 0.716; cận dưới KTC của ΔAUC gộp +0.0512 → **tiep_tuc**. chi doi chieu voi bang cua thay; quyet dinh thuoc ve PGS.
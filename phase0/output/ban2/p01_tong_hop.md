# P0.1 — tổng hợp nhiễu theo lô chụp
Config `774c692d84e7fd34`.

| Biến thể | Nguồn | AUC gộp L1 | AUC trong lô L1 | (≥10 dương/âm) | LOLO trong lô | ΔAUC gộp (lô+ảnh vs lô) | Probe lô acc (đa số) | Hạng mục |
|---|---|---|---|---|---|---|---|---|
| A_goc | control | 0.760 | 0.719 [0.686; 0.751] | 0.702 [0.665; 0.738] | 0.722 [0.690; 0.754] | +0.085 [+0.060; +0.110] | 0.43 (0.11) | tiếp tục |
| A_goc | frozen | 0.759 | 0.716 [0.682; 0.748] | 0.698 [0.661; 0.733] | 0.738 [0.706; 0.767] | +0.078 [+0.051; +0.105] | 0.47 (0.11) | tiếp tục |

## Đối chiếu bảng quyết định (nguồn có AUC trong lô thấp hơn)

- **A_goc**: nguồn `frozen`, AUC trong lô 0.716, cận dưới KTC ΔAUC gộp +0.0512 → tiếp tục.

Đây chỉ là đối chiếu với bảng của thầy; quyết định thuộc về PGS.
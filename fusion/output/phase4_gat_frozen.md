# Phase 4 — GAT (nguon anh `frozen`)

- Sinh luc: 2026-09-27T13:00:19.550278+00:00 | thiet bi: cuda
- 3 seed x toi da 300 epoch (early stopping, patience 6) | hidden 64, 4 head, dropout 0.3, lr 0.005 | LayerNorm | diem hang xom: CO

| Nhom | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Toan bo dev | 6882 | 823 | 0.722 [0.702–0.742] | 0.340 [0.306–0.375] |
| CO anh | 1571 | 393 | 0.745 [0.717–0.773] | 0.518 [0.465–0.567] |
| KHONG anh | 5311 | 430 | 0.632 [0.606–0.659] | 0.136 [0.117–0.161] |

## Doi chieu voi Phase 3 (bootstrap ghep cap, cung benh nhan)

| So sanh | Nhom | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|---|
| GAT − B3b | khong_anh | -0.0018 | [-0.0198; +0.0179] | 41% | **TRUOT** |
| GAT − B3 | khong_anh | +0.0291 | [+0.0085; +0.0542] | 100% | DAT |
| GAT − B0 | khong_anh | +0.0135 | [-0.0052; +0.0327] | 92% | **TRUOT** |
| GAT − B2 | co_anh | +0.0034 | [-0.0143; +0.0242] | 64% | DAT |
| GAT − L2 | co_anh | -0.0022 | [-0.0242; +0.0211] | 42% | DAT |

*`phai_hon`: KTC phai nam tron ben duong thi GAT moi thuc su hon. `khong_kem`: chi truot khi GAT thua ro rang.*

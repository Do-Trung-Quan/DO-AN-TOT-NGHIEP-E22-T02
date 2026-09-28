# Phase 4 — GAT (nguon anh `control`)

- Sinh luc: 2026-09-27T12:57:55.416806+00:00 | thiet bi: cuda
- 3 seed x toi da 300 epoch (early stopping, patience 6) | hidden 64, 4 head, dropout 0.3, lr 0.005 | LayerNorm | diem hang xom: CO

| Nhom | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Toan bo dev | 6882 | 823 | 0.724 [0.703–0.744] | 0.323 [0.290–0.356] |
| CO anh | 1571 | 393 | 0.721 [0.692–0.750] | 0.474 [0.424–0.519] |
| KHONG anh | 5311 | 430 | 0.636 [0.610–0.664] | 0.134 [0.117–0.158] |

## Doi chieu voi Phase 3 (bootstrap ghep cap, cung benh nhan)

| So sanh | Nhom | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|---|
| GAT − B3b | khong_anh | -0.0042 | [-0.0238; +0.0168] | 32% | **TRUOT** |
| GAT − B3 | khong_anh | +0.0097 | [-0.0137; +0.0337] | 74% | **TRUOT** |
| GAT − B0 | khong_anh | +0.0178 | [-0.0027; +0.0369] | 96% | **TRUOT** |
| GAT − B2 | co_anh | -0.0225 | [-0.0435; -0.0012] | 2% | **TRUOT** |
| GAT − L2 | co_anh | -0.0332 | [-0.0566; -0.0080] | 1% | **TRUOT** |

*`phai_hon`: KTC phai nam tron ben duong thi GAT moi thuc su hon. `khong_kem`: chi truot khi GAT thua ro rang.*

# Phase 4 — GAT (khong diem hang xom) (nguon anh `control`)

- Sinh luc: 2026-09-27T12:59:28.036019+00:00 | thiet bi: cuda
- 3 seed x toi da 300 epoch (early stopping, patience 6) | hidden 64, 4 head, dropout 0.3, lr 0.005 | LayerNorm | diem hang xom: KHONG

| Nhom | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Toan bo dev | 6882 | 823 | 0.721 [0.700–0.740] | 0.324 [0.291–0.358] |
| CO anh | 1571 | 393 | 0.723 [0.695–0.753] | 0.484 [0.436–0.532] |
| KHONG anh | 5311 | 430 | 0.635 [0.608–0.663] | 0.128 [0.112–0.153] |

## Doi chieu voi Phase 3 (bootstrap ghep cap, cung benh nhan)

| So sanh | Nhom | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|---|
| GAT − B3b | khong_anh | -0.0053 | [-0.0269; +0.0148] | 30% | **TRUOT** |
| GAT − B3 | khong_anh | +0.0087 | [-0.0136; +0.0343] | 75% | **TRUOT** |
| GAT − B0 | khong_anh | +0.0168 | [-0.0018; +0.0381] | 96% | **TRUOT** |
| GAT − B2 | co_anh | -0.0201 | [-0.0402; +0.0010] | 3% | DAT |
| GAT − L2 | co_anh | -0.0308 | [-0.0523; -0.0079] | 0% | **TRUOT** |

*`phai_hon`: KTC phai nam tron ben duong thi GAT moi thuc su hon. `khong_kem`: chi truot khi GAT thua ro rang.*

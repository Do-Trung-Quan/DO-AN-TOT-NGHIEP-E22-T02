# Phase 4 — GAT (tat do thi) (nguon anh `control`)

- Sinh luc: 2026-09-27T12:58:35.931802+00:00 | thiet bi: cuda
- 3 seed x toi da 300 epoch (early stopping, patience 6) | hidden 64, 4 head, dropout 0.3, lr 0.005 | LayerNorm | diem hang xom: CO

| Nhom | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Toan bo dev | 6882 | 823 | 0.719 [0.697–0.739] | 0.321 [0.289–0.356] |
| CO anh | 1571 | 393 | 0.720 [0.691–0.749] | 0.480 [0.431–0.528] |
| KHONG anh | 5311 | 430 | 0.626 [0.598–0.654] | 0.137 [0.119–0.161] |

## Doi chieu voi Phase 3 (bootstrap ghep cap, cung benh nhan)

| So sanh | Nhom | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|---|
| GAT − B3b | khong_anh | -0.0141 | [-0.0355; +0.0073] | 9% | **TRUOT** |
| GAT − B3 | khong_anh | -0.0002 | [-0.0251; +0.0239] | 48% | **TRUOT** |
| GAT − B0 | khong_anh | +0.0079 | [-0.0134; +0.0298] | 76% | **TRUOT** |
| GAT − B2 | co_anh | -0.0232 | [-0.0447; +0.0015] | 4% | DAT |
| GAT − L2 | co_anh | -0.0339 | [-0.0564; -0.0084] | 0% | **TRUOT** |

*`phai_hon`: KTC phai nam tron ben duong thi GAT moi thuc su hon. `khong_kem`: chi truot khi GAT thua ro rang.*

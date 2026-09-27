# Phase 3 — baseline B0..B3  (nguon anh `control`)

- Sinh luc: 2026-09-27T11:39:14.459768+00:00 | thiet bi: cuda
- Dac trung lam sang: bang tho 139 cot + diem nhanh ts
- Tap dev 6882 node, 823 ca duong. Test 1148 node chua dung.
- 5 fold co san x 3 seed, 60 epoch; KTC 95% bootstrap theo benh nhan

## Toan bo dev

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 6882 | 823 | 0.633 [0.613–0.652] | 0.190 [0.173–0.213] |
| B1 — chi anh (node co anh) | 1571 | 393 | 0.749 [0.723–0.777] | 0.518 [0.467–0.567] |
| B2 — lam sang + anh | 6882 | 823 | 0.713 [0.693–0.733] | 0.320 [0.287–0.355] |
| B3 — + vector anh hang xom | 6882 | 823 | 0.711 [0.690–0.730] | 0.320 [0.289–0.354] |
| B3b — + diem nguy co hang xom | 6882 | 823 | 0.716 [0.695–0.736] | 0.317 [0.286–0.351] |
| L1 — hoi quy logistic, chi anh | 1571 | 393 | 0.760 [0.732–0.788] | 0.518 [0.466–0.569] |
| L2 — hoi quy logistic, lam sang+anh | 6882 | 823 | 0.729 [0.708–0.747] | 0.338 [0.305–0.374] |
| L3b — hoi quy logistic, + diem hang xom | 6882 | 823 | 0.727 [0.706–0.746] | 0.337 [0.304–0.372] |
| (tham chieu) nhanh ts goc | 6882 | 823 | 0.641 [0.620–0.662] | 0.218 [0.196–0.246] |

## Nhom CO anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 1571 | 393 | 0.645 [0.615–0.676] | 0.393 [0.347–0.443] |
| B2 — lam sang + anh | 1571 | 393 | 0.744 [0.717–0.771] | 0.506 [0.456–0.556] |
| B3 — + vector anh hang xom | 1571 | 393 | 0.733 [0.705–0.759] | 0.501 [0.451–0.547] |
| B3b — + diem nguy co hang xom | 1571 | 393 | 0.738 [0.711–0.764] | 0.499 [0.448–0.545] |
| L2 — hoi quy logistic, lam sang+anh | 1571 | 393 | 0.754 [0.728–0.780] | 0.516 [0.463–0.568] |
| L3b — hoi quy logistic, + diem hang xom | 1571 | 393 | 0.753 [0.726–0.779] | 0.515 [0.462–0.565] |
| (tham chieu) nhanh ts goc | 1571 | 393 | 0.643 [0.610–0.675] | 0.391 [0.345–0.441] |

## Nhom KHONG anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 5311 | 430 | 0.618 [0.593–0.645] | 0.115 [0.102–0.133] |
| B2 — lam sang + anh | 5311 | 430 | 0.632 [0.604–0.657] | 0.124 [0.108–0.147] |
| B3 — + vector anh hang xom | 5311 | 430 | 0.627 [0.599–0.653] | 0.124 [0.108–0.145] |
| B3b — + diem nguy co hang xom | 5311 | 430 | 0.640 [0.613–0.665] | 0.131 [0.115–0.154] |
| L2 — hoi quy logistic, lam sang+anh | 5311 | 430 | 0.651 [0.625–0.677] | 0.140 [0.122–0.164] |
| L3b — hoi quy logistic, + diem hang xom | 5311 | 430 | 0.648 [0.622–0.674] | 0.139 [0.121–0.163] |
| (tham chieu) nhanh ts goc | 5311 | 430 | 0.616 [0.588–0.644] | 0.128 [0.111–0.151] |

## Cong Phase 3 — xet bang khoang tin cay, khong so so tran

| Cong | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|
| Anh + lam sang co kem hon chi anh khong (nhom co anh) | -0.0050 | [-0.0225; +0.0127] | 28% | DAT |
| Hang xom co giup nhom KHONG anh khong (B3) | +0.0081 | [-0.0145; +0.0324] | 74% | **TRUOT** |
| Hang xom co giup nhom KHONG anh khong (B3b) | +0.0220 | [+0.0041; +0.0393] | 99% | DAT |
| Mo hinh sau co hon hoi quy logistic khong (nhom co anh) | -0.0107 | [-0.0291; +0.0068] | 12% | DAT |

*`phai_hon` = KTC phai nam tron ben duong. `khong_kem` = chi truot khi KTC nam tron ben am, tuc thua ro rang.*

**Moc cho Phase 4 (GNN phai vuot):** nhom KHONG anh **0.648** (L3b — hoi quy logistic, + diem hang xom) | nhom CO anh **0.754** (L2 — hoi quy logistic, lam sang+anh)

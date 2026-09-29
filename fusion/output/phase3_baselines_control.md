# Phase 3 — baseline B0..B3  (nguon anh `control`)

- Sinh luc: 2026-09-29T04:27:14.808611+00:00 | thiet bi: cpu
- Dac trung lam sang: bang tho 139 cot + diem nhanh ts
- Tap dev 6882 node, 823 ca duong. Test 1148 node chua dung.
- 5 fold co san x 3 seed, 60 epoch; KTC 95% bootstrap theo benh nhan

## Toan bo dev

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 6882 | 823 | 0.635 [0.615–0.654] | 0.190 [0.173–0.215] |
| B1 — chi anh (node co anh) | 1571 | 393 | 0.749 [0.722–0.777] | 0.520 [0.469–0.569] |
| B2 — lam sang + anh | 6882 | 823 | 0.706 [0.685–0.726] | 0.314 [0.282–0.349] |
| B3 — + vector anh hang xom | 6882 | 823 | 0.713 [0.692–0.732] | 0.319 [0.287–0.353] |
| B3b — + diem nguy co hang xom | 6882 | 823 | 0.711 [0.690–0.731] | 0.319 [0.286–0.353] |
| L1 — hoi quy logistic, chi anh | 1571 | 393 | 0.760 [0.732–0.788] | 0.518 [0.466–0.569] |
| L2 — hoi quy logistic, lam sang+anh | 6882 | 823 | 0.729 [0.708–0.747] | 0.338 [0.306–0.374] |
| L3b — hoi quy logistic, + diem hang xom | 6882 | 823 | 0.730 [0.709–0.748] | 0.338 [0.304–0.373] |
| (tham chieu) nhanh ts goc | 6882 | 823 | 0.641 [0.620–0.662] | 0.218 [0.196–0.246] |

## Nhom CO anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 1571 | 393 | 0.643 [0.613–0.674] | 0.385 [0.340–0.436] |
| B2 — lam sang + anh | 1571 | 393 | 0.737 [0.711–0.765] | 0.505 [0.456–0.555] |
| B3 — + vector anh hang xom | 1571 | 393 | 0.735 [0.708–0.762] | 0.508 [0.460–0.557] |
| B3b — + diem nguy co hang xom | 1571 | 393 | 0.740 [0.714–0.766] | 0.506 [0.457–0.553] |
| L2 — hoi quy logistic, lam sang+anh | 1571 | 393 | 0.754 [0.728–0.780] | 0.517 [0.464–0.568] |
| L3b — hoi quy logistic, + diem hang xom | 1571 | 393 | 0.753 [0.726–0.779] | 0.514 [0.463–0.566] |
| (tham chieu) nhanh ts goc | 1571 | 393 | 0.643 [0.610–0.675] | 0.391 [0.345–0.441] |

## Nhom KHONG anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 5311 | 430 | 0.622 [0.596–0.650] | 0.117 [0.104–0.135] |
| B2 — lam sang + anh | 5311 | 430 | 0.625 [0.598–0.651] | 0.118 [0.104–0.138] |
| B3 — + vector anh hang xom | 5311 | 430 | 0.633 [0.607–0.659] | 0.121 [0.107–0.139] |
| B3b — + diem nguy co hang xom | 5311 | 430 | 0.630 [0.603–0.656] | 0.127 [0.110–0.149] |
| L2 — hoi quy logistic, lam sang+anh | 5311 | 430 | 0.651 [0.625–0.677] | 0.139 [0.122–0.164] |
| L3b — hoi quy logistic, + diem hang xom | 5311 | 430 | 0.653 [0.628–0.679] | 0.140 [0.123–0.165] |
| (tham chieu) nhanh ts goc | 5311 | 430 | 0.616 [0.588–0.644] | 0.128 [0.111–0.151] |

## Cong Phase 3 — xet bang khoang tin cay, khong so so tran

| Cong | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|
| Anh + lam sang co kem hon chi anh khong (nhom co anh) | -0.0114 | [-0.0299; +0.0053] | 8% | DAT |
| Hang xom co giup nhom KHONG anh khong (B3) | +0.0111 | [-0.0108; +0.0321] | 84% | **TRUOT** |
| Hang xom co giup nhom KHONG anh khong (B3b) | +0.0080 | [-0.0104; +0.0249] | 82% | **TRUOT** |
| Mo hinh sau co hon hoi quy logistic khong (nhom co anh) | -0.0167 | [-0.0345; +0.0012] | 3% | DAT |

*`phai_hon` = KTC phai nam tron ben duong. `khong_kem` = chi truot khi KTC nam tron ben am, tuc thua ro rang.*

**Moc cho Phase 4 (GNN phai vuot):** nhom KHONG anh **0.653** (L3b — hoi quy logistic, + diem hang xom) | nhom CO anh **0.754** (L2 — hoi quy logistic, lam sang+anh)

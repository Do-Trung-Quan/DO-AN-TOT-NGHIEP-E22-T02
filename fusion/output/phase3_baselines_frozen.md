# Phase 3 — baseline B0..B3  (nguon anh `frozen`)

- Sinh luc: 2026-09-29T04:39:27.136732+00:00 | thiet bi: cpu
- Dac trung lam sang: bang tho 139 cot + diem nhanh ts
- Tap dev 6882 node, 823 ca duong. Test 1148 node chua dung.
- 5 fold co san x 3 seed, 60 epoch; KTC 95% bootstrap theo benh nhan

## Toan bo dev

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 6882 | 823 | 0.635 [0.615–0.654] | 0.190 [0.173–0.215] |
| B1 — chi anh (node co anh) | 1571 | 393 | 0.753 [0.727–0.780] | 0.528 [0.480–0.578] |
| B2 — lam sang + anh | 6882 | 823 | 0.692 [0.671–0.713] | 0.305 [0.274–0.338] |
| B3 — + vector anh hang xom | 6882 | 823 | 0.702 [0.680–0.723] | 0.303 [0.272–0.336] |
| B3b — + diem nguy co hang xom | 6882 | 823 | 0.702 [0.680–0.722] | 0.308 [0.277–0.344] |
| L1 — hoi quy logistic, chi anh | 1571 | 393 | 0.759 [0.732–0.785] | 0.516 [0.466–0.567] |
| L2 — hoi quy logistic, lam sang+anh | 6882 | 823 | 0.715 [0.694–0.735] | 0.330 [0.297–0.365] |
| L3b — hoi quy logistic, + diem hang xom | 6882 | 823 | 0.716 [0.695–0.736] | 0.329 [0.295–0.363] |
| (tham chieu) nhanh ts goc | 6882 | 823 | 0.641 [0.620–0.662] | 0.218 [0.196–0.246] |

## Nhom CO anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 1571 | 393 | 0.643 [0.613–0.674] | 0.385 [0.340–0.436] |
| B2 — lam sang + anh | 1571 | 393 | 0.739 [0.712–0.766] | 0.512 [0.463–0.560] |
| B3 — + vector anh hang xom | 1571 | 393 | 0.729 [0.701–0.757] | 0.491 [0.443–0.541] |
| B3b — + diem nguy co hang xom | 1571 | 393 | 0.748 [0.720–0.774] | 0.508 [0.461–0.558] |
| L2 — hoi quy logistic, lam sang+anh | 1571 | 393 | 0.747 [0.718–0.773] | 0.505 [0.455–0.551] |
| L3b — hoi quy logistic, + diem hang xom | 1571 | 393 | 0.746 [0.718–0.772] | 0.504 [0.454–0.549] |
| (tham chieu) nhanh ts goc | 1571 | 393 | 0.643 [0.610–0.675] | 0.391 [0.345–0.441] |

## Nhom KHONG anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 5311 | 430 | 0.622 [0.596–0.650] | 0.117 [0.104–0.135] |
| B2 — lam sang + anh | 5311 | 430 | 0.622 [0.595–0.647] | 0.118 [0.104–0.137] |
| B3 — + vector anh hang xom | 5311 | 430 | 0.634 [0.607–0.661] | 0.129 [0.112–0.152] |
| B3b — + diem nguy co hang xom | 5311 | 430 | 0.630 [0.601–0.656] | 0.126 [0.110–0.148] |
| L2 — hoi quy logistic, lam sang+anh | 5311 | 430 | 0.647 [0.622–0.674] | 0.139 [0.121–0.164] |
| L3b — hoi quy logistic, + diem hang xom | 5311 | 430 | 0.649 [0.623–0.676] | 0.141 [0.122–0.166] |
| (tham chieu) nhanh ts goc | 5311 | 430 | 0.616 [0.588–0.644] | 0.128 [0.111–0.151] |

## Cong Phase 3 — xet bang khoang tin cay, khong so so tran

| Cong | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|
| Anh + lam sang co kem hon chi anh khong (nhom co anh) | -0.0134 | [-0.0313; +0.0038] | 7% | DAT |
| Hang xom co giup nhom KHONG anh khong (B3) | +0.0112 | [-0.0146; +0.0349] | 81% | **TRUOT** |
| Hang xom co giup nhom KHONG anh khong (B3b) | +0.0071 | [-0.0114; +0.0231] | 78% | **TRUOT** |
| Mo hinh sau co hon hoi quy logistic khong (nhom co anh) | -0.0081 | [-0.0263; +0.0091] | 18% | DAT |

*`phai_hon` = KTC phai nam tron ben duong. `khong_kem` = chi truot khi KTC nam tron ben am, tuc thua ro rang.*

**Moc cho Phase 4 (GNN phai vuot):** nhom KHONG anh **0.649** (L3b — hoi quy logistic, + diem hang xom) | nhom CO anh **0.753** (B1 — chi anh (node co anh))

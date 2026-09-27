# Phase 3 — baseline B0..B3  (nguon anh `frozen`)

- Sinh luc: 2026-09-27T11:45:46.783638+00:00 | thiet bi: cuda
- Dac trung lam sang: bang tho 139 cot + diem nhanh ts
- Tap dev 6882 node, 823 ca duong. Test 1148 node chua dung.
- 5 fold co san x 3 seed, 60 epoch; KTC 95% bootstrap theo benh nhan

## Toan bo dev

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 6882 | 823 | 0.633 [0.613–0.652] | 0.190 [0.173–0.213] |
| B1 — chi anh (node co anh) | 1571 | 393 | 0.753 [0.728–0.781] | 0.530 [0.482–0.579] |
| B2 — lam sang + anh | 6882 | 823 | 0.697 [0.677–0.718] | 0.307 [0.275–0.341] |
| B3 — + vector anh hang xom | 6882 | 823 | 0.690 [0.669–0.712] | 0.304 [0.272–0.339] |
| B3b — + diem nguy co hang xom | 6882 | 823 | 0.702 [0.680–0.722] | 0.310 [0.279–0.344] |
| L1 — hoi quy logistic, chi anh | 1571 | 393 | 0.759 [0.732–0.785] | 0.517 [0.466–0.568] |
| L2 — hoi quy logistic, lam sang+anh | 6882 | 823 | 0.716 [0.694–0.735] | 0.330 [0.297–0.365] |
| L3b — hoi quy logistic, + diem hang xom | 6882 | 823 | 0.714 [0.692–0.734] | 0.329 [0.295–0.363] |
| (tham chieu) nhanh ts goc | 6882 | 823 | 0.641 [0.620–0.662] | 0.218 [0.196–0.246] |

## Nhom CO anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 1571 | 393 | 0.645 [0.615–0.676] | 0.393 [0.347–0.443] |
| B2 — lam sang + anh | 1571 | 393 | 0.742 [0.714–0.768] | 0.504 [0.457–0.557] |
| B3 — + vector anh hang xom | 1571 | 393 | 0.740 [0.712–0.767] | 0.512 [0.462–0.566] |
| B3b — + diem nguy co hang xom | 1571 | 393 | 0.747 [0.719–0.774] | 0.510 [0.462–0.561] |
| L2 — hoi quy logistic, lam sang+anh | 1571 | 393 | 0.747 [0.718–0.774] | 0.506 [0.456–0.552] |
| L3b — hoi quy logistic, + diem hang xom | 1571 | 393 | 0.746 [0.717–0.772] | 0.504 [0.454–0.550] |
| (tham chieu) nhanh ts goc | 1571 | 393 | 0.643 [0.610–0.675] | 0.391 [0.345–0.441] |

## Nhom KHONG anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 5311 | 430 | 0.618 [0.593–0.645] | 0.115 [0.102–0.133] |
| B2 — lam sang + anh | 5311 | 430 | 0.629 [0.602–0.654] | 0.122 [0.106–0.142] |
| B3 — + vector anh hang xom | 5311 | 430 | 0.603 [0.577–0.631] | 0.109 [0.096–0.129] |
| B3b — + diem nguy co hang xom | 5311 | 430 | 0.634 [0.605–0.659] | 0.125 [0.109–0.146] |
| L2 — hoi quy logistic, lam sang+anh | 5311 | 430 | 0.647 [0.622–0.674] | 0.139 [0.121–0.164] |
| L3b — hoi quy logistic, + diem hang xom | 5311 | 430 | 0.646 [0.620–0.673] | 0.139 [0.121–0.164] |
| (tham chieu) nhanh ts goc | 5311 | 430 | 0.616 [0.588–0.644] | 0.128 [0.111–0.151] |

## Cong Phase 3 — xet bang khoang tin cay, khong so so tran

| Cong | Chenh lech ROC | KTC 95% | P | Ket qua |
|---|---|---|---|---|
| Anh + lam sang co kem hon chi anh khong (nhom co anh) | -0.0112 | [-0.0298; +0.0049] | 9% | DAT |
| Hang xom co giup nhom KHONG anh khong (B3) | -0.0155 | [-0.0451; +0.0062] | 9% | **TRUOT** |
| Hang xom co giup nhom KHONG anh khong (B3b) | +0.0153 | [-0.0036; +0.0325] | 96% | **TRUOT** |
| Mo hinh sau co hon hoi quy logistic khong (nhom co anh) | -0.0057 | [-0.0242; +0.0119] | 24% | DAT |

*`phai_hon` = KTC phai nam tron ben duong. `khong_kem` = chi truot khi KTC nam tron ben am, tuc thua ro rang.*

**Moc cho Phase 4 (GNN phai vuot):** nhom KHONG anh **0.646** (L3b — hoi quy logistic, + diem hang xom) | nhom CO anh **0.753** (B1 — chi anh (node co anh))

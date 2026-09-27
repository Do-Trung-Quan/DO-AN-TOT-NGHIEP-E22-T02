# Phase 3 — baseline B0..B3  (nguon anh `control`)

- Sinh luc: 2026-09-27T10:35:43.592136+00:00 | thiet bi: cuda
- Dac trung lam sang: bang tho 139 cot + diem nhanh ts
- Tap dev 6882 node, 823 ca duong. Test 1148 node chua dung.
- 5 fold co san x 3 seed, 60 epoch; KTC 95% bootstrap theo benh nhan

## Toan bo dev

| Mo hinh                        | n    | duong | ROC-AUC             | PR-AUC              |
| --------------------------------| ------| -------| ---------------------| ---------------------|
| B0 — chi lam sang              | 6882 | 823   | 0.633 [0.613–0.652] | 0.190 [0.173–0.213] |
| B1 — chi anh (node co anh)     | 1571 | 393   | 0.749 [0.723–0.777] | 0.518 [0.467–0.567] |
| B2 — lam sang + anh            | 6882 | 823   | 0.713 [0.693–0.733] | 0.320 [0.287–0.355] |
| B3 — lam sang + anh + hang xom | 6882 | 823   | 0.719 [0.699–0.739] | 0.315 [0.284–0.348] |
| (tham chieu) nhanh ts goc      | 6882 | 823   | 0.641 [0.620–0.662] | 0.218 [0.196–0.246] |

## Nhom CO anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 1571 | 393 | 0.645 [0.615–0.676] | 0.393 [0.347–0.443] |
| B2 — lam sang + anh | 1571 | 393 | 0.744 [0.717–0.771] | 0.506 [0.456–0.556] |
| B3 — lam sang + anh + hang xom | 1571 | 393 | 0.729 [0.701–0.756] | 0.483 [0.433–0.530] |
| (tham chieu) nhanh ts goc | 1571 | 393 | 0.643 [0.610–0.675] | 0.391 [0.345–0.441] |

## Nhom KHONG anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 5311 | 430 | 0.618 [0.593–0.645] | 0.115 [0.102–0.133] |
| B2 — lam sang + anh | 5311 | 430 | 0.632 [0.604–0.657] | 0.124 [0.108–0.147] |
| B3 — lam sang + anh + hang xom | 5311 | 430 | 0.645 [0.619–0.669] | 0.134 [0.117–0.157] |
| (tham chieu) nhanh ts goc | 5311 | 430 | 0.616 [0.588–0.644] | 0.128 [0.111–0.151] |

## Cong Phase 3

| Cong | Do duoc | Ket qua |
|---|---|---|
| B2 >= B1 tren nhom co anh | 0.744 vs 0.749 | **TRUOT** |
| B3 > B0 tren nhom khong anh | 0.645 vs 0.618 | DAT |

**Moc cho Phase 4:** GNN phai vuot B3 o nhom khong anh (**0.645**) va vuot B2 o nhom co anh (**0.744**).

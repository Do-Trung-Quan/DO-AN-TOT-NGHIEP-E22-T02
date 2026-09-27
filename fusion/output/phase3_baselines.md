# Phase 3 — baseline B0..B3  (nguon anh `control`)

- Sinh luc: 2026-09-27T10:01:15.369759+00:00 | thiet bi: cpu
- Dac trung lam sang: bang tho 139 cot + diem nhanh ts
- Tap dev 6882 node, 823 ca duong. Test 1148 node chua dung.
- 5 fold co san x 1 seed, 20 epoch; KTC 95% bootstrap theo benh nhan

## Toan bo dev

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 6882 | 823 | 0.648 [0.627–0.667] | 0.203 [0.184–0.227] |
| B1 — chi anh (node co anh) | 1571 | 393 | 0.749 [0.722–0.776] | 0.529 [0.477–0.579] |
| B2 — lam sang + anh | 6882 | 823 | 0.722 [0.701–0.741] | 0.333 [0.301–0.368] |
| B3 — lam sang + anh + hang xom | 6882 | 823 | 0.721 [0.702–0.741] | 0.335 [0.303–0.370] |
| (tham chieu) nhanh ts goc | 6882 | 823 | 0.641 [0.620–0.662] | 0.218 [0.196–0.246] |

## Nhom CO anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 1571 | 393 | 0.663 [0.633–0.693] | 0.419 [0.374–0.469] |
| B2 — lam sang + anh | 1571 | 393 | 0.742 [0.717–0.769] | 0.510 [0.460–0.559] |
| B3 — lam sang + anh + hang xom | 1571 | 393 | 0.731 [0.704–0.757] | 0.509 [0.459–0.554] |
| (tham chieu) nhanh ts goc | 1571 | 393 | 0.643 [0.610–0.675] | 0.391 [0.345–0.441] |

## Nhom KHONG anh

| Mo hinh | n | duong | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| B0 — chi lam sang | 5311 | 430 | 0.625 [0.601–0.652] | 0.119 [0.105–0.137] |
| B2 — lam sang + anh | 5311 | 430 | 0.635 [0.608–0.662] | 0.132 [0.115–0.156] |
| B3 — lam sang + anh + hang xom | 5311 | 430 | 0.639 [0.613–0.665] | 0.137 [0.118–0.164] |
| (tham chieu) nhanh ts goc | 5311 | 430 | 0.616 [0.588–0.644] | 0.128 [0.111–0.151] |

## Cong Phase 3

| Cong | Do duoc | Ket qua |
|---|---|---|
| B2 >= B1 tren nhom co anh | 0.742 vs 0.749 | **TRUOT** |
| B3 > B0 tren nhom khong anh | 0.639 vs 0.625 | DAT |

**Moc cho Phase 4:** GNN phai vuot B3 o nhom khong anh (**0.639**) va vuot B2 o nhom co anh (**0.742**).

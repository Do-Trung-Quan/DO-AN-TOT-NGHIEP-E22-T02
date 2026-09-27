# Nghiem thu Phase 1+2 — nguon `frozen`, do thi `graph_ketqua_k10_bridge`

- Sinh luc: 2026-09-27T09:03:37.271781+00:00
- Bo du lieu: `fusion_dataset_ketqua_frozen.npz`
- Do thi   : `graph_ketqua_k10_bridge.npz`
- **Ket luan: DAT — duoc phep sang Phase 3**

| # | Chi tieu | Gia tri | Nguong | Ket qua |
|---|---|---|---|---|
| 1 | So node | `8030/8030` | = 8030 | DAT |
| 2 | Node co anh | `1835/1835` | = 1835 | DAT |
| 3 | NaN/Inf o hang CO anh | `0` | = 0 | DAT |
| 4 | Chieu hang so (float64) | `0/256` | = 0 | DAT |
| 5 | Node khong anh = NaN (khong dien 0) | `True` | True | DAT |
| 6 | Benh nhan bac cau dev/test | `0` | = 0 | DAT |
| 7 | Homophily vuot moc ngau nhien | `0.8061 vs 0.7893 (+0.0168)` | > 0 | DAT |
| 8 | Homophily lop duong | `0.2075 (lift 1.73x)` | bao cao | — |
| 9 | Node duong co >=1 hang xom duong | `89.2%` | bao cao | — |
| 10 | Isolated-node coverage | `100.0%` | bao cao | — |
| 11 | Cross-modality edge ratio | `0.3217` | bao cao | — |
| 12 | Bac trung binh | `15.04 (p10/p50/p90 = 10/14/22)` | bao cao | — |
| 13 | Thanh phan lien thong | `1 (lon nhat 8,030)` | bao cao | — |
| 14 | Ty le benh co anh / khong anh | `25.18% vs 8.05% (gap 3.13 lan)` | bao cao | — |

## Doc chi so 8 — vi sao homophily lop duong moi la thuoc do that

Du lieu lech 1:37 nen homophily tho luon cao san (0.8061) chi vi
97,4% node la am — no khong cho biet do thi co giup phat hien BENH hay khong.

Chi so can nhin: mot benh nhan mac benh thi **20.8%** hang xom
gan nhat cua ho cung mac benh, so voi **11.97%** ky vong ngau nhien
— **lift 1.73x**. Do thi rat giau thong tin cho lop thieu so.

## Doc chi so 10 — cau noi modality

Isolated-node coverage = **100.0%**: ty le node KHONG co anh
ma cham duoc it nhat mot hang xom CO anh. Day chinh la kenh de 6.195 benh nhan
khong co phim X-quang 'hoc ke' thong tin anh — co che trung tam cua do an.

## Doc chi so 14 — shortcut `has_image`

Nhom co anh co ty le benh cao gap **3.13 lan** nhom khong anh.
Model co the hoc 'co anh => nguy co cao' ma khong can nhin noi dung phim.
Phase 4 chan bang 3 lop: missing-token hoc duoc, modality dropout, va ablation
tat co `has_image`.

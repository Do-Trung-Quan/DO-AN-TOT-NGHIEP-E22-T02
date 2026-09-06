# Bao cao kiem dinh dac trung anh (QC) — bo FROZEN

- Sinh luc: 2026-09-05T05:21:04.747192+00:00
- Nguon dac trung: `image_features.parquet`
- Chi muc: `image_index.parquet`
- **Ket luan: TRUOT — TU CHOI ban giao**

| # | Chi tieu | Gia tri | Nguong | Ket qua |
|---|---|---|---|---|
| 1 | So hang khop chi muc | `109/1835` | = 1835 | **TRUOT** |
| 2 | Chieu hang so (float64) | `0/256` | = 0 | DAT |
| 3 | Benh nhan nam o >1 fold | `0` | = 0 | DAT |
| 4 | NaN/Inf trong hang co anh | `0` | = 0 | DAT |
| 5 | Ten file / ho ten trong artifact | `0` | = 0 | DAT |
| 6 | Cosine TB (raw) | `0.8963` | bao cao | — |
| 6 | Cosine TB (da mean-center) | `-0.0019` | bao cao | — |
| 7 | Effective rank | `10.67/256` | bao cao | — |
| 7 | Phuong sai PC1 | `40.8%` | bao cao | — |
| 8 | PR-AUC probe tren nhan bnn | `0.2096 (nen 0.0917)` | bao cao | — |
| 9 | Do chinh xac doan lo chup | `khong du lo` | bao cao | — |

## Cac muc TRUOT

- 1. So hang khop chi muc = 109/1835, can = 1835

## Doc chi so 6 — quan trong cho phase fusion

Cosine trung binh raw = **0.8963**, sau mean-centering = **-0.0019**.

Embedding sau co mot thanh phan trung binh chung rat lon (anisotropy).
Bo 256 chieu chet **khong** xu ly duoc dieu nay — do thuc te cho thay
cosine 512 chieu (0.759) va 256 chieu song (0.755) gan nhu bang nhau.

> **Bat buoc voi ben fusion:** phai mean-center hoac standardize truoc khi
> dung do thi kNN. Neu khong, moi anh se 'giong nhau ~76%' va do thi
> gan nhu vo nghia.

## Doc chi so 9 — nhieu loan theo lo chup

Ty le duong theo ngay chup lech 8 lan (20191115 = 5.5% ... 20181223 = 44.8%,
ca biet 20181224 = 58.7%). Embedding ma hoa manh thong tin may chup / dot kham.
Khi bao cao ket qua fusion nen tach metric theo lo.

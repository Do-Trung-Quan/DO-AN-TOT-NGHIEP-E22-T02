# Baseline nhan `ketqua` — hai nhanh va uoc luong fusion

- Sinh luc: 2026-09-27T09:04:13.406467+00:00
- Tap danh gia: **1835 nguoi co anh**, 462 ca bui phoi, ty le 0.252 (bai bao: 1633 nguoi, 462 ca, ty le 0,283)
- Du doan ngoai mau gop tu 5 fold; KTC 95% bootstrap theo benh nhan (1000 lan); nguong chon tren 4 fold, ap cho fold con lai
- Nhanh TS: `output/ketqua/ts_predictions.parquet` — train tren 8030 nguoi

## 1. Ket qua do an (1835 nguoi co anh)

| Mo hinh | ROC-AUC | PR-AUC | Accuracy | macro-F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| TS — Keras nhanh timeseries | 0.641 [0.608–0.669] | 0.388 [0.348–0.435] | 0.675 [0.654–0.695] | 0.576 [0.551–0.600] | 0.381 [0.335–0.424] | 0.774 [0.752–0.796] |
| ANH — control (probe) | 0.761 [0.736–0.786] | 0.544 [0.496–0.591] | 0.744 [0.723–0.764] | 0.670 [0.645–0.695] | 0.537 [0.490–0.583] | 0.814 [0.792–0.834] |
| ANH — frozen (probe, doi chung) | 0.759 [0.733–0.784] | 0.533 [0.485–0.580] | 0.774 [0.754–0.794] | 0.679 [0.653–0.705] | 0.457 [0.409–0.503] | 0.881 [0.864–0.898] |
| FUSION muon — TS + ANH control | 0.766 [0.739–0.791] | 0.551 [0.502–0.597] | 0.767 [0.748–0.787] | 0.684 [0.658–0.708] | 0.502 [0.456–0.548] | 0.857 [0.838–0.875] |

## 2. So bai bao cong bo (Table 5 + 6, khong co PR-AUC)

| Mo hinh | ROC-AUC | Accuracy | macro-F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| DenseNet201 (chi anh) | 0.96 | 0.85 | 0.70 | 0.78 | 0.87 |
| ConvNeXtLarge (chi anh) | 0.98 | 0.88 | 0.72 | 0.83 | 0.89 |
| Qwen2-VL (chi text) | 0.94 | 0.78 | 0.58 | 0.71 | 0.75 |
| Qwen2-VL (anh + text) | 0.96 | 0.86 | 0.70 | 0.81 | 0.87 |
| SiCLIP k=3 (anh + text) | 0.98 | 0.94 | 0.85 | 0.91 | 0.93 |

## 3. Fusion co them duoc gi so voi tung nhanh

| So sanh | Chi so | Chenh lech | KTC 95% | P(fusion tot hon) |
|---|---|---|---|---|
| FUSION − ANH control | ROC-AUC | +0.005 | [-0.004; +0.013] | 86% |
| FUSION − ANH control | PR-AUC | +0.007 | [-0.008; +0.021] | 81% |
| FUSION − TS | ROC-AUC | +0.125 | [+0.096; +0.155] | 100% |
| FUSION − TS | PR-AUC | +0.163 | [+0.120; +0.199] | 100% |

## 4. Nhanh TS theo tung tap (cohort train: 8030 nguoi)

| Tap | n | Ca duong | Ty le | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| dev (OOF) | 6882 | 823 | 0.120 | 0.641 | 0.218 |
| test (hold-out) | 1148 | 138 | 0.120 | 0.647 | 0.223 |
| co anh (dev+test) | 1835 | 462 | 0.252 | 0.641 | 0.388 |
| KHONG anh (dev+test) | 6195 | 499 | 0.081 | 0.617 | 0.127 |

## Luu y khi doc

- Bai bao loai 202 ca am (1373 -> 1171) theo tieu chi o Phu luc bang 2 (chua co); do an giu nguyen 1835 phim cua 1748 nguoi.
- ANH la probe tuyen tinh tren vector control. Chi tiet hai bo dac trung anh: imagefeat/output/eval_ketqua.md
- FUSION muon chi la moc duoi — cach ket hop don gian nhat co the.
- Accuracy, macro-F1 phu thuoc ty le duong: bai bao 0,283 vs do an 0.252. ROC-AUC khong phu thuoc ty le, la chi so so sanh cong bang nhat.

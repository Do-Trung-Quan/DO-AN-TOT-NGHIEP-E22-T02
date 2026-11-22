# Tuần 7 — Xuất mô hình PG đã khoá cho web (học lại, lưu trọng số, kiểm tái tạo)

- Script `scripts_v3/v3_xuat_pg.py`; cấu hình từng fold lấy từ `selected_PG_nghe10.json` (không dò lại).
- Mô hình lưu ở `cache_v3/mo_hinh_PG_web_smoke/` (riêng tư). Thiết bị cuda. Thời gian 24 s.
- Phần test chỉ so điểm mới với điểm đã lưu ngày 07/11, **không dùng nhãn test, không tính chỉ số test**.

## Theo fold

| fold | cau_hinh | epoch | epoch_noi_bo_moi | epoch_noi_bo_khop | lech_oof_max | giay |
|---|---|---|---|---|---|---|
| 0 | 1 | 5 | 5 | True | 0.0000 | 23 |

## So điểm mới với điểm đã lưu

| tap | nhom | n | lech_tb | lech_max | pearson | spearman | auc_moi | auc_cu |
|---|---|---|---|---|---|---|---|---|
| dev OOF (oof_cal) | có phim | 299 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.7923 | 0.7923 |
| dev OOF (oof_cal) | không phim | 1038 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6956 | 0.6956 |

AUC trong lô dev (có phim): mới 0.7721, cũ 0.7721.

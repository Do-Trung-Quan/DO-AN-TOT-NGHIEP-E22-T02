# Tuần 8 — Xuất mô hình PG-noctx (không dùng nút tỉnh) cho web

- Script `scripts_v3/v3_xuat_pg.py --ten PG-noctx`; cấu hình từng fold lấy từ `selected_PG-noctx_nghe10.json` (không dò lại).
- Mô hình lưu ở `cache_v3/mo_hinh_PG_noctx_web/` (riêng tư). Thiết bị cuda. Thời gian 109 s.
- Phần test chỉ so điểm mới với điểm đã lưu, **không dùng nhãn test, không tính chỉ số test**.

## Theo fold

| fold | cau_hinh | epoch | epoch_noi_bo_moi | epoch_noi_bo_khop | lech_oof_max | giay |
|---|---|---|---|---|---|---|
| 0 | 27 | 3 | 3 | True | 0.0000 | 16 |
| 1 | 21 | 7 | 7 | True | 0.0000 | 34 |
| 2 | 0 | 18 | 18 | True | 0.0000 | 70 |
| 3 | 15 | 8 | 8 | True | 0.0000 | 90 |
| 4 | 27 | 5 | 5 | True | 0.0000 | 108 |

## So điểm mới với điểm đã lưu

| tap | nhom | n | lech_tb | lech_max | pearson | spearman | auc_moi | auc_cu |
|---|---|---|---|---|---|---|---|---|
| dev OOF (oof_cal) | có phim | 1500 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.7598 | 0.7598 |
| test (test_cal), KHONG dung nhan | có phim | 264 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | nan | nan |
| dev OOF (oof_cal) | không phim | 5183 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6303 | 0.6303 |
| test (test_cal), KHONG dung nhan | không phim | 884 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | nan | nan |

AUC trong lô dev (có phim): mới 0.7133, cũ 0.7133.

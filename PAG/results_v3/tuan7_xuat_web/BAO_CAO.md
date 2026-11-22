# Tuần 7 — Xuất mô hình PG đã khoá cho web (học lại, lưu trọng số, kiểm tái tạo)

- Script `scripts_v3/v3_xuat_pg.py`; cấu hình từng fold lấy từ `selected_PG_nghe10.json` (không dò lại).
- Mô hình lưu ở `cache_v3/mo_hinh_PG_web/` (riêng tư). Thiết bị cuda. Thời gian 119 s.
- Phần test chỉ so điểm mới với điểm đã lưu ngày 07/11, **không dùng nhãn test, không tính chỉ số test**.

## Theo fold

| fold | cau_hinh | epoch | epoch_noi_bo_moi | epoch_noi_bo_khop | lech_oof_max | giay |
|---|---|---|---|---|---|---|
| 0 | 1 | 5 | 5 | True | 0.0000 | 20 |
| 1 | 21 | 7 | 7 | True | 0.0000 | 42 |
| 2 | 13 | 10 | 10 | True | 0.0000 | 72 |
| 3 | 23 | 6 | 6 | True | 0.0000 | 93 |
| 4 | 14 | 9 | 9 | True | 0.0000 | 118 |

## So điểm mới với điểm đã lưu

| tap | nhom | n | lech_tb | lech_max | pearson | spearman | auc_moi | auc_cu |
|---|---|---|---|---|---|---|---|---|
| dev OOF (oof_cal) | có phim | 1500 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.7680 | 0.7680 |
| test (test_cal), KHONG dung nhan | có phim | 264 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | nan | nan |
| dev OOF (oof_cal) | không phim | 5183 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6551 | 0.6551 |
| test (test_cal), KHONG dung nhan | không phim | 884 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | nan | nan |

AUC trong lô dev (có phim): mới 0.7163, cũ 0.7163.

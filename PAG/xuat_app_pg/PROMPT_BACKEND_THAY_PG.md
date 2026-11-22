# Prompt cho phiên backend: bỏ New Way, chỉ dùng mô hình PG

Phiên làm việc ở `E:\Hoc hanh\silicosis-backend` (theo `Claude.md` của repo đó).

---

**Mục tiêu:** bỏ hẳn mô hình New Way. Mọi lượt chẩn đoán chỉ chạy mô hình PG. User đã xác nhận dữ liệu trong DB chỉ là
dữ liệu thử, xoá được. Chỉ làm backend (FastAPI `predict/` + Spring Boot); **không** đụng web `sillicosis-web` trong
phiên này, cuối phiên cập nhật `docs/api-predictions.md` để phía web nối sau.

**Hiện trạng (đã kiểm, xem lại trong code):**
- FastAPI đã có `POST /predict/pg` (commit `c55e5f4`): multipart `lam_sang` (chuỗi JSON, bắt buộc `tinh`) + `file`
  (PNG/JPEG, tuỳ chọn). Trả `PGResult` (`predict/app/schemas.py`): `xac_suat`, `xac_suat_theo_fold`, `co_anh`, `tinh`,
  `chu_y_vung`, `qa_anh`, `canh_bao`, `canh_bao_dau_vao`, `disclaimer`, `experimental`, `model_name`, `model_version`.
  **Không có ngưỡng, không có kết luận dương/âm** (ngưỡng PG đang chờ thầy duyệt — không tự đặt).
- Spring Boot: `prediction/PredictClient.java` mới gọi `/predict/clinical` và `/predict/clinical-image` (New Way).
  Bảng `predictions` (V4) có các cột New Way `p_lam_sang`, `nguong_lam_sang`, `bao_nguy_co_lam_sang`, `p_anh`, `p_ghep`,
  `nguong_ghep`, `bao_nguy_co_ghep`, `ket_luan_cuoi` và ràng buộc `predictions_result_check` bắt buộc chúng khi SUCCESS.

**Việc cần làm (đưa kế hoạch ngắn cho user duyệt trước khi code):**
1. **FastAPI:** bỏ `app/new_way.py`, `ml/new_way/`, endpoint `/predict/clinical`, `/predict/clinical-image`,
   `/schema`, `/health` của New Way (hoặc cho `/health`, `/schema` trỏ sang PG — chọn một, ghi rõ). Không sửa logic
   các file trong `predict/ml/pg/` (đã kiểm khớp mô hình gốc). Sửa/xoá test New Way, giữ test PG.
2. **Migration V7** (Flyway, không sửa V1–V6): xoá dòng cũ của `predictions` (dữ liệu thử), bỏ 8 cột New Way,
   thêm `xac_suat` (0–1), `xac_suat_theo_fold` (JSONB), `chu_y_vung` (JSONB, null khi không có ảnh), `canh_bao` (JSONB
   danh sách chuỗi); giữ `co_anh`, `model_name`, `model_version`, các cột ảnh, `lam_sang`, `exam_date`, `user_id`.
   Viết lại `predictions_result_check`: SUCCESS ⇒ `xac_suat`, `co_anh`, `model_version` có, `error_code` null,
   `co_anh = (image_storage_key IS NOT NULL)`; FAILED ⇒ có `error_code`, `xac_suat` null. Tệp ảnh cũ trên đĩa không tự
   xoá — ghi chú trong migration như V4.
3. **Java:** `PredictClient` chỉ gọi `/predict/pg` (multipart `lam_sang` + `file`). Entity `Prediction`,
   `PredictionResponse`, `PredictionService` đổi sang trường PG; bỏ mọi chỗ dùng ngưỡng / `ketLuanCuoi`.
   `ClinicalValidator`: `tinh` bắt buộc, chỉ nhận `Binh Dinh`, `Dong Nai`, `Hai Duong`, `Phu Yen`, `Thai Nguyen`
   → 422 với lỗi trên trường `tinh`. Lỗi 422/503 của FastAPI chuyển tiếp như hiện nay (FAILED + `error_code`).
   Retry lượt FAILED vẫn hoạt động.
4. **Test:** `./mvnw test` + pytest `predict/tests`. Ca chỉ lâm sàng, ca có ảnh, thiếu/sai `tinh` → 422, mô hình chưa
   nạp → 503/FAILED, retry. Dữ liệu giả tự đặt, không dùng dữ liệu bệnh nhân thật.
5. **Tài liệu:** cập nhật `docs/api-predictions.md` (shape mới của `Prediction`, trường `tinh`, không có nhãn/ngưỡng)
   và README `predict/`.

**Quy tắc:** trả lời tiếng Việt; không đọc `.env`/`.env.local`; không in/log dữ liệu bệnh nhân; **không commit, không
push — user tự làm git**; báo lại file đã thêm/sửa và kết quả test thật.

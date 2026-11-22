# Prompt cho phiên backend: thêm mô hình PG vào service dự đoán

Dán nguyên phần dưới vào phiên Claude Code mở ở `E:\Hoc hanh\silicosis-backend`.

---

Thêm mô hình thứ hai **PG** vào service FastAPI `predict/`, chạy song song với New Way (không thay, không sửa hành vi New Way).

**Nguồn (chỉ đọc, không sửa):** `E:\Hoc hanh\ĐÒ ÁN\Fusion\DO-AN-TOT-NGHIEP-E22-T02\PAG\xuat_app_pg\`
- Đọc `MO_TA_API.md` trước: nó mô tả cách gọi, kết quả, cài đặt và bảng mọi trường đầu vào cùng giá trị hợp lệ.
- Mã: `du_doan_pg.py` (lớp `MoHinhPG`, hàm `du_doan(lam_sang: dict, anh=None, duong_dan_biovil=None)`), `mo_hinh_pg.py`,
  `nghe.py`, `xu_ly_anh.py`, `requirements.txt`. Không sửa logic các file này (đã kiểm khớp với mô hình gốc tới 3,6e-7).
- Trọng số: `E:\Hoc hanh\ĐÒ ÁN\Fusion\DO-AN-TOT-NGHIEP-E22-T02\PAG\cache_v3\mo_hinh_PG_web\` (~23 MB, dữ liệu riêng tư).
- BioViL-T: `E:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Pre-train BioViL-T\biovil_t_image_model_proj_size_128.pt`.

**Việc cần làm:**
1. Chép mã vào `predict/ml/pg/` và trọng số vào `predict/ml/pg/mo_hinh_PG_web/`, theo đúng cách New Way đang làm với
   `ml/new_way/mo_hinh` (kể cả `.gitignore`: trọng số không lên GitHub công khai). Không commit, không push — user tự làm git.
2. `app/pg.py` bọc `MoHinhPG` giống `app/new_way.py`: nạp một lần trong `lifespan`, nạp lỗi thì endpoint PG trả 503 còn New Way vẫn chạy;
   thêm vào `Settings`: `pg_dir`, `biovil_path`. Gọi mô hình trong threadpool và có khoá (một `MoHinhPG` không an toàn khi gọi song song).
3. Endpoint mới, cùng kiểu với endpoint New Way hiện có (multipart: `clinical` JSON + `image` tuỳ chọn), ví dụ `POST /predict/pg`;
   dùng lại bộ kiểm ảnh, giới hạn kích thước, định dạng lỗi `{code, message, errors}` sẵn có.
4. Kiểm đầu vào: `tinh` **bắt buộc** và phải là một trong `Binh Dinh`, `Dong Nai`, `Hai Duong`, `Phu Yen`, `Thai Nguyen`
   → nếu không, trả 422 với lỗi trên trường `tinh`. Các trường khác: đúng tên cột và giá trị trong bảng mục 4 của `MO_TA_API.md`;
   trường lạ thì bỏ qua và báo trong `errors`/cảnh báo, không làm sập. `bmi`, `fev1_fvc`, `*_lv` do mô hình tự tính — không cần gửi.
5. Trả về: `xac_suat`, `xac_suat_theo_fold`, `co_anh`, `tinh`, `chu_y_vung`, `qa_anh`, `canh_bao`, `disclaimer`, `experimental`,
   tên/phiên bản mô hình. **Không có ngưỡng**: không trả nhãn dương/âm, không đặt ngưỡng mặc định (ngưỡng chờ duyệt).
6. Test (pytest) như phần New Way: nạp mô hình, ca chỉ lâm sàng, ca có ảnh, tỉnh sai → 422, ảnh hỏng → 400, mô hình chưa nạp → 503.
   Dùng dữ liệu giả tự đặt (không dùng dữ liệu bệnh nhân thật).
7. Viết đoạn hướng dẫn cho phía web (đặt trong README của `predict/`): form PG có **ô chọn tỉnh 5 giá trị** (hiển thị có dấu:
   Bình Định, Đồng Nai, Hải Dương, Phú Yên, Thái Nguyên; gửi chuỗi không dấu), các ô chọn cho trường danh mục với đúng chuỗi trong bảng,
   hiển thị `xac_suat` dạng phần trăm kèm `disclaimer`, `canh_bao`; `chu_y_vung` chỉ ghi "tham khảo".

**Quy tắc:** trả lời bằng tiếng Việt; không đọc `.env`/`.env.local`; không in dữ liệu bệnh nhân; báo lại các file đã thêm/sửa
và kết quả test.

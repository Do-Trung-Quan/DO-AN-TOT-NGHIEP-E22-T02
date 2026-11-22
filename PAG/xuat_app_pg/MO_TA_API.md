# Mô hình PG — mô tả cho backend

Viết cho: người viết backend/web (FastAPI) tích hợp mô hình PG. Sinh tự động bởi `tao_mo_ta.py`; đừng sửa tay.

## 1. Mô hình là gì

- Nhãn `ketqua`: phim X-quang được đọc là **nghi bụi phổi silic** (không phải chẩn đoán xác định).
- PG = Graph Transformer 14 nút: 6 vùng phổi trên ảnh (phải/trái × trên/giữa/dưới), 6 nhóm lâm sàng, 1 nút tỉnh, 1 nút tổng hợp.
  Bản đã khoá ngày 07/11/2026 và đã chấm test một lần. Có ảnh hay không đều chạy được (không ảnh thì 6 nút ảnh tắt).
- Đầu ra = **xác suất đã hiệu chỉnh** (Platt, riêng nhóm có phim / không phim), trung bình 5 mô hình fold × 3 seed.
- **Chưa có ngưỡng đã khoá** → backend chỉ hiển thị xác suất, không tự kết luận dương/âm cho đến khi có ngưỡng được duyệt.
- **Chỉ nhận 5 tỉnh** của dữ liệu học (nút tỉnh là một đầu vào của mô hình). Tỉnh khác → `ValueError`; web nên dùng ô chọn,
  hiển thị tên có dấu, gửi chuỗi không dấu như bảng dưới.
- Đồng Nai và Phú Yên gần như không có ca có phim lúc học → khi có ảnh ở hai tỉnh này, kết quả trả kèm cảnh báo.

## 2. Gọi

```python
from du_doan_pg import MoHinhPG
mh = MoHinhPG("đường/dẫn/mo_hinh_PG_web")          # nạp 1 lần lúc khởi động (~vài giây); kiểm sha256 từng file
kq = mh.du_doan(ban_ghi)                            # chỉ lâm sàng
kq = mh.du_doan(ban_ghi, anh=bytes_anh, duong_dan_biovil="…/biovil_t_image_model_proj_size_128.pt")
```

- `ban_ghi`: dict, khoá = tên cột như Excel gốc (bảng mục 4). Thiếu khoá hay giá trị `None`/`""` = bỏ trống. Số có thể gửi dạng chuỗi,
  chấp nhận dấu phẩy thập phân (`"165,5"`).
- `anh`: đường dẫn, `bytes` hoặc `PIL.Image` của **phim ngực thẳng** (PNG/JPG). Ảnh không lưu lại.
  Đường dẫn BioViL-T cũng có thể đặt bằng biến môi trường `BIOVIL_T_PATH`; thư mục mô hình bằng `PG_MODEL_DIR`.
- Không an toàn khi gọi song song nhiều luồng trên cùng một `MoHinhPG` (dùng khoá hoặc hàng đợi).
- Thời gian (GPU máy học): ~1 s/ảnh xử lý, lần gọi ảnh đầu tiên lâu hơn vì nạp mô hình ảnh. Chỉ lâm sàng: tức thì.

Kết quả `du_doan` (dict):

| Khoá | Ý nghĩa |
|---|---|
| `xac_suat` | xác suất `ketqua` = 1, đã hiệu chỉnh (0–1) |
| `xac_suat_theo_fold` | 5 xác suất của 5 mô hình fold (độ dao động giữa các mô hình) |
| `co_anh` | có dùng ảnh hay không |
| `tinh` | tỉnh đã nhận |
| `chu_y_vung` | khi có ảnh: tỉ lệ chú ý của nút tổng hợp lên 6 vùng `RU RM RL LU LM LL` (tổng = 1); **chỉ để tham khảo**, không phải bản đồ tổn thương; không ảnh → `null` |
| `qa_anh` | kiểm tra ảnh: `dao_am` (đã tự đảo âm bản), `du_phoi` (tìm đủ hai phổi), `phoi_R`/`phoi_L` (tỉ lệ diện tích), `ti_le_LR`, `cham_mep`, `ocr`, `vung_rong` (vùng không tách được) |
| `canh_bao` | danh sách câu tiếng Việt nên hiện cho người dùng |
| `mo_hinh`, `nhan`, `experimental`, `disclaimer` | mô tả; `disclaimer` = "Kết quả thử nghiệm, không thay thế chẩn đoán của bác sĩ." |

## 3. Cài đặt

- Python 3.13 (máy học), các gói trong `requirements.txt` (phiên bản lúc học; torch bản CUDA hay CPU đều được).
- Thư mục mô hình `mo_hinh_PG_web/` (~23 MB): `cau_hinh_chung.json` + `fold0…fold4/` (3 `seed*.pt`, `ma_hoa_anh.npz`,
  `ma_hoa_lam_sang.json`, `platt.json`, `cau_hinh.json`). **Dữ liệu riêng tư — không đưa lên GitHub công khai.**
- Chỉ khi nhận ảnh: file BioViL-T `biovil_t_image_model_proj_size_128.pt`; trọng số tách phổi PSPNet do `torchxrayvision`
  tự tải lần đầu vào `~/.torchxrayvision` (máy chủ không có mạng thì chép sẵn thư mục này); RapidOCR kèm sẵn mô hình trong gói.
- Kiểm tra trên máy học: `python kiem_tra.py` (so với điểm đã lưu, lệch lớn nhất 3,6e-7; ảnh khớp tuyệt đối).

## 4. Trường đầu vào

Mô hình so khớp **chính xác chuỗi** với các giá trị đã học (phân biệt hoa thường, giữ cả dấu ngoặc kép nếu có).
Giá trị chữ khác danh sách → coi như "có ghi nhưng không nhận ra" (không lỗi). Nên dùng ô chọn với đúng các chuỗi dưới đây.
Một nhóm bỏ trống hoàn toàn thì nút nhóm đó tắt.

**Nhân khẩu & thể trạng**

| Trường | Kiểu | Giá trị mô hình nhận ra |
|---|---|---|
| `gioitinh` | chữ | `Nam`, `Nu` |
| `namsinh` | số | trống → trung vị lúc học (≈ 1981) + cờ thiếu |
| `cao` | số | trống → trung vị lúc học (≈ 165) + cờ thiếu |
| `can` | số | trống → trung vị lúc học (≈ 60) + cờ thiếu |
| `bmi` | — | tự tính từ `cao`, `can` (gửi lên sẽ bị ghi đè) |
| `A6` | chữ | `Da ket hon va dang chung song`, `Chua ket hon`, `Li than`, `Li di`, `Goa` |
| `hatd` | số | trống → trung vị lúc học (≈ 120) + cờ thiếu |
| `hatt` | số | trống → trung vị lúc học (≈ 78) + cờ thiếu |
| `mach` | số | trống → trung vị lúc học (≈ 76) + cờ thiếu |
| `theluc` | số | trống → trung vị lúc học (≈ 2) + cờ thiếu |

**Phơi nhiễm/nghề**

| Trường | Kiểu | Giá trị mô hình nhận ra |
|---|---|---|
| `tuoinghe` | số | trống → trung vị lúc học (≈ 10) + cờ thiếu |
| `nampx` | số | trống → trung vị lúc học (≈ 9) + cờ thiếu |
| `cv5nam` | chữ | `Khong`, `Co` |
| `tgian1` | số | trống → trung vị lúc học (≈ 5) + cờ thiếu |
| `tgian2` | số | trống → trung vị lúc học (≈ 4) + cờ thiếu |
| `A7` | chữ | `VLXD`, `Luyen kim`, `"Khai thac, che tac da"`, `Khac`, `Khai thac da Granit`, `Khai thac/che bien quang`, `Khai thac, che tac da`, `SX gom/su/thuy tinh` |
| `A9a` | số | trống → trung vị lúc học (≈ 10) + cờ thiếu |
| `A9b` | số | trống → trung vị lúc học (≈ 9) + cờ thiếu |
| `A10` | chữ | `Hop dong toan thoi gian`, `Hop dong ban thoi gian`, `Vien chuc`, `Khac` |
| `A11` | chữ | `Hop dong dai han`, `Hop dong ngan han`, `Khong ky hop dong`, `Vien chuc`, `Khac` |
| `A12` | chữ | `Khong`, `Co` |
| `F4mu` | chữ | `Rat thuong xuyen`, `Thuong xuyen`, `Khong su dung`, `Thinh thoang`, `Hiem khi` |
| `F4ung` | chữ | `Rat thuong xuyen`, `Thuong xuyen`, `Khong su dung`, `Thinh thoang`, `Hiem khi` |
| `F4KhauTrang` | chữ | `Rat thuong xuyen`, `Thuong xuyen`, `Thinh thoang`, `Khong su dung`, `Hiem khi` |
| `F4gang` | chữ | `Rat thuong xuyen`, `Thuong xuyen`, `Khong su dung`, `Thinh thoang`, `Hiem khi` |
| `F4BHLD` | chữ | `Rat thuong xuyen`, `Thuong xuyen`, `Khong su dung`, `Thinh thoang` |
| `F4Kinh` | chữ | `Khong su dung`, `Rat thuong xuyen`, `Thuong xuyen`, `Thinh thoang`, `Hiem khi` |
| `cviec_lv` | — | tự tính từ chữ nghề tự do ở cột `cviec` (bị ghi đè) |
| `pxuong_lv` | — | tự tính từ chữ nghề tự do ở cột `pxuong` (bị ghi đè) |
| `cviec1_lv` | — | tự tính từ chữ nghề tự do ở cột `cviec1` (bị ghi đè) |
| `cviec2_lv` | — | tự tính từ chữ nghề tự do ở cột `cviec2` (bị ghi đè) |

**Hút thuốc**

| Trường | Kiểu | Giá trị mô hình nhận ra |
|---|---|---|
| `hutthuoc` | chữ | `Khong hut thuoc`, `Co hut thuoc` |
| `slthuoc` | số | trống → trung vị lúc học (≈ 9) + cờ thiếu |

**Triệu chứng & tiền sử**

| Trường | Kiểu | Giá trị mô hình nhận ra |
|---|---|---|
| `tiensuhh` | chữ | `Khong`, `Co` |
| `benhhh` | chữ | `VIEM HONG`, `viem hong`, `VIEM MUI DI UNG`, `VIEM HONG MAN` |
| `tsnoi` | chữ | `KHONG`, `THA`, `viem da day`, `VIEM DD`, `viem DD`, `DA DAY`, `viem hong man tinh` |
| `tsngoai` | chữ | `KHONG`, `MO RT`, `mo de`, `mo ruot thua` |
| `ho` | chữ | `Khong`, `Co` |
| `tdho` | chữ | `Ban ngay`, `Ca ngay va dem`, `Ban dem` |
| `tsho` | chữ | `Tung con`, `Khac`, `Lien tuc` |
| `n1` | chữ | `thinh thoang`, `THINH THOANG`, `thi thoang` |
| `khacdom` | chữ | `Khong`, `Co` |
| `loaidom` | chữ | `Dom nhay`, `Dom mu` |
| `tdkhacdo` | chữ | `Ban ngay`, `Ca ngay va dem`, `Ban dem` |
| `khotho` | chữ | `Khong`, `Co` |
| `mdkhotho` | chữ | `Kho tho khi gang suc`, `Kho tho tung con`, `"Kho tho thuong xuyen, ca khi nghi ngoi"`, `Kho tho thuong xuyen, ca khi nghi ngoi`, `Kho tho khi lam viec nhe` |
| `tdkhotho` | chữ | `Ban ngay`, `Ca ngay va dem`, `Ban dem` |
| `daunguc` | chữ | `Khong`, `Co` |
| `vitridau` | chữ | `Dau sau xuong uc`, `Nguc trai`, `Hai ben nguc`, `Nguc phai` |
| `daulan` | chữ | `Khong`, `Co` |
| `tcdau` | chữ | `De ep`, `Khac`, `Xiet chat`, `Nhu dao dam` |
| `n2` | chữ | `nhoi` |
| `tgdau` | số | trống → trung vị lúc học (≈ 2) + cờ thiếu |
| `ytodau` | chữ | `Khi gang suc`, `Khac`, `Khi nghi ngoi` |
| `n3` | chữ | `TU NHIEN` |
| `ytotang` | chữ | `nghi ngoi do dau`, `khong`, `met moi dau tang`, `NGHI NGOI` |
| `chaymui` | chữ | `Khong`, `Co` |
| `khan` | chữ | `Khong`, `Co` |
| `khokhe` | chữ | `Khong`, `Co` |
| `dhkhac` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `metmoi` | chữ | `Khong`, `Co` |
| `sutcan` | chữ | `Khong`, `Co` |
| `socansut` | số | trống → trung vị lúc học (≈ 3) + cờ thiếu |
| `tgsut` | số | trống → trung vị lúc học (≈ 60) + cờ thiếu |
| `bnn` | số | trống → trung vị lúc học (≈ 0) + cờ thiếu |

**Hô hấp ký**

| Trường | Kiểu | Giá trị mô hình nhận ra |
|---|---|---|
| `fvctt` | số | trống → trung vị lúc học (≈ 3.3) + cờ thiếu |
| `fev1tt` | số | trống → trung vị lúc học (≈ 3) + cờ thiếu |
| `fev1_fvc` | — | tự tính từ `fev1tt`/`fvctt` (bị ghi đè) |

**Khám thực thể**

| Trường | Kiểu | Giá trị mô hình nhận ra |
|---|---|---|
| `longnguc` | chữ | `Can doi` |
| `khoangls` | chữ | `Binh thuong`, `Gian rong` |
| `rungthan` | chữ | `Binh thuong` |
| `rungt1` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungt2` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungt3` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungt4` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungt5` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungt6` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungg1` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungg2` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungg3` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungg4` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungg5` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `rungg6` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `go` | chữ | `Binh thuong` |
| `goduc1` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `goduc2` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `goduc3` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `goduc4` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `goduc5` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `goduc6` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `riraopn` | chữ | `Binh thuong`, `Giam` |
| `ran` | chữ | `Khong`, `Co` |
| `ranam` | chữ | `Co` |
| `ranno` | chữ | `Khong` |
| `ranrit` | chữ | `Khong` |
| `ranngay` | chữ | `Khong` |
| `vtam1` | chữ | `Khong` |
| `vtam2` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtam3` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtam4` | chữ | `Khong` |
| `vtam5` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtam6` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtno1` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtno2` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtno3` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtno4` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtno5` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtno6` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtrit1` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtrit2` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtrit3` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtrit4` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtrit5` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtrit6` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtngay1` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtngay2` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtngay3` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtngay4` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtngay5` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |
| `vtngay6` | chữ | chỉ học được "có ghi / bỏ trống" (mọi giá trị khác rỗng tính như nhau) |

**Bối cảnh** (bắt buộc)

| Trường | Kiểu | Giá trị |
|---|---|---|
| `tinh` | chữ | `Binh Dinh` (Bình Định), `Dong Nai` (Đồng Nai), `Hai Duong` (Hải Dương), `Phu Yen` (Phú Yên), `Thai Nguyen` (Thái Nguyên) |

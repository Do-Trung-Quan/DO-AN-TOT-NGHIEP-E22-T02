"""Sinh MO_TA_API.md tu thu muc mo hinh (bang truong dau vao lay thang tu ma_hoa_lam_sang.json cua 5 fold).
    python tao_mo_ta.py [thu_muc_mo_hinh]
"""
import json
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else DIR.parent / "cache_v3" / "mo_hinh_PG_web"
C = json.loads((D / "cau_hinh_chung.json").read_text(encoding="utf-8"))
S = [json.loads((D / f"fold{k}" / "ma_hoa_lam_sang.json").read_text(encoding="utf-8")) for k in range(5)]
TINH = {"Binh Dinh": "Bình Định", "Dong Nai": "Đồng Nai", "Hai Duong": "Hải Dương", "Phu Yen": "Phú Yên", "Thai Nguyen": "Thái Nguyên"}
TU_TINH = {"bmi": "tự tính từ `cao`, `can` (gửi lên sẽ bị ghi đè)", "fev1_fvc": "tự tính từ `fev1tt`/`fvctt` (bị ghi đè)"}
for c in ("cviec", "pxuong", "cviec1", "cviec2"):
    TU_TINH[f"{c}_lv"] = f"tự tính từ chữ nghề tự do ở cột `{c}` (bị ghi đè)"


def dong(c):
    k = S[0][c][0]
    if c in TU_TINH:
        return f"| `{c}` | — | {TU_TINH[c]} |"
    if k == "num":
        med = sorted(s[c][1] for s in S)[2]
        return f"| `{c}` | số | trống → trung vị lúc học (≈ {med:g}) + cờ thiếu |"
    lv = []
    for s in S:
        lv += [x for x in s[c][1] if x not in lv and x != "__NA__"]
    if not lv:
        return f"| `{c}` | chữ | chỉ học được \"có ghi / bỏ trống\" (mọi giá trị khác rỗng tính như nhau) |"
    return f"| `{c}` | chữ | " + ", ".join(f"`{x}`" for x in lv) + " |"


L = []
for g, cs in C["nhom_bien"].items():
    L += [f"\n**{g}**\n", "| Trường | Kiểu | Giá trị mô hình nhận ra |", "|---|---|---|"] + [dong(c) for c in cs]
L += ["\n**Bối cảnh** (bắt buộc)\n", "| Trường | Kiểu | Giá trị |", "|---|---|---|",
      "| `tinh` | chữ | " + ", ".join(f"`{t}` ({TINH.get(t, t)})" for t in C["tinh_hop_le"]) + " |"]
BANG = "\n".join(L)

DOC = f"""# Mô hình PG — mô tả cho backend

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
{BANG}
"""
(DIR / "MO_TA_API.md").write_text(DOC, encoding="utf-8")
print("da ghi MO_TA_API.md")

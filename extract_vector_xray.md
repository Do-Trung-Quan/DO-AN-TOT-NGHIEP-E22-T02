# Trích xuất vector đặc trưng ảnh X-quang — tài liệu bàn giao cho phase fusion

> **Đối tượng đọc:** người xây dựng pipeline fusion ở repo khác.
> **Tóm tắt một dòng:** 1835 vector 256 chiều, out-of-fold, khoá `img_id`, không PII, huấn luyện trên nhãn `ketqua`.
>
> ⚠️ **Nếu bạn chỉ đọc một mục, đọc [§6 Ba điều bắt buộc](#6-ba-điều-bắt-buộc-với-bên-fusion).** Bỏ qua §6.1 sẽ làm hỏng đồ thị kNN mà không có triệu chứng gì.

---

## 1. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Script pipeline | ✅ Đã viết đầy đủ, regex join đã kiểm chứng trên 1835/1835 tên file thật |
| Artifact (`.parquet`) | ❌ **Chưa sinh ra** — máy soạn tài liệu không có Python, không có GPU, không có ảnh gốc |
| Người thực thi | hnim2910, trên máy có GPU + ảnh gốc |

Toàn bộ số liệu trong tài liệu này được đo **trực tiếp trên bộ vector cũ đã commit** (`FinetuneNewData/vectors/`), không phải suy đoán. Các con số về artifact mới sẽ do `qc_report.py` sinh ra sau khi chạy.

---

## 2. Vì sao phải làm lại — 3 lỗi của bộ vector cũ

Bộ `FinetuneNewData/vectors/` (commit 26/8) **không dùng được cho fusion**. Đã kiểm định bằng số:

### 2.1 · Một nửa vector là hằng số

```
256/512 chiều có ptp == 0 (đo trong float64) trên toàn bộ 1835 ảnh
Khối chết: đúng dải liên tục 256–511
  dim 256 = 0.00907195   dim 257 = -0.031278   dim 258 = -0.00173306
```

**Nguyên nhân:** BioViL-T dùng `RESNET50_MULTI_IMAGE` — là model *temporal*, thiết kế để nhận ảnh hiện tại **+ ảnh tiền sử**. Nửa sau của `img_embedding` là đặc trưng sai khác so với phim cũ. Khi chỉ cấp 1 ảnh, nhánh đó trả về một embedding "missing" đã học — hằng số với mọi đầu vào.

**Bẫy kiểm định:** kiểm bằng `float32` chỉ lộ ra **1** chiều chết. Phải dùng `float64` hoặc `ptp()` mới thấy đủ 256.

### 2.2 · Rò rỉ bệnh nhân trong split

Script `finetune_biovilt_silicosis.py:213` dùng `train_test_split` ngẫu nhiên thuần. 7/7 script SetA trước đó đều dùng `GroupShuffleSplit` + `assert` theo `patient_base` — lớp bảo vệ này bị gỡ khi chuyển sang bộ dữ liệu mới.

| Chỉ số | Đo được |
|---|---|
| Bệnh nhân ở **cả** train và test | 62 người |
| Ảnh test thuộc người đã có ảnh trong train | 66/360 = **18,3%** |
| Người có nhãn **đổi** giữa các lần chụp | 25 |

Docstring của script khẳng định *"mỗi bệnh nhân 1 ảnh, không có bệnh nhân nào xuất hiện ở cả train lẫn test"*. Điều này **sai sự thật** — 187 bệnh nhân có nhiều lần khám (tối đa 3). `assert` ở dòng 225 kiểm `file_name` chứ không kiểm bệnh nhân, nên luôn pass và tạo cảm giác an toàn giả.

### 2.3 · 80,4% vector sinh từ ảnh model đã học

Split cũ: train 1253 + val 222 = 1475/1835 ảnh model đã thấy. Chỉ 360 ảnh là sạch. Không có cross-fitting.

---

## 3. Khoá join — phát hiện then chốt

Đây là thứ cho phép chuyển hẳn sang dữ liệu timeseries mới.

**`timeseriesDATA/info.csv` có đúng 1835 dòng, khớp 1:1 với 1835 ảnh. Khoá join là cột `id`.**

Match theo **tên** chỉ đạt **67,4%** vì tên file có lỗi chính tả:

```
016_LE_THI_UYEN_20181219.jpg   <->  hoten = "Le Thi Yen"
155_TRUOGN_VAN_MIEN_...        <->  "Truong Van Mien"
2007_HAONG_VAN_ANH_...         <->  "Hoang Van Anh"
```

Match theo **`id`** đạt **1835/1835**. Có 5 pattern tên file:

| # | Pattern | Ví dụ | `id` ở đâu | Số ảnh |
|---|---|---|---|---|
| 1 | `NNNN_TEN_YYYYMMDD` | `008_MAC_DUY_THANG_20181219.jpg` | tiền tố | 1209 |
| 2 | `YYYYMMDD_HHMMSS_ID_PID_SID` | `20181222_075752_1024_12798_14253.jpg` | trường 3 | 619 |
| 3 | `YYYYMMDD_HHMMSS_ID_TEN` | `20181223_075707_1235_DO THANH TRI.jpg` | trường 3 | ↑ |
| 4 | `YYYYMMDD_HHMMSS_ID_TEN NS` | `20191113_081107_8001_THAI VAN THI 1971.jpg` | trường 3 | ↑ |
| 5 | `TEN_ID[_YYYYMMDD]` | `NGUYEN QUANG DAO_3931.jpg` | sau tên | 7 |

**Kết quả kiểm chứng regex trên toàn bộ 1835 tên file thật:**

```
khớp pattern     = 1835/1835   (P1=1209, P2-4=619, P5=7)
id duy nhất      = 1835        id trùng lặp = 0
id không có trong info.csv = 0
SAI LỆCH NHÃN (ketqua==1  vs  thư mục silicosis) = 0
```

Sai lệch nhãn bằng **0 trên 1835 mẫu** là bằng chứng khoá join chính xác — không phải trùng hợp.

---

## 4. Nhãn — đọc kỹ mục này trước khi báo cáo bất kỳ con số nào

| Cột | Nguồn | Dương | Tỉ lệ | Vai trò |
|---|---|---|---|---|
| `label_ketqua` | `info.csv.ketqua == 1` | 462 | 25,2% | ✅ **NHÃN HUẤN LUYỆN** |
| `label_bnn` | `info.csv.bnn == 'Co'` | 99 | 5,4% | 🎯 **NHÃN ĐÍCH** — chỉ mang theo, không dùng train |

**`ketqua` là kết quả đọc phim, KHÔNG phải chẩn đoán bệnh nghề nghiệp.** Đây là nhãn proxy.

**Vì sao train trên `ketqua` chứ không phải `bnn`:** `bnn` chỉ có 99 ca dương / **86 bệnh nhân dương**. Chia 5 fold còn ~17 bệnh nhân dương mỗi fold — không đủ để fine-tune ResNet50, gần như chắc chắn không hội tụ. `ketqua` có 462 dương, học được, và tổn thương phổi trên phim là tín hiệu liên quan trực tiếp.

> ⛔ **Cấm đặt AUC của đặc trưng ảnh cạnh AUC nhánh timeseries trong cùng một bảng** nếu hai bên khác nhãn. Hai bài toán khác nhau. Nếu bắt buộc phải so, chỉ so trên `label_bnn` qua probe out-of-fold.

Bật head phụ trên `bnn` bằng `--aux-weight 0.3` nếu muốn thử; mặc định tắt (`0.0`).

---

## 5. Artifact bàn giao

### 5.1 · `imagefeat/output/image_features.parquet` — file chính

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `img_id` | int32 | **Khoá chính** = `info.csv.id`. Join trực tiếp với timeseries. |
| `has_image` | int8 | 1 = có ảnh. Toàn bộ 1835 hàng đều = 1. |
| `fold_id` | int8 | Fold cross-fitting đã sinh vector này (0–4). |
| `img_feat_0` … `img_feat_255` | float32 | 256 chiều sống. |

### 5.2 · Các file kèm

| File | Nội dung |
|---|---|
| `image_index.parquet` | `img_id`, `file_hash`, `patient_group`, `label_ketqua`, `label_bnn`, `batch_date` |
| `image_features_control.parquet` | Bộ **đối chứng** (P2) — cùng schema, `fold_id = -1` |
| `image_features_manifest.json` | Checkpoint + SHA256, nhãn train, transform, seed, chiến lược split, phiên bản thư viện |
| `image_features_qc.md` | Báo cáo kiểm định — **không có file này thì không nhận bàn giao** |

### 5.3 · Không có gì trong artifact

- ❌ Tên file gốc, họ tên bệnh nhân, số điện thoại
- ❌ Split 1475/360 của nhánh ảnh — **cố ý loại bỏ**, xem §6.2

Bảng ánh xạ ngược `img_id → tên file` nằm ở `LOCAL_ONLY_filename_map.csv`, đã chặn trong `.gitignore`, **không bao giờ commit**.

### 5.4 · Nạp dữ liệu

```python
import pandas as pd

features = pd.read_parquet("imagefeat/output/image_features.parquet")
index    = pd.read_parquet("imagefeat/output/image_index.parquet")
clinical = pd.read_csv("timeseriesDATA/info.csv", low_memory=False)

data = (clinical
        .merge(index,    left_on="id", right_on="img_id", how="left")
        .merge(features, on="img_id",  how="left"))
assert data["img_id"].notna().all(), "join thất bại"
```

---

## 6. Ba điều BẮT BUỘC với bên fusion

### 6.1 · ⚠️ Phải mean-center trước khi dựng đồ thị kNN

**Đây là điều quan trọng nhất trong tài liệu này.** Có một hiểu lầm phổ biến rằng bỏ 256 chiều chết sẽ chữa được đồ thị kNN. **Sai.** Số đo thực tế trên 140 ảnh mẫu:

| Cách tính | Cosine trung bình |
|---|---|
| 512 chiều, raw | **0,759** |
| 256 chiều sống, raw | **0,755** ← gần như **không đổi** |
| 256 chiều sống, đã trừ mean | **0,003** |

Khối chết chỉ chiếm **1,5%** tổng `norm²` (‖chết‖² = 0,105 vs ‖sống‖² = 7,07) — về mặt cơ học không thể kéo cosine lên 0,76. Nguyên nhân thật là **anisotropy / cone effect**: embedding có một thành phần trung bình chung rất lớn, hiện tượng bình thường của mọi embedding sâu.

```python
# ĐÚNG
from sklearn.preprocessing import StandardScaler
image_features = StandardScaler().fit_transform(image_features)   # fit trên TRAIN
graph = build_knn_graph(image_features)

# SAI — mọi ảnh "giống nhau ~76%", đồ thị gần như vô nghĩa
graph = build_knn_graph(raw_features)
```

*Ghi chú kỹ thuật:* `sklearn.preprocessing.StandardScaler` **không** sinh NaN với cột phương sai 0 — nó gọi `_handle_zeros_in_scale`, đặt `scale_ = 1`. NaN chỉ xảy ra nếu tự chuẩn hoá bằng `(x - mean) / std` thuần numpy.

### 6.2 · Split fusion phải là split của nhánh timeseries

Artifact **cố ý không mang theo** split 1475/360 của nhánh ảnh. Nếu ai đó dùng nhầm nó làm split fusion, toàn bộ cơ chế chống rò rỉ của nhánh timeseries sụp đổ.

Nếu cần một split mới: dùng `StratifiedGroupKFold` với `groups = index["patient_group"]`. **1647 bệnh nhân / 1835 ảnh** — 187 người có nhiều lần khám, nên chia theo hàng là rò rỉ.

### 6.3 · PR-AUC là metric chính, không phải ROC-AUC

`bnn` chỉ 5,4% dương. ROC-AUC sẽ trông đẹp một cách gây hiểu lầm. Nền PR-AUC = **0,054**.

---

## 7. Đặc điểm dữ liệu phải khai báo trong báo cáo

### 7.1 · Nhiễu loạn theo lô chụp

Tỉ lệ dương lệch **8 lần** giữa các đợt khám:

| Ngày chụp | Số ảnh | Tỉ lệ dương |
|---|---|---|
| 20191113 / 20191115 | 73 / 164 | 5,5% |
| 20181219 | 199 | 14,1% |
| 20190605 | 129 | 41,9% |
| 20181223 | 165 | 44,8% |
| 20181224 | 75 | **58,7%** |

Embedding mã hoá rất mạnh thông tin máy chụp / đợt khám. **Nên báo cáo metric tách theo lô**, và cân nhắc phân tầng theo `batch_date` khi chia fold.

### 7.2 · Neural collapse — vector "512 chiều" thực chất bao nhiêu?

Đo trên 256 chiều sống của bộ cũ:

```
PC1 chiếm 86,6% phương sai · PC1+PC2 = 91,0% · effective rank ≈ 2,1/256
```

Hiện tượng bình thường sau fine-tune bằng CE nhị phân. Hệ quả cho fusion:

- **Không nén xuống 1–8 chiều.** Các hướng phương sai thấp (~9% phương sai) lại mang thông tin cho `bnn`.
- **Không giữ nguyên 512.** Một nửa là hằng số.
- **Điểm ngọt: 256 chiều sống, hoặc PCA 32–64.**

`qc_report.py` sẽ in lại hai chỉ số này cho bộ vector mới — con số có thể khác vì đổi nhãn huấn luyện.

---

## 8. Pipeline mới — thay đổi so với script cũ

| Hạng mục | Script cũ | Bản mới |
|---|---|---|
| Split | `train_test_split` ngẫu nhiên | `StratifiedGroupKFold(5)` theo `patient_group` |
| `assert` | trên `file_name` (vô dụng) | trên `patient_group`, cả train/val, train/heldout, val/heldout |
| Nhãn | tên thư mục `normal`/`silicosis` | `label_ketqua` từ `info.csv` |
| Chọn checkpoint | `val loss` | **`val PR-AUC`** — đồng chuẩn nhánh timeseries |
| Độ chính xác trích xuất | `autocast` fp16 | **fp32, tắt autocast** |
| Số chiều xuất | 512 (một nửa là rác) | 256 sống + `assert` 0 chiều hằng |
| Khoá | `fname` (chứa PII) | `img_id` |

**Cross-fitting:**

```
StratifiedGroupKFold(5)   groups = patient_group,  y = label_ketqua
for k in 0..4:
    train trên 4 fold  →  trích vector cho fold k   (fold model CHƯA thấy)
ghép lại → 1835 vector, mỗi vector out-of-fold
```

Khi bảo vệ, cả hai nhánh nói được cùng một câu: **"mọi đặc trưng đều là out-of-fold"**.

---

## 9. Chạy lại

```bash
python imagefeat/build_index.py --images-root "E:/.../NEW DATA"   # ~1 phút, in "1835/1835"
python imagefeat/crossfit_finetune.py --smoke                     # kiểm tra nhanh trước
python imagefeat/crossfit_finetune.py                             # ~1 buổi GPU
python imagefeat/extract_image_features.py                        # ~3 phút, bộ đối chứng
python imagefeat/qc_report.py                                     # cổng nghiệm thu
```

| Script | Vai trò |
|---|---|
| `imagefeat/build_index.py` | Join theo `id`, sinh nhãn + nhóm bệnh nhân, gỡ PII |
| `imagefeat/crossfit_finetune.py` | Cross-fitting 5 fold, xuất vector out-of-fold |
| `imagefeat/extract_image_features.py` | Bộ đối chứng từ checkpoint SetA có sẵn |
| `imagefeat/qc_report.py` | 5 assert cứng + 4 chỉ số báo cáo |

---

## 10. Điều kiện nghiệm thu

`qc_report.py` thoát mã 0. **Không có `image_features_qc.md` thì không nhận bàn giao.**

**5 assert cứng — trượt là từ chối:**

| # | Chỉ tiêu | Ngưỡng |
|---|---|---|
| 1 | Số hàng khớp chỉ mục | = 1835/1835 |
| 2 | Số chiều hằng số (**kiểm float64**) | = 0 |
| 3 | Bệnh nhân nằm ở >1 fold | = 0 |
| 4 | NaN/Inf trong hàng có ảnh | = 0 |
| 5 | Tên file / họ tên trong artifact | = 0 |

**4 chỉ số báo cáo (phải có mặt, không đặt ngưỡng):**

6. Cosine TB trước/sau mean-centering — bắt lỗi ở §6.1
7. Effective rank + % phương sai PC1
8. PR-AUC probe out-of-fold trên `bnn` (nền 0,054)
9. Độ chính xác đoán lô chụp từ embedding

> **Mốc so sánh — đọc kỹ:** bộ vector cũ đạt PR-AUC probe 0,235 trên `bnn`, nhưng con số đó tính trên vector **còn nhiễm rò rỉ** (thổi phồng ~0,07 AUC). Sau khi cross-fit sạch, **PR-AUC tụt là bình thường và đúng đắn** — không phải dấu hiệu làm hỏng. Đừng lấy 0,235 làm chuẩn để đuổi theo.

---

## 11. Cảnh báo PII — cần xử lý

`timeseriesDATA/info.csv` chứa **họ tên (`hoten`), năm sinh (`namsinh`) và số điện thoại (`sdt`)** của 1835 người, đã commit lên GitHub. `FinetuneNewData/vectors/*meta*.csv` chứa họ tên đầy đủ trong cột `fname`.

Đã thêm cả hai vào `.gitignore`. **Nhưng `.gitignore` chỉ chặn commit mới — lịch sử git vẫn còn dữ liệu.** Cần `git filter-repo` để dọn triệt để; phải thống nhất cả nhóm trước khi force-push.

---

## 12. Việc chưa làm

| Việc | Ghi chú |
|---|---|
| Chạy pipeline sinh artifact | Cần máy có GPU + ảnh gốc |
| Kiểm chứng "SetA ∩ NEW DATA = ∅" | Giả định của bộ đối chứng (P2), **chưa kiểm bằng số**. Nếu có danh sách ảnh SetA thì đối chiếu `patient_group`. |
| Dọn lịch sử git | Cần thống nhất cả nhóm |
| Đặc trưng ít collapse hơn | Chỉ làm nếu PR-AUC fusion vẫn thấp sau cross-fitting. Hướng: backbone đóng băng chưa fine-tune, hoặc lấy ở tầng trước lớp chiếu. |

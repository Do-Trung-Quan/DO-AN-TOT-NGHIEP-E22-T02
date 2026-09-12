# Overview — Nhánh ảnh (`Chest-X-ray`)

> Tài liệu này dành cho người **mới tham gia** nhánh ảnh. Đọc xong bạn sẽ hiểu: đồ án làm gì, nhánh ảnh đóng vai trò gì, mỗi folder/file là gì, và roadmap từ bước đầu tiên tới khi có model sẵn sàng **fusion** với nhánh Timeseries.

---

## 1. Bức tranh tổng thể của đồ án

**Đề tài:** *Nghiên cứu mô hình học đa phương thức dựa trên đồ thị (Graph-based Multimodal Learning) cho bài toán chẩn đoán bệnh phổi nghề nghiệp, kết hợp ảnh X-quang ngực + dữ liệu chức năng lâm sàng.*

**Mục tiêu cuối:** đưa vào 1 bộ (ảnh X-quang mới + dữ liệu timeseries mới) → mô hình chẩn đoán **có/không mắc bệnh nghề nghiệp** (accuracy mục tiêu > 0.8).

Đồ án chia làm **3 nhánh** chạy song song rồi hợp nhất:

```
┌─────────────────────┐     ┌──────────────────────────┐
│  Nhánh ẢNH (đây)     │     │  Nhánh TIMESERIES        │
│  X-quang → vector    │     │  phiếu khám → vector     │
│  đặc trưng ảnh       │     │  đặc trưng lâm sàng      │
└──────────┬──────────┘     └────────────┬─────────────┘
           │                             │
           └──────────────┬──────────────┘
                          ▼
              ┌────────────────────────┐
              │  Nhánh FUSION (GNN)    │
              │  node ảnh + node TS,   │
              │  edge quan hệ → predict│
              └────────────────────────┘
```

- **Nhánh ảnh** (branch `Chest-X-ray` — bạn đang ở đây): huấn luyện một **backbone CNN trích xuất đặc trưng ảnh**. Input = ảnh X-quang phổi, output = 1 vector đặc trưng.
- **Nhánh Timeseries** (branch `Timeseries`): xử lý + mã hóa dữ liệu phiếu khám (~8000 dòng), cho qua LSTM (nhóm cột đo trên 6 vùng phổi) + Dense (cột tĩnh) → 1 vector đặc trưng lâm sàng.
- **Fusion**: đồ thị dân số 8.030 node (node = bệnh nhân, đặc trưng = vector timeseries ⊕ vector ảnh nếu có), so sánh 3 backbone GNN. Xem [fusion_roadmap.md](fusion_roadmap.md).

**Nhãn mục tiêu cuối cùng:** cột `bnn` (0 = khỏe mạnh, 1 = có bệnh nghề nghiệp).

---

## 2. Nhánh ảnh làm gì — tóm tắt 1 câu

> Huấn luyện một mô hình **nhìn ảnh X-quang phổi và nhận ra dấu hiệu bệnh bụi phổi nghề nghiệp** (đốm mờ, nốt xơ, tổn thương), để cuối cùng biến mỗi ảnh thành 1 **vector đặc trưng** đưa vào bước fusion.

Pipeline gồm 2 giai đoạn kinh điển trong deep learning y tế:

```
   PRETRAIN backbone (học đặc trưng phổi chung, kho ảnh lớn)
              │
              ▼
   FINE-TUNE nhị phân (dạy model phân biệt Có/Không bệnh, data có nhãn)
              │
              ▼
   Backbone trích đặc trưng  →  vector ảnh  →  FUSION
```

**Bài toán fine-tune là nhị phân:** `Co` (mắc bệnh) vs `Khong` (không), lấy từ cột `bnn`.

---

## 3. Ý nghĩa từng folder & file

### 3.1. Tài liệu (gốc repo)

| File | Vai trò |
|---|---|
| `[ĐỐ ÁN TỐT NGHIỆP] - Document.docx` | Đề bài gốc + roadmap khái quát của thầy (nguồn chân lý về *yêu cầu*). |
| `README.md` | Tóm tắt tiếng Anh về pipeline nhánh ảnh + bảng kết quả 2 backbone đầu (BYOL, CheXNet). |
| `HUONG_DAN_ABLATION_LOSS.md` | Hướng dẫn thí nghiệm **đổi hàm loss** (giữ nguyên backbone BioViL-T). Có bảng kết quả các loss. |
| `overview.md` | **File này.** |
| `.gitattributes` | Cấu hình **Git LFS** cho các file `.pth`/`.pt` (model nặng >40 MB). |
| `.gitignore` | Bỏ qua data thật (ảnh bệnh nhân, excel), `__pycache__`, `images_256/`… vì lý do riêng tư + dung lượng. |

### 3.2. Giai đoạn PRETRAIN (backbone chưa gắn nhãn bệnh)

| Folder | Nội dung |
|---|---|
| `Pre-train/` | **BYOL tự huấn luyện** (self-supervised) trên NIH ChestX-ray14 (~112k ảnh không nhãn) → `xray_byol_backbone.pth` (ResNet18). Gồm script train, log, loss curve, và `FINETUNE_GUIDE.md` (tài liệu bàn giao rất chi tiết cách load backbone + fine-tune). |
| `Pre-train CheXNET/` | Weight **CheXNet (DenseNet121)** — **tải sẵn** từ repo `arnoweng/CheXNet` (KHÔNG tự train, chỉ remap key state-dict rồi fine-tune). |
| `Pre-train BioViL-T/` | Weight **BioViL-T** (Microsoft, `hi-ml-multimodal`) — backbone vision-language pretrained trên X-quang. `biovil_t_image_model_proj_size_128.pt`. **Đây là backbone được chọn cuối cùng.** |

**File quan trọng trong `Pre-train/`:**
- `pretrain_byol.py` — script pretrain BYOL đã dùng (60 epoch, ~10.4h RTX 3060).
- `resize_images.py` — resize ảnh NIH về 256×256 trước khi train.
- `check_collapse.py` — kiểm tra feature có bị "collapse" không (sức khỏe của self-supervised).
- `xray_byol_backbone.pth` — **sản phẩm backbone BYOL** (42.7 MB, ResNet18).
- `pretrain_log.txt`, `pretrain_loss.csv/.png`, `pretrain_config.json` — log + biểu đồ + hyperparams (dùng cho báo cáo).
- `FINETUNE_GUIDE.md` — **đọc kỹ nếu bạn fine-tune**: cách load backbone, preprocessing bắt buộc, chiến lược chống imbalance, k-fold, metrics.

### 3.3. Giai đoạn FINE-TUNE (so sánh backbone)

Mỗi folder = 1 lần fine-tune 1 backbone khác nhau, **cùng data + cùng split (seed 42)** để so công bằng:

| Folder | Backbone | Script | Kết quả (ROC-AUC) |
|---|---|---|---|
| `Finetune 2/` | BYOL-ResNet18 | `cnn-model.py` | 0.832 |
| `Finetune 3/` | CheXNet-DenseNet121 | `cnn-model-chexnet.py` | 0.847 |
| `Finetune 4/` | **BioViL-T** (fine-tune đầy đủ) | `finetune_biovilt_setA.py` | **0.902** ✅ tốt nhất |

Mỗi folder chứa: script `.py`, model tốt nhất `best_model_*.pth`, `evaluation_*.png` (ROC + confusion matrix), `train_log*.txt`, `confusion_matrix_*.csv`.
- `Finetune 2/SetA_Labels.xlsx` — **file nhãn dùng chung** cho mọi lần fine-tune (2 cột: `file_name`, `bnn`; ~2129 ảnh).

### 3.4. Giai đoạn ABLATION LOSS (giữ backbone BioViL-T, đổi hàm loss)

Sau khi chốt backbone = BioViL-T, thầy yêu cầu thử **đổi hàm loss** xem có cải thiện không (ablation study — chỉ đổi đúng 1 yếu tố). Xem `HUONG_DAN_ABLATION_LOSS.md`.

| Folder | Loss | Script | ROC-AUC | Sens | Spec |
|---|---|---|---|---|---|
| `Finetune CrossEntropy/` | CrossEntropy (mặc định) | `finetune_biovilt_ce.py` | **0.913** | 0.847 | 0.861 |
| `Finetune WeightedCE/` | Weighted CE | `finetune_biovilt_wce.py` | 0.891 | 0.837 | 0.839 |
| `Finetune 4/` | Focal γ=2 | `finetune_biovilt_setA.py` | 0.902 | 0.906 | 0.781 |
| `Finetune Focal5/` | Focal γ=5 | `finetune_biovilt_focal5.py` | 0.893 | 0.871 | 0.818 |
| *(chưa làm)* | Focal γ=1, Label smoothing | — | — | — | — |

**Kết luận ablation (đã ghi trong hướng dẫn):** đổi loss **gần như không tăng AUC** (0.891–0.913, chênh ~0.02 nằm trong nhiễu của 339 ảnh test). Loss chủ yếu **dịch cân bằng Sensitivity ↔ Specificity**, không phải để "tăng điểm". CrossEntropy mặc định thậm chí cho AUC cao nhất.

---

## 4. Chi tiết kỹ thuật (để bắt tay vào code ngay)

### 4.1. Data fine-tune
- **Nhãn:** `Finetune 2/SetA_Labels.xlsx` → 2 cột `file_name`, `bnn` (`Co`→1, `Khong`→0).
- **Ảnh:** thư mục `images_256/train` (KHÔNG có trong repo — data riêng tư/nặng, gitignore).
- **Split:** `GroupShuffleSplit(seed=42)` 2 tầng → Train 70% / Val 15% / **Test 15% (=339 ảnh, cố định)**. Group theo `patient_base` (bỏ hậu tố `_số.jpg`) để **chống rò rỉ bệnh nhân** giữa các tập.

### 4.2. Backbone chốt = BioViL-T (chi tiết trong `Finetune 4/finetune_biovilt_setA.py`)
- Load local từ `Pre-train BioViL-T/…proj_size_128.pt` (không có → tự tải HuggingFace).
- Feature để gắn head: `out.img_embedding` (**512-d**, giữ lại 256 chiều đầu — xem mục 7.4) → head `Dropout(0.3) + Linear(→2)`.
- **Tiền xử lý RIÊNG của BioViL-T:** `Resize(512) → Crop(448) → ToTensor(0–1) → ExpandChannels`. **Không** dùng ImageNet normalize; ảnh xám nhân bản thành 3 kênh. *(Khác BYOL/CheXNet — mỗi backbone có preprocessing riêng, đừng trộn lẫn.)*
- **Train 2-stage:** Stage 1 đóng băng backbone chỉ train head; Stage 2 (từ epoch `unfreeze`) mở toàn bộ với LR nhỏ cho backbone (1e-5) + LR lớn cho head (1e-4).
- **Chống imbalance:** `WeightedRandomSampler` + class weights trong loss.
- **AMP** (mixed precision) để vừa VRAM 6 GB.
- **Chọn threshold bằng Youden's J** (cân bằng Sens/Spec) khi đánh giá, thay vì mặc định 0.5.

### 4.3. Cách chạy (ví dụ)
```bash
# Pretrain BYOL (nếu train lại backbone tự-giám-sát)
python "Pre-train/resize_images.py"
python "Pre-train/pretrain_byol.py" --epochs 60 --batch-size 64

# Fine-tune từng backbone
python "Finetune 2/cnn-model.py"              # BYOL-ResNet18
python "Finetune 3/cnn-model-chexnet.py"      # CheXNet-DenseNet121
python "Finetune 4/finetune_biovilt_setA.py"  # BioViL-T (thêm --smoke để test nhanh 2 epoch)

# Ablation loss (giữ backbone BioViL-T)
python "Finetune CrossEntropy/finetune_biovilt_ce.py"
python "Finetune Focal5/finetune_biovilt_focal5.py"
```

> ⚠️ Các đường dẫn trong script đang **hard-code theo máy Windows** (`e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\...`). Khi chạy máy khác phải sửa `EXCEL_PATH`, `IMAGE_DIR`, `OUTPUT_DIR`, `BACKBONE_PATH`.

---

## 5. Giai đoạn TRÍCH ĐẶC TRƯNG cho fusion — `imagefeat/` ✅ hoàn thành 09/2026

Đây là cây cầu nối nhánh ảnh sang nhánh fusion. Sản phẩm bàn giao là **3 bộ vector đặc trưng** cho cùng 1.835 ảnh, cùng schema, để nhánh fusion so sánh xem nguồn nào tốt nhất.

### 5.1. Các file trong `imagefeat/`

| File | Vai trò |
|---|---|
| `build_index.py` | **Nguồn sự thật duy nhất.** Ghép 1.835 ảnh với `info.csv` theo cột `id` (khớp 1835/1835), sinh `image_index.parquet`. Gỡ PII: tên file → hash. |
| `extract_image_features.py` | Trích đặc trưng bằng backbone đóng băng. `--frozen` = không nạp checkpoint; mặc định = nạp checkpoint SetA. |
| `crossfit_finetune.py` | Cross-fit 5 fold theo `patient_uid`: mỗi ảnh nhận vector từ model **chưa từng thấy nó**. Lưu từng fold ra `_folds/` nên chạy lại là tiếp tục. |
| `qc_report.py` | **Cổng nghiệm thu.** 5 assert cứng + 4 chỉ số báo cáo. Trượt là từ chối bàn giao. |
| `output/image_index.parquet` | `(1835, 7)`: `img_id, file_hash, patient_uid, patient_group, label_ketqua, label_bnn, batch_date` |
| `output/image_features*.parquet` | 3 bộ đặc trưng, mỗi bộ `(1835, 259)` |
| `colab/phase_0_3_frozen.ipynb`<br>`colab/phase_0_4_crossfit.ipynb` | Notebook chạy trên Colab T4, bấm Run all |

### 5.2. Khóa nối sang nhánh timeseries

```
image_index.img_id  ==  fusion_node_meta.id   (khớp 1835/1835, đã kiểm chứng)
```

**Không dùng tên file** làm khóa: tên file chứa họ tên bệnh nhân. Nhánh timeseries cũng đã bỏ `file_name` khỏi mọi artifact vì lý do này.

`patient_uid` lấy từ `output/patient_identity.parquet` do nhánh Timeseries sinh ra, để **hai nhánh dùng chung một định nghĩa "ai là ai"**. Công thức cũ (`họ tên + năm sinh`) bỏ sót những ca gõ nhầm tên — ví dụ thật:

```
id   39  Vu Thi Tuoi   1972  Nu  Hai Duong  0346398595   ← có ảnh
id 3317  vu thi tuou   1972  Nu  Hai Duong  0346398595   ← có ảnh
```

Cùng số điện thoại, một người, lệch một chữ cái. Cách cũ tách thành 2 nhóm → hai phim của cùng một người rơi vào 2 fold khác nhau. **13/87 người có nhiều phim bị lỗi này**, nay đã khắc phục.

### 5.3. Ba bộ đặc trưng — khác nhau ở đâu

| Bộ | Backbone fine-tune trên | Nhãn huấn luyện | Rò rỉ |
|---|---|---|---|
| **`control`** | SetA — 2.129 ảnh **khác hoàn toàn** (overlap = 0, đã kiểm chứng bằng số) | **`bnn`** | Không — backbone chưa từng thấy 1.835 ảnh này |
| **`frozen`** | **Không fine-tune** | — | Không, theo cấu tạo |
| **`crossfit`** | Chính 1.835 ảnh này, 5 fold theo `patient_uid` | **`ketqua`** | Không — mỗi vector là out-of-fold |

### 5.4. Kết quả nghiệm thu

Cả 3 bộ đều qua 5 cổng assert cứng:

```
1835/1835 hàng · 0 chiều hằng (kiểm float64) · 0 NaN/Inf
0 bệnh nhân nằm ở >1 fold · 0 tên người trong artifact
```

Riêng `crossfit`: 5 fold cân bằng (363–370 ảnh, ~350 người/fold), **0 bệnh nhân bắc cầu**.

### 5.5. So sánh chất lượng — probe tuyến tính, CV 5-fold nhóm theo `patient_uid`

| Bộ | PC1 | Effective rank | `bnn` ROC | **`bnn` PR-AUC** | `ketqua` ROC | `ketqua` PR-AUC |
|---|---|---|---|---|---|---|
| `control` | 90,7% | 1,80 | 0,8646 | **0,2522** | 0,7335 | 0,5176 |
| `frozen` | 34,3% | 14,79 | 0,8630 | 0,2183 | 0,7390 | 0,5186 |
| `crossfit` | 48,6% | 5,56 | 0,8146 | 0,1865 | **0,8165** | **0,6153** |

*(nền PR-AUC: `bnn` = 0,0540 · `ketqua` = 0,2518)*

Khoảng tin cậy bootstrap 95% cho `bnn` PR-AUC:

```
control   0,2608  [0,1950 – 0,3404]
frozen    0,2232  [0,1715 – 0,2778]
crossfit  0,1934  [0,1441 – 0,2512]

control − crossfit = +0,0674  [+0,0033 – +0,1400]   P(control > crossfit) = 98,1%
```

### 5.6. ⚠️ Phát hiện quan trọng: nhãn huấn luyện quyết định, không phải miền dữ liệu

Kết quả **ngược với dự đoán ban đầu**. Giả thuyết khi thiết kế Phase 0.4 là *"cross-fit thích nghi đúng miền dữ liệu nên sẽ mạnh nhất"*. Thực tế:

* `crossfit` **tốt nhất** cho `ketqua` (PR-AUC 0,6153 so với ~0,518) — đúng như thiết kế, nó được huấn luyện trên nhãn đó.
* `crossfit` **kém nhất** cho `bnn` (0,1865), và khoảng cách với `control` là **có ý nghĩa thống kê** (P = 98,1%).

Lời giải nằm ở **nhãn huấn luyện**, không phải ở miền dữ liệu:

| Bộ | Nhãn huấn luyện | Miền ảnh | Kết quả trên `bnn` |
|---|---|---|---|
| `control` | **`bnn`** ✅ đúng nhãn đích | SetA — khác miền | **tốt nhất** |
| `frozen` | không có | — | trung bình |
| `crossfit` | `ketqua` ❌ nhãn proxy | đúng miền | **kém nhất** |

Càng tối ưu mạnh cho nhãn proxy `ketqua` (tổn thương trên phim), biểu diễn càng vứt bỏ những hướng thông tin mà nhãn đích `bnn` (bệnh nghề nghiệp) cần. Bệnh nghề nghiệp không chỉ là tổn thương trên phim — nó còn phụ thuộc tiền sử phơi nhiễm, thời gian tiếp xúc, nghề nghiệp.

**Hệ quả cho phase fusion:** dùng **`control` làm nguồn chính**, hai bộ còn lại làm nhánh ablation. Đây cũng là lựa chọn đã ghi trong `fusion_roadmap.md` từ đầu — nay có bằng chứng số ủng hộ.

### 5.7. Hai điều nhánh fusion BẮT BUỘC phải biết

**1. Phải mean-center trước khi dựng đồ thị kNN.** Cosine trung bình giữa các ảnh ở dạng thô là **0,63 – 0,98** tùy bộ; sau khi mean-center chỉ còn **~0,002 – 0,008**. Nếu dựng kNN trên vector thô, mọi ảnh sẽ "giống nhau 60–98%" và đồ thị gần như vô nghĩa.

**2. Nhiễu loạn theo lô chụp.** Embedding đoán được **lô chụp** với độ chính xác 41–47% (nền theo lớp lớn nhất 10,9%, 16 lô). Tỷ lệ dương theo ngày chụp lệch 8 lần (5,5% → 44,8%). Khi báo cáo kết quả fusion nên tách metric theo `batch_date`.

---

## 6. Roadmap nhánh ảnh — trạng thái cuối

| Bước | Việc | Trạng thái |
|---|---|---|
| **0. Chuẩn bị data** | Thu thập ảnh, resize, chuẩn hóa tên khớp nhãn. | ✅ Xong |
| **1. Pretrain backbone** | BYOL trên NIH ChestX-ray14; thêm CheXNet và BioViL-T tải sẵn. | ✅ Xong |
| **2. Fine-tune & so backbone** | BYOL (0.832) < CheXNet (0.847) < ImageNet (0.891) < **BioViL-T (0.902)**. | ✅ Xong — chốt BioViL-T |
| **3. Ablation loss** | CE / Weighted CE / Focal γ=2,5. Kết luận: loss không tăng AUC, chỉ dịch Sens↔Spec. | ✅ Xong |
| **4. Chốt mô hình trích đặc trưng** | `Finetune CrossEntropy/best_model_ce.pth` (AUC 0.913). | ✅ Xong |
| **5. Xây chỉ mục chuẩn** | `build_index.py` → `image_index.parquet`, khóa `img_id`, gỡ PII, kèm `patient_uid`. | ✅ Xong (09/2026) |
| **6. Trích 3 bộ đặc trưng** | `control` (P2) · `frozen` (P0.3) · `crossfit` (P0.4). Cả 3 qua cổng QC. | ✅ Xong (09/2026) |
| **7. Bàn giao fusion** | 3 bộ `.parquet` + `image_index.parquet` + 3 QC report + 3 manifest. | ✅ **Sẵn sàng** |

**Nhánh ảnh đã hoàn thành nhiệm vụ.** Mọi artifact cần cho fusion đã có, tái lập được bằng mã đã commit, và đã qua nghiệm thu.

---

## 7. ⚠️ Ghi chú: những chỗ làm KHÁC với file docx

> Các điểm nhánh ảnh **đã lệch/mở rộng** so với roadmap khái quát trong `.docx`. Phần lớn là cải tiến hợp lý, nhưng cần biết để báo cáo cho khớp.

1. **Backbone cuối cùng KHÔNG phải backbone tự-pretrain.**
   - *docx:* pretrain SimCLR/BYOL → fine-tune, tức backbone do mình tự pretrain.
   - *Thực tế:* BYOL tự-pretrain chỉ đạt AUC 0.832 (thấp nhất). Chốt **BioViL-T** (weight pretrained tải sẵn từ Microsoft, 0.902). SimCLR chưa hề được dùng.

2. **Data fine-tune KHÔNG phải "500 ảnh của mình".**
   - *docx:* fine-tune với 500 ảnh X-quang của mình.
   - *Thực tế:* fine-tune dùng **set_A ≈ 2.129 ảnh**. Bộ 1.835 ảnh của dự án được dùng ở giai đoạn trích đặc trưng (mục 5), không dùng để chọn backbone.

3. **Phân phối test set ngược với thực tế triển khai.**
   - Test set set_A có 202 `Co` / 137 `Khong` (bệnh chiếm đa số), trong khi bộ 1.835 ảnh chỉ có 5,4% `bnn` dương. → AUC 0.913 trên set_A **không giữ nguyên** khi áp lên phân phối thật.

4. **Vector đặc trưng là 512-D → giữ 256, không phải 2048-D.**
   - BioViL-T (`RESNET50_MULTI_IMAGE`) trả về `img_embedding` 512 chiều. Nhưng đây là mô hình **temporal** nhận *ảnh hiện tại + ảnh tiền sử*; khi chỉ cấp 1 ảnh thì **256 chiều sau (256..511) là hằng số** ở cả 1.835 ảnh. Đã đo bằng `float64` và loại bỏ.
   - ⚠️ Phép kiểm này **phải dùng `float64`**: ở `float32` chỉ phát hiện 1/256 chiều chết.

5. **Nhãn huấn luyện của `crossfit` là `ketqua`, KHÔNG phải `bnn`.**
   - `bnn` chỉ có 99 ca dương / 86 bệnh nhân dương. Chia 5 fold còn ~17 người dương mỗi fold — không đủ fine-tune ResNet50. `ketqua` có 462 dương (25,2%), học được.
   - **Hệ quả:** không được đặt AUC nhánh ảnh cạnh AUC nhánh timeseries trong cùng một bảng — hai bài toán khác nhau.

6. **Chỉ mới 1 lần split cho phần chọn backbone, chưa k-fold.**
   - Kết quả so 4 backbone (0.832 → 0.913) đều từ **1 split duy nhất** trên 339 ảnh test. Chênh ~0.02 AUC có thể chỉ là nhiễu. Riêng phần trích đặc trưng (mục 5) đã dùng 5-fold nhóm theo bệnh nhân.

7. **Đường dẫn hard-code theo máy cá nhân** vẫn còn trong các script `Finetune */` cũ. Các script trong `imagefeat/` đã chuyển sang tham số dòng lệnh (`--images-root`, `--identity`).

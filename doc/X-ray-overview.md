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
- **Fusion** (dự kiến): tạo đồ thị nối node ảnh và node timeseries, đưa vào GNN (GCN/GAT/DGCNN) để dự đoán cuối cùng. Xây ~8000 đồ thị (500 cái có cả ảnh + TS, còn lại chỉ TS).

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
- Feature để gắn head: `out.img_embedding` (~2048-d) → head `Dropout(0.3) + Linear(→2)`.
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

## 5. Roadmap nhánh ảnh — từ đầu tới sẵn sàng fusion

| Bước | Việc | Trạng thái |
|---|---|---|
| **0. Chuẩn bị data** | Thu thập ảnh X-quang, resize, làm sạch cơ bản; chuẩn hóa tên file khớp nhãn. | ✅ Xong |
| **1. Pretrain backbone** | BYOL self-supervised trên NIH ChestX-ray14 (~112k ảnh) → `xray_byol_backbone.pth`. Ngoài ra chuẩn bị thêm backbone CheXNet (tải sẵn) và BioViL-T (tải sẵn). | ✅ Xong |
| **2. Fine-tune & so sánh backbone** | Fine-tune nhị phân `Co/Khong` trên set_A, cùng split seed 42. So 4 backbone: BYOL (0.832) < CheXNet (0.847) < ImageNet (0.891) < **BioViL-T (0.902)**. | ✅ Xong — **chốt BioViL-T** |
| **3. Ablation loss** | Giữ BioViL-T, thử CE / Weighted CE / Focal γ=2,5 / (còn Focal γ=1, Label smoothing). Kết luận: loss không tăng AUC, chỉ dịch Sens↔Spec. | 🟡 Gần xong (thiếu 2 loss) |
| **4. Chốt mô hình trích đặc trưng** | Chọn checkpoint tốt nhất (VD `Finetune CrossEntropy/best_model_ce.pth`, AUC 0.913) làm **feature extractor** đóng băng. | 🟡 Cần chốt chính thức |
| **5. Trích vector cho fusion** | Dùng backbone đã chốt, bỏ head phân loại, chạy **500 ảnh của dự án** (ảnh khớp với timeseries) → xuất mỗi ảnh 1 vector `img_embedding` (~2048-d), lưu ra file (npy/parquet) kèm `file_name`/`id` để nhánh fusion nối. | ⬜ **CHƯA có script — việc tiếp theo** |
| **6. Bàn giao fusion** | Đưa vector ảnh + vector timeseries vào GNN, tạo node/edge, train predict `bnn`. | ⬜ Thuộc nhánh Fusion |

**Việc cần làm tiếp theo rõ ràng nhất:** viết script **Bước 5** (feature extraction) — hiện chưa tồn tại. Nó là cây cầu nối nhánh ảnh sang fusion: load `best_model_*.pth`, `model.eval()`, forward 500 ảnh dự án qua `backbone(...).img_embedding`, lưu ma trận `[500, 2048]` + danh sách id.

### Kết quả hiện có (cùng test set 339 ảnh, cùng split seed 42)

| Model | ROC-AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| BYOL-ResNet18 | 0.832 | 0.76 | 0.88 | 0.58 |
| CheXNet-DenseNet121 | 0.847 | 0.79 | 0.78 | 0.81 |
| BioViL-T + Focal γ=2 | 0.902 | 0.86 | 0.906 | 0.781 |
| **BioViL-T + CrossEntropy** | **0.913** | 0.85 | 0.847 | 0.861 |

→ Đã **vượt mốc mục tiêu 0.8** ở mức backbone ảnh. Bước fusion sẽ kết hợp thêm dữ liệu lâm sàng để đẩy cao hơn/ổn định hơn.

---

## 6. ⚠️ Ghi chú: những chỗ đồng nghiệp làm KHÁC với file docx

> Đây là các điểm nhánh ảnh **đã lệch/mở rộng** so với roadmap khái quát trong `.docx`. Không hẳn là sai — phần lớn là cải tiến hợp lý — nhưng cần biết để báo cáo cho khớp.

1. **Backbone cuối cùng KHÔNG phải là backbone tự-pretrain.**
   - *docx:* "Pretrain với kho ảnh X-quang trên mạng dùng **SimCLR/BYOL** → fine-tune". Tức là backbone chính là do mình tự pretrain.
   - *Thực tế:* BYOL tự-pretrain chỉ đạt AUC 0.832 (thấp nhất). Đồng nghiệp đã **thử thêm CheXNet và BioViL-T (đều là weight pretrained tải sẵn từ bên ngoài)** và **chọn BioViL-T** (0.902) làm chính thức. → Backbone chốt là **pretrained ngoài**, không phải self-supervised như docx hình dung. (SimCLR chưa hề được dùng; chỉ có BYOL.)

2. **Data fine-tune KHÔNG phải "500 ảnh của mình".**
   - *docx:* "Fine tune với **500 ảnh X-quang của mình**".
   - *Thực tế:* các lần fine-tune dùng **set_A ≈ 2129 ảnh** (`SetA_Labels.xlsx`), một tập silicosis lớn hơn, **không phải** 500 ảnh khớp với timeseries. 500 ảnh dự án (folder Drive) hiện **chưa được đưa vào pipeline** — chúng dành cho **Bước 5 (trích vector fusion)**, chưa làm.

3. **Phân phối test set ngược với thực tế triển khai.**
   - Test set set_A có **202 `Co` / 137 `Khong`** (bệnh chiếm đa số).
   - Nhưng `FINETUNE_GUIDE.md` (bàn giao BYOL) mô tả data thật khớp timeseries **imbalance ~14:1 nghiêng về `Khong`** (29 `Co` / 404 `Khong`). → Metrics đẹp trên set_A **chưa chắc giữ nguyên** khi áp lên phân phối thật của dự án. Cần lưu ý khi tuyên bố "AUC 0.913".

4. **Đường đi nhãn đã đổi giữa chừng.**
   - `FINETUNE_GUIDE.md` ban đầu hướng dẫn fine-tune bằng cột `bnn` của file chính `Main_data_fixed_Not_Encode_Mapping_New.xlsx` (433 ảnh khớp). Con đường này **đã bị bỏ**, thay bằng `SetA_Labels.xlsx`. Hai nguồn nhãn này **không đồng nhất về quy mô lẫn phân phối** — đừng nhầm khi đọc tài liệu cũ.

5. **Chỉ mới 1 lần split, chưa k-fold.**
   - `FINETUNE_GUIDE.md` và chính ghi chú ablation khuyến nghị **Stratified K-Fold / nhiều seed** để số liệu ổn định. Hiện mọi kết quả đều từ **1 split duy nhất**; chênh lệch ~0.02 AUC giữa các model/loss có thể chỉ là nhiễu. Nên chạy k-fold trước khi kết luận chắc trong báo cáo.

6. **Tên mô hình fusion chưa thống nhất.**
   - *docx:* nói chung là **GNN (GCN/GAT)**, xây ~8000 đồ thị.
   - *README:* ghi cụ thể **DGCNN**. Chỉ là khác biệt tên gọi/biến thể — cần chốt thống nhất khi sang nhánh fusion.

7. **Đường dẫn hard-code theo máy Windows** của người train (`e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\...`). Ai chạy lại phải sửa path; đây là nợ kỹ thuật nhỏ nên cân nhắc chuyển sang config/biến môi trường.

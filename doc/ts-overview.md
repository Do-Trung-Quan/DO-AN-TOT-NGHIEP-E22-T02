# Overview — Nhánh Timeseries (`Timeseries`)

> Tài liệu này dành cho người **mới tham gia** nhánh Timeseries (dữ liệu lâm sàng). Đọc xong bạn sẽ hiểu: đồ án làm gì, nhánh timeseries đóng vai trò gì, mỗi folder/file là gì, và roadmap từ bước đầu tiên tới khi có model sẵn sàng **fusion** với nhánh ảnh. Xem thêm [X-ray-overview.md](X-ray-overview.md) cho nhánh ảnh.

---

## 1. Bức tranh tổng thể của đồ án

**Đề tài:** *Nghiên cứu mô hình học đa phương thức dựa trên đồ thị (Graph-based Multimodal Learning) cho bài toán chẩn đoán bệnh phổi nghề nghiệp, kết hợp ảnh X-quang ngực + dữ liệu chức năng lâm sàng.*

**Mục tiêu cuối:** đưa vào 1 bộ (ảnh X-quang mới + dữ liệu timeseries mới) → mô hình chẩn đoán **có/không mắc bệnh nghề nghiệp** (accuracy mục tiêu > 0.8).

Đồ án chia làm **3 nhánh** chạy song song rồi hợp nhất:

```
┌─────────────────────┐     ┌──────────────────────────┐
│  Nhánh ẢNH           │     │  Nhánh TIMESERIES (đây)  │
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

- **Nhánh ảnh** (branch `Chest-X-ray`): huấn luyện backbone CNN trích đặc trưng ảnh X-quang → 1 vector đặc trưng ảnh. Mô hình tốt nhất BioViL-T + CrossEntropy đạt AUC ~0.913. Xem [X-ray-overview.md](X-ray-overview.md).
- **Nhánh Timeseries** (branch `Timeseries` — bạn đang ở đây): xử lý + mã hóa dữ liệu phiếu khám (~8030 bệnh nhân, trong đó 1835 bệnh nhân đã match ảnh X-quang), cho qua LSTM (nhóm cột đo trên 6 vùng phổi) + Embedding (nghề nghiệp) + Dense (cột tĩnh) → 1 vector đặc trưng lâm sàng.
- **Fusion** (dự kiến): tạo đồ thị nối node ảnh và node timeseries, đưa vào GNN (GCN/GAT/DGCNN) để dự đoán cuối cùng. Xây đồ thị kết hợp 1,835 node có cả ảnh + TS (với 99 ca mắc bệnh `bnn=1`). Bài báo tham khảo: [`doc/Paper_128-Graph_Convolutional_Network_for_Occupational_Disease_Prediction.pdf`](Paper_128-Graph_Convolutional_Network_for_Occupational_Disease_Prediction.pdf).

**Nhãn mục tiêu cuối cùng (theo đề bài):** cột `bnn` (0 = khỏe mạnh, 1 = có bệnh nghề nghiệp).

---

## 2. Nhánh Timeseries làm gì — tóm tắt 1 câu

> Biến mỗi phiếu khám sức khỏe (hàng trăm cột: nhân khẩu, nghề nghiệp, triệu chứng, khám lâm sàng 6 vùng phổi, đo chức năng hô hấp) thành **dữ liệu số sạch**, rồi qua mạng đa nhánh (Embedding + LSTM + Dense) để ra **1 vector đặc trưng lâm sàng** phục vụ chẩn đoán và bước fusion.

Toàn bộ công việc nằm trong **1 notebook duy nhất**: [`ĐỒ_ÁN.ipynb`](../ĐỒ_ÁN.ipynb). Pipeline:

```
   Excel hợp nhất (8030 dòng, 1835 bệnh nhân match ảnh X-quang)
              │  lọc cột rác + cột rò rỉ nhãn
              ▼
   Fill missing + sửa outlier + tách has_image (1835 bệnh nhân có ảnh)
              │
              ▼
   Encoding: binary / ordinal / one-hot / scale số + phân loại nghề 10 cấp
              │
              ▼
   Tách 3 nhánh dữ liệu:  Job(4)  |  LSTM 3D(6 vùng × k)  |  Numerical
              │
              ▼
   Keras multi-input (Embedding + LSTM + Dense) -> StratifiedGroupKFold 5-Fold
              │
              ▼
   Ensemble Feature Extraction (Lấy trung bình lớp fusion_dense 32-D)
              │
              ▼
    output/timeseries_features.parquet + fusion_node_meta.parquet -> Sẵn sàng cho GNN Fusion
```

---

## 3. Ý nghĩa từng folder & file

| Đường dẫn | Vai trò |
|---|---|
| **`ĐỒ_ÁN.ipynb`** | **File chính** — toàn bộ pipeline timeseries (load → clean → encode → model → train). |
| `data/Main_data_Merged_1835_Images.xlsx` | **Dữ liệu thô gộp mới nhất** (8030 bệnh nhân), cập nhật thông tin lâm sàng & match chuẩn xác **1,835 file ảnh X-quang**. |
| `data/Main_data_fixed_Not_Encode_Mapping_New.xlsx` | Dữ liệu thô gốc (8030 dòng, 433 ảnh). Đã lưu trữ bảo toàn. |
| `data/new data.csv` | Dữ liệu bổ sung từ thầy (1835 dòng có ảnh X-quang). Đã lưu trữ bảo toàn. |
| `output/timeseries_features.parquet` | **Sản phẩm chính** — ma trận đặc trưng lâm sàng 32D `(8030, 34)`: `id`, `has_image`, `ts_feat_0..31`. |
| `output/fusion_node_meta.parquet` | **Bắt buộc cho fusion** — `(8030, 13)` chứa nhãn `bnn`, `split`, `fold_id`, `patient_uid` và cột phụ trợ dựng cạnh. |
| `output/patient_identity.parquet` | `(8030, 2)` — `id` + `patient_uid`, **dành cho nhánh ảnh dùng chung** một định nghĩa danh tính. Không chứa PII. |
| `scripts/patient_identity.py` | Luật xác định "cùng một người" + phân loại nghề 10 cấp. Notebook import từ đây. |
| `scripts/build_merged_dataset.py` | Tái lập `data/Main_data_Merged_1835_Images.xlsx` từ file gốc + `data/info.csv`. |
| `output/timeseries_features.npy` | Ma trận float32 thô `[8030, 32]`. |
| `doc/X-ray-overview.md` | Tài liệu tổng quan nhánh ảnh (file tham khảo song song). |
| `doc/Paper_128-...Occupational_Disease_Prediction.pdf` | **Bài báo tham khảo** cho bước fusion bằng Graph Convolutional Network. |
| `doc/ts-overview.md` | **File này.** |

---

## 4. Chi tiết pipeline trong `ĐỒ_ÁN.ipynb` (theo từng cell)

| Cell | Nội dung | Điểm cần nhớ |
|---|---|---|
| **Mount Drive** | `drive.mount` (Colab) | Notebook thiết kế để chạy trên **Google Colab** (hoặc chạy offline). |
| **1. Load + lọc cột** | Load dữ liệu từ `data/Main_data_Merged_1835_Images.xlsx`. Drop cột định danh (`tinh, hoten, sdt, sobh`), nhãn chi tiết (`bnncuthe`), và **cột đọc phim X-quang** (`chatluongphim, ketqua, matdotonthuong, kichthuoctt, tonthuongkhac`) để **tránh rò rỉ sang nhánh ảnh**. | Giữ cột `id` làm chìa khóa nối sang nhánh ảnh (khớp `img_id` 1835/1835). |
| **2. Fill missing** | Text thiếu → `"không"`, số thiếu → `0`; sửa outlier `F4gang=6.0` về mode; **tách `has_image` ra mảng riêng**. | 1,835 bệnh nhân có ảnh, 6,195 bệnh nhân không có. `file_name` bị loại hẳn vì chứa họ tên (PII). |
| **3. Feature Engineering + Encoding** | Mã hóa Binary/Ordinal/One-hot. Tính tuổi động bằng năm hiện tại trừ năm sinh (`datetime.now().year - namsinh`). | `bnn` được loại hoàn toàn khỏi các cột đặc trưng. |
| **4. Phân loại nghề 10 cấp** | Rule-based keyword → phân 4 cột nghề (`cviec, pxuong, cviec1, cviec2`) thành **ID nguyên 0–9** theo mức độ độc hại. | Dùng làm input cho lớp Embedding. |
| **5. Tách 3 nhánh dữ liệu** | - **LSTM 3D:** tensor `[N, 6 vùng, k đặc trưng]`. <br>- **Numerical:** các cột số còn lại. <br>- **Job:** 4 cột nghề (ID). <br>- **Chia dữ liệu:** `StratifiedGroupKFold(7)` theo `patient_uid` → Dev 6.882 (85,7%) / Test 1.148 (14,3%). | Phân tầng theo `bnn`, gom nhóm theo bệnh nhân. |
| **6. Model Keras đa nhánh** | Hàm `build_model()`. Nhánh Job sử dụng `Flatten()` giữ nguyên đặc trưng của từng ô nghề. | Lớp áp chót `fusion_dense` thiết lập 32 chiều nén thông tin lâm sàng. |
| **7. Train + Đánh giá** | - **StratifiedGroupKFold 5-Fold** theo `patient_uid` trên tập Dev.<br>- Imputer/Scaler fit độc lập trong fold chống rò rỉ.<br>- Quét ngưỡng F2-Score trên OOF (ngưỡng tối ưu `0.75`).<br>- Ensemble trên Test Hold-out (1.148 mẫu): **ROC-AUC = 0.9556**, **PR-AUC = 0.5133**, **Recall = 58.06%** (18/31 ca bệnh), **Precision = 42.86%**. | Biểu đồ lưu tại `output/learning_curves.png` và `output/confusion_matrix.png`. |
| **8. Trích xuất đặc trưng** | Trích xuất đặc trưng Ensemble 32 chiều bằng cách trung bình hóa lớp `fusion_dense` của 5 mô hình fold cho 8030 bệnh nhân. | Xuất `output/timeseries_features.parquet` (khóa `id` + `has_image`), `.npy`, và `output/fusion_node_meta.parquet`. |

---

## 5. Roadmap nhánh Timeseries — Trạng thái hiện tại

| Bước | Việc | Trạng thái |
|---|---|---|
| **0. Chuẩn bị data thô** | Gộp dữ liệu mới (1,835 ảnh, 99 ca bệnh) với 8,030 bệnh nhân gốc → `data/Main_data_Merged_1835_Images.xlsx`. | ✅ Xong |
| **1. Lọc cột rác + chống rò rỉ** | Bỏ cột định danh, nhãn chi tiết `bnncuthe`, cột đọc phim X-quang. | ✅ Xong |
| **2. Fill missing + làm sạch** | Điền thiếu theo rule, sửa outlier, tách `has_image` (1,835 bệnh nhân có ảnh). | ✅ Xong |
| **3. Mã hóa toàn bộ** | Mã hóa nhị phân / phân loại / scale số + phân loại nghề 10 cấp. Tính tuổi động theo năm hiện tại. | ✅ Xong |
| **4. Định hình data 3D + đa nhánh** | Tensor `[N, 6 vùng, k]` cho LSTM; tách job/numerical; split theo nhóm bệnh nhân, Test 14,3%. | ✅ Xong |
| **5. Model + K-Fold** | Huấn luyện StratifiedGroupKFold 5-Fold (nhóm theo `patient_uid`) với Class Weights & ReduceLROnPlateau. | ✅ Xong |
| **6. Khắc phục rò rỉ & Sai nhãn** | Target chuẩn là `bnn`, loại hoàn toàn `bnn` khỏi feature đầu vào, fit scaler/imputer trong fold. | ✅ Xong |
| **7. Đánh giá Ensemble & Quét F2** | Đánh giá ensemble trên tập Test; Tối ưu hóa ngưỡng tự động qua OOF F2-Score (ngưỡng 0.77, Recall 46.88%, PR-AUC 0.4198). | ✅ Xong |
| **8. Xuất vector đặc trưng cho fusion** | Trích xuất đặc trưng Ensemble 32 chiều trung bình 5 fold, lưu parquet và npy. | ✅ Xong |
| **9. Sửa rò rỉ bệnh nhân + đổi khóa nối** | Split ngẫu nhiên để **241 bệnh nhân** nằm ở cả dev lẫn test (20,0% node test, 15,6% ca bệnh). Chuyển sang `StratifiedGroupKFold` theo `patient_uid` (luật số điện thoại / 6 trường); đổi khóa nối `file_name` → `id` + `has_image` để gỡ PII. Xuất thêm `fusion_node_meta.parquet` và `patient_identity.parquet`. | ✅ Xong (09/2026) |

---

## 6. Lịch sử sửa đổi & Giải quyết các lỗi lớn

Mô hình đã được tối ưu hóa toàn diện:

1. **Gộp bổ sung 1,835 ảnh X-quang:** Nâng số lượng bệnh nhân có ảnh từ **433 ➔ 1,835** (với **99 ca mắc bệnh**), tháo gỡ hoàn toàn "nút thắt cổ chai" dữ liệu cho giai đoạn GNN Fusion.
2. **Đã sửa nhãn mục tiêu:** Target trỏ chính xác về cột `bnn` (Bệnh nghề nghiệp).
3. **Loại bỏ rò rỉ dữ liệu (Data Leakage):** Loại `bnn` khỏi feature đầu vào; chuẩn hóa số liệu fit độc lập inside-fold.
4. **Cải tiến chiến lược chống mất cân bằng:** Áp dụng Class Weights trực tiếp trong `.fit()`, kết hợp ReduceLROnPlateau tự động giảm LR khi bão hòa.
5. **Tối ưu hóa ngưỡng chẩn đoán bằng F2-Score:** Chọn ngưỡng tối ưu **`0.77`**, nâng Recall tập Test lên **`46.88%`** (bắt đúng 15/32 ca bệnh trong tập test hold-out).
6. **Đồng nhất kết quả (Reproducibility):** Cài đặt cố định seed toàn cục (`random`, `numpy`, `tensorflow`).

7. **Sửa rò rỉ bệnh nhân giữa train/test (09/2026):** 8.030 lượt khám chỉ thuộc **7.024 bệnh nhân** (1.002 người khám nhiều lần, chiếm 25% số dòng). Cách chia ngẫu nhiên để **241 bệnh nhân** nằm ở cả dev lẫn test — ảnh hưởng 20,0% node test và **15,6% số ca bệnh**. Đã chuyển sang `StratifiedGroupKFold` theo `patient_uid` (ưu tiên số điện thoại, xem [`scripts/patient_identity.py`](../scripts/patient_identity.py)), kèm assert cứng. OOF PR-AUC trung bình 5 fold `0.4680 → 0.4465` — con số **trung thực hơn**, không phải mô hình kém đi.
8. **Gỡ PII khỏi sản phẩm bàn giao:** bỏ `file_name` (chứa họ tên bệnh nhân) khỏi mọi artifact, thay bằng khóa `id` — đã kiểm chứng khớp `img_id` của nhánh ảnh 1835/1835.

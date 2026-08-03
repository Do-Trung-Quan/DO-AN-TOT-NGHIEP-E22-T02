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

- **Nhánh ảnh** (branch `Chest-X-ray`): huấn luyện backbone CNN trích đặc trưng ảnh X-quang → 1 vector đặc trưng ảnh. Đã đạt AUC ~0.91 (BioViL-T). Xem [X-ray-overview.md](X-ray-overview.md).
- **Nhánh Timeseries** (branch `Timeseries` — bạn đang ở đây): xử lý + mã hóa dữ liệu phiếu khám (~8030 bệnh nhân), cho qua LSTM (nhóm cột đo trên 6 vùng phổi) + Embedding (nghề nghiệp) + Dense (cột tĩnh) → 1 vector đặc trưng lâm sàng.
- **Fusion** (dự kiến): tạo đồ thị nối node ảnh và node timeseries, đưa vào GNN (GCN/GAT/DGCNN) để dự đoán cuối cùng. Xây ~8000 đồ thị (500 cái có cả ảnh + TS, còn lại chỉ TS). Bài báo tham khảo: [`doc/Paper_128-Graph_Convolutional_Network_for_Occupational_Disease_Prediction.pdf`](Paper_128-Graph_Convolutional_Network_for_Occupational_Disease_Prediction.pdf).

**Nhãn mục tiêu cuối cùng (theo đề bài):** cột `bnn` (0 = khỏe mạnh, 1 = có bệnh nghề nghiệp).

---

## 2. Nhánh Timeseries làm gì — tóm tắt 1 câu

> Biến mỗi phiếu khám sức khỏe (hàng trăm cột: nhân khẩu, nghề nghiệp, triệu chứng, khám lâm sàng 6 vùng phổi, đo chức năng hô hấp) thành **dữ liệu số sạch**, rồi qua mạng đa nhánh (Embedding + LSTM + Dense) để ra **1 vector đặc trưng lâm sàng** phục vụ chẩn đoán và bước fusion.

Toàn bộ công việc nằm trong **1 notebook duy nhất**: [`ĐỒ_ÁN.ipynb`](../ĐỒ_ÁN.ipynb). Pipeline:

```
   Raw phiếu khám (8030 dòng, ~200+ cột)
              │  lọc cột rác + cột rò rỉ nhãn
              ▼
   Fill missing + sửa outlier + tách file_name (cho nhánh ảnh)
              │
              ▼
   Encoding: binary / ordinal / one-hot / scale số + phân loại nghề 10 cấp
              │
              ▼
   Tách 3 nhánh dữ liệu:  Job(4)  |  LSTM 3D(6 vùng × k)  |  Numerical
              │
              ▼
   Keras multi-input (Embedding + LSTM + Dense) -> Stratified 5-Fold
              │
              ▼
   Ensemble Feature Extraction (Lấy trung bình lớp fusion_dense 32-D)
              │
              ▼
    output/timeseries_features.parquet (8030 bệnh nhân) -> Sẵn sàng cho GNN Fusion
```

---

## 3. Ý nghĩa từng folder & file

| Đường dẫn | Vai trò |
|---|---|
| **`ĐỒ_ÁN.ipynb`** | **File chính** — toàn bộ pipeline timeseries (load → clean → encode → model → train). Xem chi tiết Mục 4. |
| `data/Main_data_fixed_Not_Encode_Mapping_New.xlsx` | **Dữ liệu thô** (8030 bệnh nhân) đã sửa lỗi font, giữ cột liên quan, đã match `file_name` với ảnh. Chưa mã hóa. |
| `output/Main_data_Processed_Encoded.xlsx` | **Sản phẩm** — bảng đã xử lý + mã hóa hoàn chỉnh (8030 dòng), sẵn sàng nạp vào model. |
| `doc/[ĐỐ ÁN TỐT NGHIỆP] - Document.docx` | Đề bài gốc + roadmap khái quát của thầy (nguồn chân lý về *yêu cầu*). |
| `doc/X-ray-overview.md` | Tài liệu tổng quan nhánh ảnh (file tham khảo song song). |
| `doc/Paper_128-...Occupational_Disease_Prediction.pdf` | **Bài báo tham khảo** cho bước fusion bằng Graph Convolutional Network. |
| `doc/ts-overview.md` | **File này.** |
| `README.md` | Tên + mô tả 1 dòng của đề tài. |
| `.gitignore` | Bỏ qua `__pycache__`, file tạm, data nặng/riêng tư. |

---

## 4. Chi tiết pipeline trong `ĐỒ_ÁN.ipynb` (theo từng cell)

| Cell | Nội dung | Điểm cần nhớ |
|---|---|---|
| **Mount Drive** | `drive.mount` (Colab) | Notebook thiết kế để chạy trên **Google Colab** (hoặc chạy offline). |
| **1. Load + lọc cột** | Tải raw từ Google Sheet, `df_raw`; drop cột định danh (`id, tinh, hoten, sdt, sobh`), nhãn chi tiết (`bnncuthe`), và **cột đọc phim X-quang** (`chatluongphim, ketqua, matdotonthuong, kichthuoctt, tonthuongkhac`) để **tránh rò rỉ sang nhánh ảnh**; drop nhóm khảo sát `B/C/D/E*` và `F1-3, F6-13` (giữ lại `F4*` = tần suất dùng bảo hộ). | Chủ ý: loại thông tin phim X-quang khỏi nhánh timeseries để 2 nhánh độc lập. |
| **2. Fill missing** | Text thiếu → `"không"`, số thiếu → `0`; sửa outlier `F4gang=6.0` về mode; **tách `file_name` ra mảng riêng** rồi bỏ khỏi bảng feature (dành cho nhánh ảnh map ảnh). | `file_name` là cầu nối 2 nhánh — được giữ riêng, không đưa vào model TS. |
| **3. Feature Engineering + Encoding** | - **Binary (0/1):** `gioitinh`, `longnguc`, nhóm "Khong/NaN→0 else→1" (rất nhiều cột triệu chứng + 6 vùng phổi `rungt/rungg/goduc/vt*`), `khoangls/rungthan/go/riraopn` ("binh thuong"→0). <br>- **Ordinal:** `tdho, tsho, loaidom, mdkhotho...` map thứ bậc; `F4*` map tần suất bảo hộ (0–4). <br>- **One-hot:** `A6, A7, A10, A11`. <br>- **Số:** đổi dấu phẩy→chấm, `SimpleImputer(mean)` + `StandardScaler`. <br>- **Tuổi động:** Tính tuổi bằng hiệu của năm hiện tại và năm sinh (`datetime.now().year - namsinh`) thay vì hardcode 2026. | `bnn` được loại hoàn toàn khỏi các cột đặc trưng để tránh rò rỉ dữ liệu. |
| **4. Phân loại nghề 10 cấp** | Chuẩn hóa text tiếng Việt (bỏ dấu), rule-based keyword → phân 4 cột nghề (`cviec, pxuong, cviec1, cviec2`) thành **ID nguyên 0–9** theo mức độ phơi nhiễm bụi/hóa chất (lv1 = luyện/đúc/khai thác… nặng nhất → lv9 = không đi làm). | Đây là feature engineering thủ công đáng chú ý; dùng làm input cho lớp Embedding. |
| **5. Tách 3 nhánh dữ liệu** | - **LSTM 3D:** gom 7 nhóm cột đo theo 6 vùng phổi (`rungt, rungg, goduc, vtam, vtno, vtrit, vtngay`) → tensor `[N, 6 vùng, k đặc trưng]`. <br>- **Numerical:** các cột số còn lại (loại target/job/seq/`tiensuhh`). <br>- **Job:** 4 cột nghề (ID). <br>- **Chia dữ liệu:** Chia 85% dữ liệu làm tập Development (cho K-Fold) và 15% làm tập Test Hold-out tĩnh. | Đã loại bỏ hoàn toàn `bnn` và các cột nhạy cảm khỏi tập Feature. |
| **6. Model Keras đa nhánh** | Đóng gói kiến trúc trong hàm `build_model(n_seq_features, n_num_features, n_job_slots)` để dễ dàng tái tạo sạch trong K-Fold. Nhánh Job sử dụng `Flatten()` thay vì `GlobalAveragePooling1D()` để giữ nguyên đặc trưng của từng ô nghề độc lập. | Lớp áp chót `fusion_dense` thiết lập 32 chiều nén thông tin lâm sàng. |
| **7. Train + Đánh giá** | - Huấn luyện qua **Stratified 5-Fold Cross Validation** trên tập Dev.<br>- Chuẩn hóa imputer/scaler fit độc lập bên trong từng fold để chống rò rỉ.<br>- Tự động quét ngưỡng chọn tối ưu trên Out-of-Fold (OOF) Validation theo **F2-Score** (chọn ngưỡng tối ưu `0.81` để ưu tiên Recall lâm sàng).<br>- Đánh giá Ensemble trên tập Test Hold-out: ROC-AUC = `0.9384`, PR-AUC = `0.3502`, Precision = `38.71%`, Recall = `37.50%`. | Vẽ và xuất biểu đồ `learning_curves.png` và `confusion_matrix.png` ra thư mục `output/`. |
| **8. Trích xuất đặc trưng** | Trích xuất đặc trưng Ensemble 32 chiều bằng cách trung bình hóa kết quả lớp `fusion_dense` của 5 mô hình fold cho toàn bộ 8030 bệnh nhân. | Xuất ra file `output/timeseries_features.parquet` (kèm `file_name`) và `output/timeseries_features.npy` phục vụ GNN Fusion. |

---

## 5. Roadmap nhánh Timeseries — Trạng thái hiện tại

| Bước | Việc | Trạng thái |
|---|---|---|
| **0. Chuẩn bị data thô** | Sửa lỗi font, giữ cột liên quan, match `file_name` với ảnh → `Main_data_fixed_Not_Encode_Mapping_New.xlsx`. | ✅ Xong |
| **1. Lọc cột rác + chống rò rỉ** | Bỏ cột định danh, nhãn chi tiết `bnncuthe`, cột đọc phim X-quang, nhóm khảo sát thừa. | ✅ Xong |
| **2. Fill missing + làm sạch** | Điền thiếu theo rule, sửa outlier, tách `file_name`. | ✅ Xong |
| **3. Mã hóa toàn bộ** | Mã hóa nhị phân / phân loại / scale số + phân loại nghề 10 cấp. Tính tuổi động theo năm hiện tại (datetime.now().year). | ✅ Xong |
| **4. Định hình data 3D + đa nhánh** | Tensor `[N, 6 vùng, k]` cho LSTM; tách job/numerical; split tĩnh 15% Test. | ✅ Xong |
| **5. Model + K-Fold** | Huấn luyện Stratified 5-Fold Cross Validation để triệt tiêu dao động ngẫu nhiên và nâng cao độ tổng quát hóa. | ✅ Xong |
| **6. Khắc phục rò rỉ & Sai nhãn** | Thiết lập target chuẩn là `bnn`, loại hoàn toàn `bnn` khỏi feature đầu vào, chống rò rỉ chuẩn hóa bằng cách fit scaler/imputer trong fold. | ✅ Xong |
| **7. Đánh giá Ensemble & Quét F2** | Đánh giá ensemble trên tập Test bằng trung bình dự đoán 5 fold; Tối ưu hóa ngưỡng tự động qua OOF F2-Score (chọn ngưỡng 0.81). | ✅ Xong |
| **8. Xuất vector đặc trưng cho fusion** | Trích xuất đặc trưng Ensemble 32 chiều bằng cách trung bình hóa kết quả lớp `fusion_dense` của 5 mô hình fold, xuất ra parquet và npy. | ✅ Xong |

---

## 6. Lịch sử sửa đổi & Giải quyết các lỗi lớn

Mô hình đã được tối ưu hóa toàn diện để giải quyết triệt để các vấn đề phương pháp luận trước đây:

1. **Đã sửa nhãn mục tiêu:** Target được trỏ chính xác về cột `bnn` (Bệnh nghề nghiệp) thay vì `benhhh` (Bệnh hô hấp thông thường).
2. **Loại bỏ rò rỉ dữ liệu (Data Leakage):**
   * Đã loại bỏ hoàn toàn `bnn` khỏi danh sách các đặc trưng đầu vào.
   * Quá trình chuẩn hóa số liệu (`StandardScaler`, `SimpleImputer`) được thực hiện riêng biệt bên trong vòng lặp K-Fold (chỉ `fit` trên tập huấn luyện của fold đó và `transform` sang tập validation/test).
3. **Cải tiến chiến lược chống mất cân bằng:** Tắt bỏ ngẫu nhiên undersampling trên tập Train (tránh làm mất đi các mẫu dữ liệu quý giá) và chuyển hẳn sang sử dụng `class_weights` kết hợp với quét ngưỡng tối ưu hóa tự động.
4. **Tối ưu hóa ngưỡng chẩn đoán lâm sàng bằng F2-Score:** Thay vì tối đa hóa F1-Score toán học thuần túy (vốn đẩy ngưỡng lên quá cao `0.90` làm Recall tập Test giảm sâu xuống `28.1%`), hệ thống đã chuyển sang tối đa hóa **F2-Score** (ưu tiên Recall gấp đôi). Nhờ vậy, ngưỡng tối ưu tự động chọn được là **`0.81`**, nâng Recall tập Test lên **`37.50%`** và giữ Precision ở mức tốt **`38.71%`**.
5. **Đồng nhất kết quả (Reproducibility):** Cài đặt cố định seed toàn cục (`random`, `numpy`, `tensorflow`) giúp kết quả các lần chạy lặp lại chính xác 100%.

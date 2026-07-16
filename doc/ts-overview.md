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
   Keras multi-input:  Embedding + LSTM + Dense  → concat → sigmoid
              │
              ▼
   (dự kiến) lấy lớp áp chót làm  →  vector TS  →  FUSION
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

> ⚠️ **Lưu ý nguồn dữ liệu:** notebook (Cell "1") đang **tải trực tiếp từ Google Sheet online** qua `requests`, **không** đọc file `data/...xlsx` trong repo. File trong `data/` là bản đối chiếu/offline. Khi chạy lại nên thống nhất 1 nguồn (xem ghi chú Mục 6).

---

## 4. Chi tiết pipeline trong `ĐỒ_ÁN.ipynb` (theo từng cell)

| Cell | Nội dung | Điểm cần nhớ |
|---|---|---|
| **Mount Drive** | `drive.mount` (Colab) | Notebook thiết kế để chạy trên **Google Colab**. |
| **1. Load + lọc cột** | Tải raw từ Google Sheet, `df_raw`; drop cột định danh (`id, tinh, hoten, sdt, sobh`), nhãn chi tiết (`bnncuthe`), và **cột đọc phim X-quang** (`chatluongphim, ketqua, matdotonthuong, kichthuoctt, tonthuongkhac`) để **tránh rò rỉ sang nhánh ảnh**; drop nhóm khảo sát `B/C/D/E*` và `F1-3, F6-13` (giữ lại `F4*` = tần suất dùng bảo hộ). | Chủ ý: loại thông tin phim X-quang khỏi nhánh timeseries để 2 nhánh độc lập. |
| **2. Fill missing** | Text thiếu → `"không"`, số thiếu → `0`; sửa outlier `F4gang=6.0` về mode; **tách `file_name` ra mảng riêng** rồi bỏ khỏi bảng feature (dành cho nhánh ảnh map ảnh). | `file_name` là cầu nối 2 nhánh — được giữ riêng, không đưa vào model TS. |
| **3. Feature Engineering + Encoding** | - **Binary (0/1):** `gioitinh`, `longnguc`, nhóm "Khong/NaN→0 else→1" (rất nhiều cột triệu chứng + 6 vùng phổi `rungt/rungg/goduc/vt*`), `khoangls/rungthan/go/riraopn` ("binh thuong"→0). <br>- **Ordinal:** `tdho, tsho, loaidom, mdkhotho...` map thứ bậc; `F4*` map tần suất bảo hộ (0–4). <br>- **One-hot:** `A6, A7, A10, A11`. <br>- **Số:** đổi dấu phẩy→chấm, `SimpleImputer(mean)` + `StandardScaler`. <br>- Giữ nguyên `cviec, cviec1, cviec2, pxuong` để xử lý ở cell sau. | `bnn` **cũng nằm trong nhóm binary** → bị mã hóa thành cột số 0/1 (xem cảnh báo Mục 6). |
| **4. Phân loại nghề 10 cấp** | Chuẩn hóa text tiếng Việt (bỏ dấu), rule-based keyword → phân 4 cột nghề (`cviec, pxuong, cviec1, cviec2`) thành **ID nguyên 0–9** theo mức độ phơi nhiễm bụi/hóa chất (lv1 = luyện/đúc/khai thác… nặng nhất → lv9 = không đi làm). | Đây là feature engineering thủ công đáng chú ý; dùng làm input cho lớp Embedding. |
| **5. Tách 3 nhánh dữ liệu** | - **LSTM 3D:** gom 7 nhóm cột đo theo 6 vùng phổi (`rungt, rungg, goduc, vtam, vtno, vtrit, vtngay`) → tensor `[N, 6 vùng, k đặc trưng]`. <br>- **Numerical:** các cột số còn lại (loại target/job/seq/`tiensuhh`). <br>- **Job:** 4 cột nghề (ID). <br>- **Target = `benhhh`** ⚠️ (xem Mục 6). Split **train/val 70/30** (`stratify=y`, seed 42); tính `class_weight` balanced. | **Không có test set riêng**, chỉ train/val. |
| **6. Model Keras đa nhánh** | 3 input → 3 nhánh: Embedding(10→8)+Pool+Dense(16) cho job; LSTM(32)+Dense(16) cho seq; Dense(16) cho số → **Concatenate** → Dense(32)+Dropout(0.3) → **Dense(1, sigmoid)**. Optimizer Adam(1e-3), loss `binary_crossentropy`, metric AUC. | Nhánh này **tự fusion 3 nguồn TS** rồi ra output phân loại — chưa xuất vector cho GNN (xem Mục 5 & 6). |
| **7. Train + Đánh giá** | `model.fit` 50 epoch, batch 64, `class_weight`; vẽ learning curve loss/AUC (lưu PNG); đánh giá trên val: confusion matrix + classification report (threshold 0.5). | Kết quả **chưa được lưu trong notebook** (cell chưa có output) → cần chạy để lấy số. |

---

## 5. Roadmap nhánh Timeseries — từ đầu tới sẵn sàng fusion

| Bước | Việc | Trạng thái |
|---|---|---|
| **0. Chuẩn bị data thô** | Sửa lỗi font, giữ cột liên quan, match `file_name` với ảnh → `Main_data_fixed_Not_Encode_Mapping_New.xlsx`. | ✅ Xong |
| **1. Lọc cột rác + chống rò rỉ** | Bỏ cột định danh, nhãn chi tiết `bnncuthe`, cột đọc phim X-quang, nhóm khảo sát thừa. | ✅ Xong |
| **2. Fill missing + làm sạch** | Điền thiếu theo rule, sửa outlier, tách `file_name`. | ✅ Xong |
| **3. Mã hóa toàn bộ** | Binary / ordinal / one-hot / scale số + phân loại nghề 10 cấp → `Main_data_Processed_Encoded.xlsx`. | ✅ Xong |
| **4. Định hình data 3D + đa nhánh** | Tensor `[N, 6 vùng, k]` cho LSTM; tách job/numerical; split train/val. | ✅ Xong |
| **5. Model + train** | Keras đa input (Embedding + LSTM + Dense) → phân loại nhị phân. | 🟡 Có code, **cần chạy & lưu kết quả** (metrics chưa có trong notebook) |
| **6. Sửa target về `bnn` + thêm test set** | Đổi `TARGET_COL` sang `bnn`, loại `bnn` khỏi feature, thêm tập test hold-out, đánh giá lại. | ⬜ **Cần làm** (xem Mục 6) |
| **7. Xuất vector đặc trưng cho fusion** | Bỏ lớp `sigmoid`, lấy đầu ra lớp áp chót (VD `fusion_dense` 32-d hoặc `fusion_concat`) làm **vector TS** cho 8030 bệnh nhân; lưu kèm `id`/`file_name` để nối với vector ảnh. | ⬜ **CHƯA có — cầu nối sang fusion** |
| **8. Bàn giao fusion** | Đưa vector TS + vector ảnh vào GNN, tạo node/edge, train predict `bnn`. | ⬜ Thuộc nhánh Fusion |

**Việc cần làm tiếp theo rõ ràng nhất:** (a) chạy notebook để có kết quả baseline; (b) **sửa target về `bnn`** đúng đề bài; (c) viết bước **xuất vector đặc trưng** — hiện model kết thúc ở `sigmoid`, chưa expose vector trung gian cho GNN.

---

## 6. ⚠️ Ghi chú: mâu thuẫn với docx & các lỗi/điểm chưa hợp lý

> Các điểm cần biết để báo cáo cho khớp và tránh sai lệch kết quả. Sắp theo mức độ quan trọng.

1. **🔴 SAI NHÃN MỤC TIÊU — nghiêm trọng nhất.**
   - *docx:* nêu rõ **`bnn`** là "Kết luận cuối cùng Có/Không mắc bệnh nghề nghiệp… **Đây là nhãn sẽ dùng làm Output**".
   - *Thực tế:* trong notebook (cell tách nhánh) đặt `TARGET_COL = "benhhh"`. Mà theo chính docx, `benhhh` = "**Bệnh hô hấp hiện tại mắc phải**" — một triệu chứng/tình trạng hô hấp hiện thời, **không phải** kết luận bệnh nghề nghiệp. → Model đang dự đoán **sai bài toán**. Cần đổi `TARGET_COL` về `bnn`.

2. **🔴 RÒ RỈ NHÃN (label leakage) — `bnn` bị dùng làm feature.**
   - Trong cell mã hóa, `bnn` nằm trong danh sách `binary_khong_cols` nên bị biến thành cột số 0/1. Ở cell tách nhánh, `exclude_cols` chỉ loại `{benhhh, tiensuhh}` (+ seq/job), **không loại `bnn`** → `bnn` lọt vào `X_num_all` như một **feature đầu vào**.
   - Vì `bnn` (kết luận bệnh nghề nghiệp) tương quan rất mạnh với `benhhh` (bệnh hô hấp), việc để `bnn` làm input khiến model "nhìn trộm đáp án" → **metrics bị thổi phồng**. Kể cả sau khi đổi target về `bnn`, phải **loại `bnn` khỏi feature**.

3. **🟠 Nhánh này đã "fusion" nội bộ, nhưng chưa xuất vector cho GNN.**
   - *docx:* nhánh TS nên cho ra **1 vector đặc trưng** để bước sau fusion với ảnh **trên GNN**.
   - *Thực tế:* model Keras kết thúc bằng `Dense(1, sigmoid)` — tức là **tự phân loại end-to-end**, chưa expose vector trung gian. Muốn "sẵn sàng fusion" cần lấy đầu ra lớp áp chót làm vector TS (Bước 7). Ngoài ra docx hình dung TS = **LSTM (3D) + Dense (tĩnh)** ghép 2 vector; notebook thêm **nhánh thứ 3 (Embedding nghề nghiệp)** — mở rộng hợp lý nhưng khác mô tả docx, nên nêu trong báo cáo.

4. **🟠 Rò rỉ do chuẩn hóa trước khi split (data leakage nhẹ).**
   - `StandardScaler` và `SimpleImputer(mean)` được **fit trên toàn bộ dữ liệu** (cell mã hóa & cell tách nhánh) **trước khi** chia train/val. Nghĩa là thống kê của tập val đã "lọt" vào bước scale/impute. Chuẩn: fit trên **train**, rồi transform val/test. Với báo cáo nghiêm túc nên sửa để tránh lạc quan giả.

5. **🟠 Chỉ có train/val, thiếu test set hold-out.**
   - Notebook chỉ chia **70/30 train–val** và báo cáo trên val. Nhánh ảnh có train/val/test riêng. Để so sánh & kết luận công bằng, nhánh TS nên có **tập test độc lập** (hoặc k-fold CV) — nhất là khi target/feature còn đang được sửa.

6. **🟡 Nguồn dữ liệu không thống nhất.**
   - Notebook tải raw **trực tiếp từ Google Sheet online** (link + `requests`), trong khi repo có sẵn `data/Main_data_fixed_Not_Encode_Mapping_New.xlsx`. Hai nguồn có thể lệch phiên bản. Nên trỏ notebook về file trong `data/` (hoặc kiểm tra 2 nguồn khớp nhau) để tái lập được.

7. **🟡 Vài quy tắc mã hóa cần rà lại.**
   - Nhóm `khoangls/rungthan/go/riraopn`: rule `!= "binh thuong"` khiến **NaN → 1** (comment trong code cũng tự cảnh báo). Cần xác nhận NaN nên là 0 hay 1.
   - `namsinh` (năm sinh) đưa vào làm feature số và scale, thay vì đổi sang **tuổi** — cân nhắc, vì "năm sinh" ít ý nghĩa tuyến tính hơn "tuổi".
   - Phân loại nghề 10 cấp là **rule-based thủ công** (danh sách keyword) → dễ sai/khó bao phủ; nên kiểm tra tỉ lệ rơi vào `lv9`/`lv10` (không xác định) để đánh giá chất lượng.

8. **🟡 Kết quả train chưa được lưu.**
   - Các cell chưa có output đã chạy (kernel chưa khởi tạo). Chưa có con số AUC/accuracy baseline nào trong repo. Cần chạy hoàn chỉnh và lưu lại (metrics + learning curve) để đưa vào báo cáo.

9. **🟢 Tên mô hình fusion chưa thống nhất (giống ghi chú nhánh ảnh).**
   - docx nói chung **GNN (GCN/GAT)**; README/paper tham khảo dùng **GCN/DGCNN**. Chỉ là khác biệt tên gọi — cần chốt thống nhất khi sang nhánh fusion.

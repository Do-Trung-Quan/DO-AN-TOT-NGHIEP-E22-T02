# Chú thích thuật ngữ của đồ án

Tra cứu nhanh các thuật ngữ, từ tiếng Anh, viết tắt và tên biến hay dùng trong đồ án (đề cương PAG_V2, PG_update bản 3, các báo cáo Phase 0–4).
Mỗi mục: **tên** — tiếng Anh/viết tắt đầy đủ (nếu có) — nghĩa dễ hiểu — và nghĩa cụ thể trong đồ án này nếu khác nghĩa chung.

---

## 1. Dữ liệu và đơn vị phân tích

| Thuật ngữ | Nghĩa |
|---|---|
| **Lượt khám** | Một lần một người đến khám. Đồ án có 8.030 lượt khám của 7.024 người; 1.002 người khám 2–4 lần. Mỗi lượt khám là một dòng dữ liệu. |
| **Bệnh nhân / người** | Một con người, có thể có nhiều lượt khám. Mọi phép chia dữ liệu và bootstrap đều làm **theo người**. |
| **`id`** | Mã của một lượt khám (duy nhất trên mỗi dòng). Với ảnh, `img_id` = `id` của lượt khám có ảnh đó. |
| **`patient_uid`** | Mã định danh một người, tạo theo luật số điện thoại và các trường định danh (`scripts/patient_identity.py`). Là mã băm (hash) nên không lộ tên. Dùng để chia dữ liệu. |
| **`patient_group`** | Mã băm của (họ tên + năm sinh), công thức cũ. Chỉ dùng để đối chiếu với nhánh ảnh, không dùng để chia dữ liệu. |
| **Ghép nới lỏng** | Coi hai hồ sơ là cùng người nếu trùng họ tên + năm sinh + giới + tỉnh, không cần trùng số điện thoại (P0.5). |
| **Khóa ghép** | Giá trị dùng để ghép, ở đây là chuỗi "họ tên chuẩn hóa \| năm sinh \| giới \| tỉnh". |
| **PII** | Personally Identifiable Information: thông tin định danh cá nhân (họ tên, số điện thoại, tên file ảnh có tên người). Không được đưa vào báo cáo, git hay dịch vụ bên ngoài. |
| **Mã ẩn danh** | Mã số thay cho danh tính (`patient_uid`, `img_id`), dùng trong sổ kiểm toán. |
| **Phim / ảnh** | Phim X-quang ngực thẳng (CXR). Chỉ 1.835 lượt khám có ảnh. |
| **CXR** | Chest X-Ray: X-quang ngực. |
| **Lâm sàng** | Thông tin do hỏi và khám: tuổi, nghề, phơi nhiễm bụi, hút thuốc, triệu chứng, hô hấp ký… |
| **Modality (phương thức)** | Một loại nguồn dữ liệu. Đồ án có hai modality: ảnh và lâm sàng. **Đa phương thức** (multimodal) là dùng cả hai. **Thiếu modality** là người không có phim. |
| **Lô chụp (batch)** | Một đợt chụp: cùng cơ sở, cùng ngày, cùng máy. Mã lô là ngày chụp (ví dụ `20181219`). Có 18 lô, kể cả lô `unknown` (không xác định được ngày). |
| **Tỉnh (`province`)** | Nơi thu thập. Đồng Nai không có ảnh; Phú Yên có 5 ảnh. |
| **Set A / Set B (Silicodata)** | Các tập dữ liệu ảnh bên ngoài. SetA được dùng để fine-tune nguồn ảnh "control". Mask Set B bị cấm dùng cho phân đoạn phổi. |
| **Thư mục `sillicosis` cũ** | Thư mục 535 ảnh dương cũ. Toàn bộ 462 ảnh dương hiện tại là bản sao giống từng byte của thư mục này (rủi ro shortcut theo nguồn ảnh). |
| **SHA / SHA-256** | Hàm băm cho ra "dấu vân tay" của file. Hai file trùng SHA là giống hệt nhau từng byte. |
| **Hash cấu hình** | 16 ký tự đầu của SHA-256 của `phase0/config.json`. Đổi một chữ trong cấu hình thì hash đổi, nên dùng để chứng minh kết quả chạy với đúng cấu hình đã khai báo trước. |

## 2. Nhãn và các trường lâm sàng – ILO

| Thuật ngữ | Nghĩa |
|---|---|
| **Nhãn (label)** | Đáp án mô hình phải đoán. |
| **`ketqua`** | Nhãn chính: kết quả đọc phim theo ILO. 1 = dương tính (có dấu hiệu bụi phổi), 2 = âm tính. Có cho cả 8.030 lượt khám. Chưa có codebook chính thức. |
| **Dương / âm tính** | Có bệnh / không có bệnh theo nhãn. |
| **`bnn`** | Trường tiền sử "đã được công nhận bệnh nghề nghiệp" (Có/Không), có trước đợt khám. Không phải kết quả đọc phim của đợt này. |
| **Nhóm 57 ca (55 người)** | 57 lượt khám có ảnh mang `bnn=Có` nhưng `ketqua=2`, tức bất nhất giữa tiền sử và đọc phim (P0.4). Không sửa nhãn; dùng phân tích độ nhạy. |
| **ILO** | International Labour Organization (Tổ chức Lao động Quốc tế). Có bảng phân loại chuẩn để đọc phim bụi phổi, chia phổi thành 6 vùng. |
| **6 vùng ILO** | RU, RM, RL = phổi phải trên / giữa / dưới; LU, LM, LL = phổi trái trên / giữa / dưới. |
| **Silicosis** | Bụi phổi silic, thường xuất hiện hai bên và ưu thế ở vùng trên. |
| **Trường ILO** | `chatluongphim` (chất lượng phim), `matdotonthuong` (mật độ tổn thương), `kichthuoctt` (kích thước tổn thương: p/q/r/s/t/u), `vungtonthuong` (vùng có tổn thương), `tonthuongkhac` (tổn thương khác). Cấm làm đặc trưng, vì chỉ có giá trị khi `ketqua=1`, nên riêng việc có hay thiếu đã tiết lộ nhãn. |
| **`vungtonthuong`** | Vùng phổi bác sĩ ghi có tổn thương. Chỉ dùng để đánh giá attention theo vùng (H4), tuyệt đối không đưa vào huấn luyện. |
| **Khám thực thể** | Nghe/gõ phổi: rung thanh (`rungt`, `rungg`), gõ đục (`goduc`), các loại ran (`vtam`, `vtno`, `vtrit`, `vtngay`). Từng có khối 6 vùng × 7 dấu hiệu (nhánh "ts"); P0.6 cho thấy gần như trống, nên bản 3 gộp thành 1 node. |
| **Hô hấp ký** | Đo chức năng hô hấp. `fvctt`, `fev1tt`: giá trị đo tuyệt đối của FVC và FEV1. `fvclt`, `fev1lt`: giá trị lý thuyết (dự đoán), bị cấm dùng vì không tái lập được mẫu số. |
| **FVC / FEV1** | Forced Vital Capacity: dung tích sống gắng sức. Forced Expiratory Volume in 1 second: thể tích thở ra gắng sức trong giây đầu. FEV1/FVC là tỉ số giữa hai giá trị. |
| **WHITELIST / REVIEW (Gate 0)** | Danh sách biến được phép dùng / biến cần thầy duyệt, xác định ở Gate 0. |
| **Proxy** | Biến "thay mặt" cho nhãn: không phải nhãn nhưng tương quan mạnh với nhãn, ví dụ `bnn`. |
| **Codebook** | Từ điển dữ liệu giải thích ý nghĩa chính thức của từng mã. |

## 3. Chia dữ liệu và chống rò rỉ

| Thuật ngữ | Nghĩa |
|---|---|
| **Dev / test** | Dev (6.882 lượt) dùng để thử nghiệm và chọn mô hình. Test (1.148 lượt) bị khóa, chỉ mở một lần ở cuối khi thầy đã duyệt cấu hình. |
| **Fold / k-fold** | Chia dev thành k phần (ở đây 5). Mỗi lần học trên 4 phần, chấm trên phần còn lại, lặp 5 lần. Còn gọi là **kiểm định chéo** (cross-validation, CV). |
| **`fold_id`** | Số fold của mỗi dòng (0–4); `-1` nghĩa là thuộc test. |
| **StratifiedGroupKFold** | Cách chia fold vừa **giữ tỉ lệ dương/âm** gần nhau giữa các fold (stratified, phân tầng) vừa **giữ trọn mỗi nhóm** (một người) trong một fold (group). |
| **Out-of-fold (OOF)** | Điểm dự đoán của một người do mô hình **chưa học trên người đó** cho ra. Chỉ điểm OOF mới dùng để đánh giá. |
| **Validation nội bộ** | Phần cắt ra từ 4 fold huấn luyện để chọn siêu tham số; không chọn trên fold dùng để báo cáo. |
| **Rò rỉ (leakage)** | Mô hình "nhìn trộm" thông tin lẽ ra không được biết, làm kết quả đẹp giả. **Rò rỉ chủ thể**: cùng một người nằm ở cả phần học lẫn phần chấm. |
| **Trong fold (in-fold)** | Mọi bước "học từ dữ liệu" (chuẩn hóa, điền thiếu, mã hóa, chọn tham số) chỉ làm trên phần huấn luyện của từng fold. |
| **Transductive** | Cách học có dùng dữ liệu (kể cả không nhãn) của phần sẽ chấm, ví dụ đồ thị chứa cả node test. Ngược lại là **inductive**. |
| **Shortcut (đường tắt)** | Mô hình dựa vào dấu hiệu phụ (chữ trên phim, lô, nguồn ảnh) thay vì tổn thương thật. |
| **Nhiễu theo lô (batch effect)** | Khác biệt giữa các lô do máy, cách chụp, cách xử lý, không do bệnh. |
| **Leave-One-Batch-Out (LOBO)** | Lần lượt bỏ hẳn một lô khỏi huấn luyện rồi chấm trên lô đó. Còn gọi là leave-batch-out (LOLO trong các file cũ). |
| **Seed** | Hạt giống ngẫu nhiên. Cố định seed (ở đây 42) thì kết quả ngẫu nhiên lặp lại y hệt. |
| **Phân tích độ nhạy (sensitivity analysis)** | Chạy lại khi bỏ hoặc thay một giả định (ví dụ bỏ 55 người nhóm 57) để xem kết luận có thay đổi không. Khác với "sensitivity" là chỉ số ở mục 6. |
| **Ablation** | Bỏ bớt một thành phần của mô hình để xem nó có đóng góp không. |
| **Sổ kiểm toán (audit log)** | File CSV ghi mọi quyết định xử lý dữ liệu: mã ẩn danh · vấn đề · bằng chứng · quyết định · quy tắc áp dụng. |
| **Cổng (gate)** | Điểm kiểm tra phải qua mới được làm tiếp; nộp báo cáo ngắn và chờ thầy duyệt. |
| **Khai báo trước (pre-registration)** | Ghi các tham số, tiêu chí, giả thuyết trước khi xem kết quả, để không "chỉnh theo kết quả". |

## 4. Xử lý ảnh (Phase 0 – P0.1)

| Thuật ngữ | Nghĩa |
|---|---|
| **Ảnh xám** | Ảnh một kênh sáng/tối, không màu. |
| **Đảo âm** | Ảnh lưu ngược màu: xương đen, phổi trắng. Phim chuẩn có phổi tối và bụng/xương sáng. |
| **Burn-in / chữ in chìm** | Chữ in trực tiếp lên phim: tên cơ sở, ngày chụp, đôi khi tên bệnh nhân. Chính là dấu hiệu của lô. |
| **OCR** | Optical Character Recognition: nhận dạng ký tự quang học. **Chỉ phát hiện** (detection): chỉ tìm vị trí có chữ, không đọc nội dung. |
| **Phân đoạn (segmentation)** | Cho mô hình tô ra pixel nào thuộc một vật thể, ở đây là phổi. |
| **Mask** | Ảnh đen trắng đánh dấu pixel nào thuộc phổi (1) hay không (0). |
| **torchxrayvision / PSPNet** | Thư viện Python có sẵn mô hình cho X-quang ngực. PSPNet là kiến trúc mạng phân đoạn dùng để tách phổi. |
| **Thành phần liên thông** | Một mảng pixel liền nhau trong mask. Giữ 2 thành phần lớn nhất tương ứng 2 lá phổi. |
| **Hộp bao (bounding box)** | Hình chữ nhật nhỏ nhất chứa vật thể. |
| **Lề (margin)** | Phần nới thêm quanh hộp bao (5%). |
| **Pad vuông** | Thêm viền đen cho ảnh thành hình vuông, tránh méo khi đổi kích thước. |
| **Resize** | Đổi kích thước ảnh (về 448 × 448). |
| **CenterCrop** | Cắt lấy phần giữa ảnh theo kích thước cố định, bất kể phổi nằm đâu. Cách cũ, bị cấm vì có thể làm mất đỉnh phổi. |
| **V0 / V1 / V2** | V0 = toàn ảnh đã che chữ (dự phòng). V1 = cắt theo hộp bao phổi + che chữ. V2 = V1 và xóa mọi pixel ngoài mask phổi. |
| **A_goc** | Đặc trưng ảnh cũ: ảnh chưa che chữ, Resize 512 → CenterCrop 448. |
| **QA** | Quality Assurance: kiểm tra chất lượng. **QA tự động**: script gắn cờ. **QA trực quan**: người xem và chấm. |
| **Cờ (flag)** | Dấu "nghi có vấn đề" gắn cho ảnh: `dao_am`, `khong_du_2_phoi`, `ty_le_trai_phai`, `cham_mep`, `con_chu`, `dien_tich_bat_thuong`. |
| **Đạt / một phần / hỏng** | Ba mức chấm QA trực quan. Cổng: ≥ 90% ảnh ngẫu nhiên đạt. |
| **`zone_missing`** | Cờ cho ảnh chấm "hỏng": dùng vector toàn ảnh (V0) thay cho V1/V2 và không có 6 node vùng. Không loại ảnh. |
| **Phân tầng (stratified)** | Chia mẫu theo nhóm (ở đây lô × nhãn) để mỗi nhóm đều có đại diện. |
| **MAD** | Median Absolute Deviation: độ lệch tuyệt đối trung vị, thước đo độ phân tán ít bị ảnh hưởng bởi ngoại lai. |
| **Trung vị (median)** | Giá trị ở giữa khi xếp từ nhỏ đến lớn. |
| **Shard** | Một phần nhỏ của kết quả lưu riêng (100 ảnh/shard) để chạy tiếp được khi bị ngắt. |
| **`.npz` / `.parquet` / `.csv` / `.json`** | Định dạng file: mảng numpy nén / bảng dữ liệu cột / bảng văn bản / dữ liệu có cấu trúc. |

## 5. Mô hình và học máy

| Thuật ngữ | Nghĩa |
|---|---|
| **Đặc trưng (feature)** | Con số đưa vào mô hình để mô hình dựa vào đó mà đoán. |
| **Embedding / vector** | Dãy số (ở đây 256 số) do mạng nơ-ron rút ra để tóm tắt một ảnh, một bệnh nhân… |
| **Encoder** | Phần mạng biến dữ liệu (ảnh) thành embedding. |
| **BioViL-T** | Mô hình ảnh–văn bản của Microsoft, học sẵn trên ảnh X-quang ngực; dùng làm encoder ảnh. |
| **Foundation model** | Mô hình nền lớn, học sẵn trên rất nhiều dữ liệu, dùng lại cho nhiều bài toán. |
| **Frozen** | Dùng encoder giữ nguyên trọng số, không học thêm. |
| **Fine-tune (tinh chỉnh)** | Học tiếp một mô hình có sẵn trên dữ liệu mới. **Control** = BioViL-T đã fine-tune trên SetA với nhãn `bnn`. |
| **Trọng số (weights) / checkpoint** | Các con số mô hình đã học; file lưu chúng (`.pt`, `.pth`). |
| **Hồi quy logistic (LR)** | Logistic Regression: mô hình đơn giản cho bài toán có/không, gồm tổng có trọng số của đặc trưng đổi thành xác suất. |
| **L0 / L1 / L2 / L3b** | Các mô hình logistic mốc. L0 = chỉ lâm sàng; L1 = chỉ ảnh; L2 = lâm sàng + ảnh; L3b = L2 + đặc trưng hàng xóm. |
| **B0–B3b** | Các baseline của báo cáo fusion GAT (Phase 3). |
| **Z1** | Logistic trên 6 vector vùng nối lại, để so "chia vùng" với "vector toàn ảnh". |
| **X0 / X2** | XGBoost (mô hình cây quyết định tăng cường, mạnh với dữ liệu bảng) trên lâm sàng / lâm sàng + ảnh. |
| **T0 / T2** | FT-Transformer (Transformer cho dữ liệu bảng) trên lâm sàng / lâm sàng + token ảnh. |
| **Baseline** | Mô hình mốc để so sánh. |
| **MLP** | Multi-Layer Perceptron: mạng nơ-ron gồm vài lớp kết nối đầy đủ. |
| **Siêu tham số (hyperparameter)** | Tham số người chọn trước khi học (số lớp, learning rate, dropout…), khác với trọng số mô hình tự học. |
| **Learning rate / weight decay / dropout** | Tốc độ học / mức phạt trọng số lớn để chống quá khớp / tắt ngẫu nhiên một số nơ-ron khi học để chống quá khớp. |
| **AdamW** | Một thuật toán tối ưu (cách cập nhật trọng số) phổ biến. |
| **Early stopping** | Dừng học khi chỉ số trên tập validation không còn cải thiện. |
| **Loss / binary cross-entropy (BCE)** | Hàm mất mát: đo mô hình sai bao nhiêu. BCE là hàm mất mát chuẩn cho bài toán có/không. |
| **`class_weight="balanced"`** | Cho lớp ít mẫu (ca dương) trọng số lớn hơn khi học. Làm xác suất bị lệch, nên cần hiệu chuẩn lại. |
| **Quá khớp (overfitting)** | Mô hình học thuộc dữ liệu huấn luyện, kém trên dữ liệu mới. |
| **Probe** | Mô hình nhỏ dùng để "dò" xem một embedding chứa thông tin gì (ví dụ đoán lô). |
| **Tự giám sát (SSL)** | Self-Supervised Learning: học không cần nhãn. Ví dụ GraphCL, InfoGraph cho đồ thị. |
| **SVM (RBF)** | Support Vector Machine: tìm đường ranh giới tốt nhất giữa hai lớp; RBF là kiểu ranh giới cong. |
| **kNN** | k-Nearest Neighbors: k láng giềng gần nhất. Dùng để dựng đồ thị hoặc để phân loại/truy hồi ca giống nhau. |
| **Truy hồi ca tương tự (retrieval)** | Với một bệnh nhân, tìm những bệnh nhân cũ giống nhất. |
| **Precision@k** | Trong k ca được truy hồi, tỉ lệ ca có cùng nhãn. |

## 6. Đồ thị và GNN

| Thuật ngữ | Nghĩa |
|---|---|
| **Đồ thị (graph)** | Tập **node** (đỉnh) nối với nhau bằng **cạnh** (edge). |
| **GNN** | Graph Neural Network: mạng nơ-ron trên đồ thị; mỗi node tổng hợp thông tin từ node kề. |
| **Truyền tin (message passing)** | Quá trình node nhận và gộp thông tin từ hàng xóm. |
| **GAT** | Graph Attention Network: GNN trong đó node tự học mức "chú ý" tới từng hàng xóm. |
| **Graph transformer / HGT / R-GCN** | Các kiến trúc GNN. HGT (Heterogeneous Graph Transformer) và R-GCN (Relational GCN) xử lý đồ thị có nhiều loại node/cạnh. |
| **Attention** | Trọng số mô hình tự học cho biết nó "chú ý" thành phần nào nhiều. |
| **Patient-as-node** | Cách cũ: mỗi bệnh nhân là một node trong một đồ thị lớn nối các bệnh nhân. |
| **Patient-as-graph (PAG)** | Cách mới: mỗi lượt khám là một đồ thị nhỏ riêng (8.030 đồ thị), bài toán là **phân loại đồ thị**. |
| **PG** | Patient Graph: tên mô hình đề xuất ở bản 3. |
| **Node ảnh theo vùng** | 6 node tương ứng 6 vùng ILO, mang đặc trưng ảnh của vùng. |
| **Node nhóm lâm sàng** | 6 node: nhân khẩu–thể trạng · phơi nhiễm/nghề · hút thuốc · triệu chứng–tiền sử · khám thực thể · hô hấp ký. |
| **Node bối cảnh** | 1 node chứa tỉnh và năm thu thập; bỏ được trong ablation. |
| **Node bệnh nhân (node ảo)** | Node nối với mọi node khác để gom thông tin thành embedding bệnh nhân. |
| **E1–E5 (loại cạnh)** | E1 kề dọc (trên–giữa–dưới cùng một phổi); E2 đối xứng trái–phải cùng tầng; E3 lâm sàng–lâm sàng; E4 lâm sàng–vùng; E5 node bệnh nhân–mọi node. **E6** (ảnh–khám cùng vùng) đã bỏ ở bản 3. |
| **Masked average pooling** | Lấy trung bình feature map của encoder chỉ trên các pixel thuộc mask vùng, để ra vector cho từng vùng. |
| **Feature map** | Bản đồ đặc trưng không gian mà encoder tạo ra trước khi gộp thành một vector. |
| **Readout** | Bước lấy ra vector đại diện cho cả đồ thị (ở đây là vector của node bệnh nhân ở lớp cuối). |
| **Modality dropout** | Khi huấn luyện, bỏ ngẫu nhiên toàn bộ node ảnh với xác suất p, để mô hình học được cả trường hợp không có phim. |
| **Homophily** | Mức độ "giống nhau thì nối với nhau". Homophily lớp dương 1,25× nghĩa là hàng xóm của ca dương chỉ dương nhiều hơn chọn ngẫu nhiên 25%. |
| **Graph kernel (Wasserstein Weisfeiler–Lehman)** | Cách đo độ giống nhau giữa hai đồ thị để dùng với SVM. |
| **PG-shuffle / complete / noedge / global / noMD / noclin / noexam / nobnn / noctx / drop** | Các ablation của PG: xáo cạnh / nối đầy đủ / chỉ giữ E5 / 1 node ảnh toàn cục / không modality dropout / chỉ node ảnh / bỏ node khám / bỏ `bnn` / bỏ node tỉnh–năm / bỏ ảnh khi đánh giá. |

## 7. Đánh giá và thống kê

| Thuật ngữ | Nghĩa |
|---|---|
| **AUC / ROC-AUC** | Diện tích dưới đường ROC. Xác suất mô hình chấm một ca dương ngẫu nhiên cao hơn một ca âm ngẫu nhiên. 0,5 = đoán bừa, 1 = hoàn hảo. |
| **ROC** | Receiver Operating Characteristic: đường biểu diễn sensitivity theo (1 − specificity) khi thay đổi ngưỡng. |
| **PR-AUC** | Diện tích dưới đường Precision–Recall. Hợp với dữ liệu ít ca dương; phải đọc cùng **tỉ lệ dương cơ sở** (25% nhóm có ảnh, 8,1% nhóm không ảnh). |
| **AUC gộp** | AUC tính chung trên mọi lượt khám. |
| **AUC trong lô** | AUC tính riêng trong từng lô có đủ hai nhãn, rồi lấy trung bình có trọng số theo cỡ lô. Loại được phần nhờ "nhận ra lô". |
| **AUC OvR** | One-vs-Rest: với bài toán nhiều lớp, tính AUC "lớp này hay phần còn lại" cho từng lớp rồi lấy trung bình. |
| **Accuracy / accuracy cân bằng** | Tỉ lệ đoán đúng / trung bình tỉ lệ đúng của từng lớp (không bị lớp đông lấn át). |
| **Lớp đa số** | Mức accuracy khi luôn đoán lớp đông nhất, là mốc "đoán bừa có hiểu biết". |
| **Nhãn hoán vị (permutation)** | Xáo trộn ngẫu nhiên nhãn rồi chạy lại, để có mức kết quả khi không có thông tin. |
| **Sensitivity / specificity** | Độ nhạy: tỉ lệ ca dương được phát hiện. Độ đặc hiệu: tỉ lệ ca âm được xác định đúng. "Sensitivity tại specificity 0,80" là độ nhạy khi đặt ngưỡng sao cho độ đặc hiệu bằng 80%. |
| **Calibration (hiệu chuẩn)** | Xác suất mô hình đưa ra có khớp tần suất thật không (nói 30% thì đúng khoảng 30% số lần). |
| **Platt scaling / isotonic** | Hai cách hiệu chuẩn xác suất sau khi huấn luyện. |
| **Brier score** | Trung bình bình phương sai giữa xác suất dự đoán và nhãn thật; càng nhỏ càng tốt. So với Brier của "dự đoán bằng tỉ lệ nền". |
| **ECE** | Expected Calibration Error: sai số hiệu chuẩn trung bình theo các khoảng xác suất. |
| **Calibration slope / intercept** | Độ dốc / hệ số chặn của đường hiệu chuẩn; lý tưởng là 1 / 0. |
| **Δ (delta)** | Hiệu số giữa hai mô hình, ví dụ ΔAUC = AUC(B) − AUC(A). |
| **Ghép cặp (paired)** | So hai mô hình trên cùng bệnh nhân, cùng các lần bootstrap. Chính xác hơn so hai số độc lập. |
| **Bootstrap** | Bốc ngẫu nhiên lại dữ liệu (có hoàn lại) nhiều lần, tính lại chỉ số mỗi lần để ước lượng độ dao động. **Theo bệnh nhân**: bốc theo người. |
| **KTC 95%** | Khoảng tin cậy 95% (CI — confidence interval): khoảng mà giá trị thật có khả năng cao nằm trong đó. |
| **KTC cắt qua 0 / nằm trọn trên 0** | KTC của Δ chứa cả âm lẫn dương thì chưa kết luận được có khác biệt; toàn dương thì mới được nói có khác biệt. |
| **P(Δ>0)** | Tỉ lệ lần bootstrap có Δ dương. Không phải p-value. |
| **p-value** | Xác suất thấy kết quả ít nhất cực đoan như vậy nếu thật ra không có khác biệt. |
| **Hiệu chỉnh Holm** | Cách hiệu chỉnh p-value khi làm nhiều so sánh cùng lúc, để tránh "ăn may". |
| **Khám phá (exploratory)** | Phân tích không khai báo trước. Chỉ để gợi ý, không dùng để kết luận chính. |
| **Không kém hơn (non-inferiority)** | Kết luận mô hình mới không kém mô hình cũ quá một **biên** khai báo trước (ví dụ −0,02 AUC). |
| **Wilson (KTC của tỉ lệ)** | Cách tính khoảng tin cậy cho một tỉ lệ (ví dụ 48/50 ảnh đạt), chính xác hơn công thức thông thường khi mẫu nhỏ. |
| **Phân tầng khi báo cáo** | Báo cáo riêng theo nhóm (có ảnh / không ảnh, tỉnh, giới). |

## 8. Câu hỏi nghiên cứu và giả thuyết (bản 3)

| Mã | Nội dung |
|---|---|
| **RQ1 / H1** | Một PG duy nhất có ngang mô hình chuyên biệt tốt nhất ở từng nhóm (L1 cho nhóm có ảnh, L0 cho nhóm không ảnh)? Đạt khi cận dưới KTC của Δ > −0,02 AUC ở cả hai nhóm. |
| **RQ2 / H2, H3** | Cấu trúc đồ thị có đóng góp không? H2: PG so với PG-shuffle, KTC nằm trọn trên 0. H3: PG không kém T2 (cận dưới > −0,02). |
| **RQ3 / H4** | Attention trên 6 vùng có khớp `vungtonthuong`? Đạt khi AUROC attention > 0,60 và vượt baseline ngẫu nhiên. |
| **RQ4** | Embedding bệnh nhân có dùng được cho phân loại cổ điển và truy hồi ca tương tự? |
| **E1 / E2 / E3 (nhánh embedding)** | Lưu ý trùng tên với loại cạnh: E1 = embedding node bệnh nhân của PG; E2 = embedding tự giám sát (GraphCL/InfoGraph); E3 = graph kernel + SVM. |

## 9. Các hạng mục Phase 0

| Mã | Nội dung |
|---|---|
| **P0.1** | Kiểm tra nhiễu theo lô chụp: 4 chỉ số (probe lô · AUC trong lô · Δ lô + ảnh so với lô · LOBO). Là cổng chặn. |
| **P0.4** | Ý nghĩa nhãn `ketqua` và nhóm 57 ca `bnn=Có`, `ketqua=2`. |
| **P0.5** | Kiểm toán danh tính: chọn k trên dev, trùng SHA, ghép 462 ca dương, ghép nới lỏng (quy tắc 1%). |
| **P0.6** | Khảo sát dữ liệu khám theo vùng (đã đóng: gần như trống, nên bỏ E6). |
| **Gate 0 / Gate 1B** | Các cổng kiểm tra dữ liệu ở giai đoạn trước (danh sách biến, kiểm tra hô hấp ký). |
| **Kiểm chứng 30/09** | Đợt chạy lại checklist của thầy, phát hiện đồ thị kNN hỏng (93% cạnh cùng fold) và lỗi L3b. |
| **Bảng quyết định P0.1** | AUC trong lô ≥ 0,70 và KTC Δ trên 0 → tiếp tục; 0,60–0,70 → báo cáo PGS; < 0,60 → dừng nhánh ảnh. |

## 10. Công cụ và môi trường

| Thuật ngữ | Nghĩa |
|---|---|
| **Colab / T4** | Google Colab: môi trường chạy notebook trên đám mây. T4 là loại GPU miễn phí, có hạn mức sử dụng. |
| **GPU / CPU** | Bộ xử lý đồ họa (nhanh cho mạng nơ-ron) / bộ xử lý thông thường. |
| **Runtime** | Máy ảo Colab đang chạy. Khi bị thu hồi, mọi thứ trong `/content` mất; Drive vẫn còn. |
| **Notebook (`.ipynb`)** | File gồm các cell mã và chữ, chạy từng cell. |
| **Git / branch / commit / push / pull** | Hệ thống quản lý phiên bản mã / nhánh (ở đây `PAG`) / một lần lưu thay đổi / đẩy lên GitHub / kéo về. |
| **scikit-learn / PyTorch / PyTorch Geometric** | Thư viện học máy cổ điển / học sâu / mạng nơ-ron đồ thị. |
| **PGS** | Phó giáo sư, tức thầy hướng dẫn. |

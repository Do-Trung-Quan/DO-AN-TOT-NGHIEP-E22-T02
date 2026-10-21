26/09: Thực hiện fusion luôn trên và báo cáo kết quả  
Báo cáo tiến độ và kết quả fusion GAT trên ts \+ ảnh (control / frozen)

Nhắc lại phạm vi data và mục tiêu fusion:  
***Data**: Dữ liệu của đồ án gồm 8.030 lượt khám của công nhân tiếp xúc bụi silic. Trong số đó, chỉ 1.835 người có file ảnh X-quang, còn lại 6.195 người không có ảnh — chiếm 77%*  
***Mục tiêu**: tìm mô hình đa phương thức hiệu quả nhất để ghép hai nguồn thông tin — hồ sơ lâm sàng và ảnh X-quang — thành một mô hình chẩn đoán duy nhất, đồng thời tìm cách để những người không có phim vẫn được hưởng lợi từ thông tin ảnh.*

1\. Quy trình fusion

1. Xử lý trước các nguồn dữ liệu và dựng đồ thị bệnh nhân  
   \- Gom các dữ liệu rải rác cần cho việc dựng đồ thị về 1 bảng: vector đặc trưng của nhánh lâm sàng, bảng thông tin node (nhãn, cách chia dữ liệu, mã bệnh nhân), chỉ mục ảnh, và vector đặc trưng ảnh. Bước này ghép cả bốn thành một bảng 8.030 dòng, khoá nối là mã id.  
   \=\> file output: fusion\_dataset\_ketqua\_control.npz và fusion\_dataset\_ketqua\_frozen.npz  
     
   \- dựng đồ thị với mỗi bệnh nhân là 1 node, các cạnh nối bằng cách mỗi node tìm 10 người có hồ sơ lâm sàng (timeseries) giống nhất \- gọi là hàng xóm. Thêm cạnh cầu nối: mỗi người không có ảnh được nối thêm tới người có ảnh giống họ nhất, để đảm bảo không ai bị cô lập khỏi nguồn thông tin ảnh.  
   \=\> Kết quả:   
+ 58.133 cạnh, bậc trung bình 14,48  
+ 100% người không có ảnh chạm được ít nhất một hàng xóm có ảnh, trung bình là 3,23 người

	Điều đáng lo rút ra được:

+ **Chất lượng đồ thị giảm đáng kể so với lúc còn nhãn “bnn”**: *Chỉ số quan trọng nhất là **homophily lớp dương** — tức là "nếu một người mắc bệnh, thì bao nhiêu phần trăm hàng xóm của họ cũng mắc bệnh". Đo được **0,1501**, trong khi tỷ lệ bệnh chung là **0,1197**. Tức là chỉ **gấp 1,25 lần so với chọn bừa**. Đồ thị có thông tin, nhưng rất nhạt. Con số này là trần của mọi thứ mà cơ chế truyền tin trên đồ thị có thể đạt được.* 

  *(trong khi độ nâng của homophily lớp dương của nhãn bnn là **\~ x15** \- vì nhãn bnn đặc biệt thích hợp cho timeseries thứ mà đóng góp chủ yếu để dựng lên đồ thị)*

2. Dựng trước 8 baseline model để có mốc so sánh với GAT  
   Bốn mô hình mạng nơ-ron:  
   \- **B0 — chỉ dùng hồ sơ lâm sàng.** Trả lời: lâm sàng một mình làm được tới đâu?  
   \- **B1 — chỉ dùng ảnh**, và chỉ chạy trên 1.571 người có ảnh trong tập dev. Trả lời: ảnh một mình làm được tới đâu?  
   \- **B2 — dùng cả lâm sàng lẫn ảnh (nối chay concatenate)**, không dùng đồ thị. Trả lời: ghép hai nguồn có hơn từng nguồn riêng lẻ không?  
   \- **B3** — **như B2, cộng thêm vector ảnh trung bình của các hàng xóm (256 chiều)**. Trả lời: mượn cả biểu diễn ảnh của hàng xóm có ích không?  
   \- **B3b** — **như B2, nhưng chỉ cộng thêm điểm nguy cơ trung bình của các hàng xóm (1 con số).** Trả lời: mượn kết luận của hàng xóm thì sao?  
     
   **cách đánh giá**: chỉ dùng tập dev 6.882 node, chia sẵn thành 5 fold theo nhóm bệnh nhân (không ai nằm ở hai fold), mỗi fold huấn luyện 3 lần với 3 seed khác nhau rồi lấy trung bình dự đoán, và mọi con số đều kèm khoảng tin cậy 95% lấy mẫu lại theo bệnh nhân.  
     
   \=\> KẾT QUẢ:

	**Nhóm CÓ ảnh — 1.571 người, 393 ca dương (tỷ lệ 25%)**

| Mô hình | ROC-AUC | PR-AUC |
| :---- | :---- | :---- |
| B0 — chỉ lâm sàng | 0,645 \[0,615–0,676\] | 0,393 |
| B2 — lâm sàng \+ ảnh | 0,744 \[0,717–0,771\] | 0,506 |
| B3 — \+ vector ảnh hàng xóm | 0,733 \[0,705–0,759\] | 0,501 |
| B3b — \+ điểm nguy cơ hàng xóm | 0,738 \[0,711–0,764\] | 0,499 |
| L1 — hồi quy logistic, chỉ ảnh | 0,760 \[0,732–0,788\] | 0,518 |
| L2 — hồi quy logistic, lâm sàng \+ ảnh | 0,754 \[0,728–0,780\] | 0,516 |
| L3b — hồi quy logistic \+ điểm hàng xóm | 0,753 \[0,726–0,779\] | 0,515 |
| B1 — mạng nơ-ron, chỉ ảnh | 0,749 \[0,723–0,777\] | 0,518 |
| Nhánh ts gốc, để tham chiếu | 0,643 \[0,610–0,675\] | 0,391 |

**Nhóm KHÔNG ảnh — 5.311 người, 430 ca dương (tỷ lệ 8,1%)**

| Mô hình | ROC-AUC | PR-AUC |
| :---- | :---- | :---- |
| B0 — chỉ lâm sàng | 0,618 \[0,593–0,645\] | 0,115 |
| B2 — lâm sàng \+ ảnh | 0,632 \[0,604–0,657\] | 0,124 |
| B3 — \+ vector ảnh hàng xóm | 0,627 \[0,599–0,653\] | 0,124 |
| B3b — \+ điểm nguy cơ hàng xóm | 0,640 \[0,613–0,665\] | 0,131 |
| L2 — hồi quy logistic, lâm sàng \+ ảnh | 0,651 \[0,625–0,677\] | 0,140 |
| L3b — hồi quy logistic \+ điểm hàng xóm | 0,648 \[0,622–0,674\] | 0,139 |
| Nhánh ts gốc, để tham chiếu | 0,616 \[0,588–0,644\] | 0,128 |

**Toàn bộ dev — 6.882 người, 823 ca dương: B0 0,633 · B2 0,713 · B3 0,711 · B3b 0,716 · L2 0,729 · L3b 0,727 · nhánh ts gốc 0,641.**  
Bốn cổng kiểm định, so sánh ghép cặp trên cùng bệnh nhân:

| Câu hỏi | Chênh lệch | KTC 95% | P | Kết quả |
| :---- | :---- | :---- | :---- | :---- |
| Ghép lâm sàng vào có làm hỏng mô hình ảnh không? | −0,0050 | \[−0,0225; \+0,0127\] | 28% | ĐẠT (không hỏng) |
| Hàng xóm có giúp nhóm không ảnh không — bản B3 (vector)? | \+0,0081 | \[−0,0145; \+0,0324\] | 74% | TRƯỢT |
| Hàng xóm có giúp nhóm không ảnh không — bản B3b (điểm)? | \+0,0220 | \[+0,0041; \+0,0393\] | 99% | ĐẠT |
| Mô hình sâu có hơn hồi quy logistic không? | −0,0107 | \[−0,0291; \+0,0068\] | 12% | Không hơn |

Lặp lại với bộ ảnh frozen (phase3\_baselines\_frozen.md) để kiểm tra tính ổn định: nhóm không ảnh — B0 0,618 · B3 0,603 · B3b 0,634 · L2 0,647. Cổng B3b: \+0,0153, KTC \[−0,0036; \+0,0325\], P \= 96%. Cổng B3: −0,0155, tức mượn vector còn làm hại.

\=\> Kết luận của Phase 3: 

+ mượn kết luận của hàng xóm thì hiệu quả và lặp lại được ở cả hai bộ ảnh; mượn cả vector biểu diễn thì không, ở cả hai bộ.  
+ việc thêm thông tin từ hàng xóm không giúp được gì và thậm chí đang làm tụt kết quả cả nhánh ảnh và nối chay ảnh \+ ts  
+ Hồi quy logistic chỉ ảnh và mạng nơ ron chỉ ảnh vẫn giữ kqua tốt nhất, còn cứ có sự tham gia của timeseries thì kết quả lại bị kéo xuống (vì nhãn ketqua hiện tại đặc biệt thích hợp cho mỗi ảnh)

3. Dựng GAT

\- B3 và B3b gộp hàng xóm bằng trung bình cộng cố định: ai cũng nghe hàng xóm như nhau, kể cả người đã có phim của chính mình. GAT thì học trọng số cho từng cạnh — hàng xóm nào đáng tin thì nghe nhiều, không đáng tin thì nghe ít.  
Chạy kèm 3 ablation để so sánh:

- Tắt đồ thị (--no-graph): cùng kiến trúc, cùng tham số, chỉ khác là node không nói chuyện với hàng xóm. Chênh lệch chính là đóng góp thuần của đồ thị.  
- Bỏ khối điểm hàng xóm (--no-nb-score): vẫn truyền tin, nhưng không cho sẵn điểm nguy cơ của hàng xóm.  
- Đổi sang bộ ảnh frozen: xem kết luận có phụ thuộc bộ đặc trưng ảnh không.

KẾT QUẢ:

**Bốn cấu hình GAT, ROC-AUC:**

| Cấu hình | Toàn dev | Có ảnh | Không ảnh |
| ----- | ----- | ----- | ----- |
| **GAT đầy đủ, nguồn control** | 0,724 | 0,721 \[0,692–0,750\] | **0,636 \[0,610–0,664\]** |
| GAT tắt đồ thị | 0,719 | 0,720 | 0,626 |
| GAT bỏ khối điểm hàng xóm | 0,721 | 0,723 | 0,635 |
| GAT nguồn frozen | 0,722 | 0,745 | 0,632 |

**So sánh ghép cặp trên cùng bệnh nhân:**

| So sánh | Nhóm | Chênh lệch | KTC 95% | P |
| ----- | ----- | ----- | ----- | ----- |
| **GAT − GAT tắt đồ thị** (đóng góp thuần của đồ thị) | không ảnh | **\+0,0100** | \[−0,0041; \+0,0238\] | 91% |
| GAT − GAT bỏ điểm hàng xóm | không ảnh | \+0,0011 | \[−0,0112; \+0,0125\] | 55% |
| GAT − B0 (chỉ lâm sàng) | không ảnh | \+0,0178 | \[−0,0027; \+0,0369\] | 96% |
| GAT − B3b | không ảnh | −0,0042 | \[−0,0238; \+0,0168\] | 32% |
| GAT − L2 (hồi quy logistic) | không ảnh | −0,0147 | \[−0,0339; \+0,0040\] | 6% |
| **GAT − B2** | có ảnh | **−0,0225** | **\[−0,0435; −0,0012\]** | 2% |
| **GAT − L1** | có ảnh | **−0,0389** | **\[−0,0612; −0,0157\]** | 0% |

Cách đọc: khoảng tin cậy **cắt qua 0** nghĩa là chưa phân biệt được, **nằm trọn một bên** mới là khác biệt thật.

**Xếp hạng cuối cùng ở nhóm không ảnh:** L2 0,651 → L3b 0,648 → B3b 0,640 → GAT 0,636 → GAT bỏ điểm hàng xóm 0,635 → B2 0,632 → B3 0,627 → GAT tắt đồ thị 0,626.

Đánh giá:

+ Đồ thị đóng góp \+0,0100 cho nhóm không ảnh (ko đáng kể), nhưng khoảng tin cậy cắt qua 0 (P \= 91%). Nói đúng mức: có xu hướng dương, chưa đủ để khẳng định với 430 ca dương.  
+ Khối điểm hàng xóm không cộng thêm gì vào GAT (+0,0011). Vì bản thân đồ thị đã mang thông tin đó rồi — hai cơ chế trùng nhau.  
+ Ở nhóm có ảnh, GAT thua rõ ràng. Thua B2 (P \= 2%) và thua L1 (P \= 0%).

Tóm lại:

+ **Thông tin ảnh lan được sang người chưa chụp phim**. B3b hơn B0 \+0,0220 ROC, KTC \[+0,0041; \+0,0393\], P \= 99%, và lặp lại được ở bộ ảnh thứ hai (+0,0153, P \= 96%).  
+ Mượn điểm kết luận của hàng xóm thì hiệu quả; mượn cả vector biểu diễn thì không, ở cả hai bộ ảnh.  
+ **GAT chưa vượt được baseline**. Đóng góp của truyền tin là \+0,0100 với P \= 91% — chưa đủ khẳng định. Ở nhóm có ảnh, GAT thua với P ≤ 2%.  
+ **Hồi quy logistic đang là mô hình tốt nhất ở cả hai nhóm, GAT thua toàn tập**: 0,760 (có ảnh) và 0,651 (không ảnh).  
+ **Đồ thị có cấu trúc yếu** — homophily lớp dương chỉ gấp 1,25 lần mức ngẫu nhiên.
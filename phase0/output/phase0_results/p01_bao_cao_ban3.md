# Báo cáo P0.1 — Kiểm tra nhiễu theo lô chụp trên ảnh đã xử lý (đề cương bản 3)

**Kính gửi thầy,**

Theo chỉ đạo của thầy khi gửi bản 3 (17/10): *"đã bỏ node khám theo vùng (E6), làm xử lý ảnh + QA và chạy lại P0.1 rồi gửi thầy xem, không sửa nhãn mà dùng phân tích độ nhạy"*, em xin báo cáo kết quả **P0.1**. P0.1 là cổng chặn duy nhất còn lại của Phase 0.

Báo cáo gồm: (1) vấn đề cần kiểm tra; (2) quy trình xử lý ảnh và kết quả QA; (3) bốn chỉ số của P0.1; (4) so sánh với đặc trưng cũ; (5) đối chiếu bảng quyết định; (6) nhận xét và các điểm xin thầy cho ý kiến.

Các thuật ngữ được giải thích ngay khi xuất hiện lần đầu. Bảng tra cứu đầy đủ ở [doc/chu_thich.md](../../../doc/chu_thich.md).

> **Tóm tắt.**
> - Cả 1.835 ảnh đã được che chữ và cắt theo phổi. QA trực quan đạt 96% (48/50 ảnh ngẫu nhiên), không ảnh nào hỏng.
> - Sau xử lý, **AUC trong lô** của mô hình chỉ dùng ảnh (L1) nằm trong khoảng **0,697–0,715**, so với 0,716–0,720 của ảnh cũ. Mức giảm không có ý nghĩa thống kê: mọi khoảng tin cậy của chênh lệch đều cắt qua 0.
> - Ảnh vẫn mang thêm thông tin so với mã lô ở mọi cấu hình: ΔAUC từ +0,068 đến +0,084, cận dưới khoảng tin cậy ≥ +0,045.
> - Tuy nhiên, khả năng đoán lô từ ảnh chỉ giảm từ 43–47% xuống 35%, vẫn gấp khoảng 3 lần mức đoán bừa (11%). Như vậy "dấu vân tay" của lô vẫn nằm trong chính vùng phổi.
> - Theo bảng quyết định: 5/6 cấu hình rơi vào ô "tiếp tục". Riêng V2 với nguồn control (0,697) rơi vào ô "báo cáo PGS", nhưng chỉ dưới ngưỡng 0,70 đúng 0,003. Khoảng tin cậy của **mọi** cấu hình đều chứa 0,70, nên kết quả là **sát ngưỡng**.


---

## 0. Thông tin chung

| Mục | Giá trị | Giải thích |
|---|---|---|
| Mẫu phân tích | 1.571 lượt khám có ảnh trong tập **dev** (393 dương, 1.493 bệnh nhân) | **dev** là phần dữ liệu dùng để thử nghiệm. Tập **test** (1.148 lượt) vẫn khóa, không được dùng ở đây. |
| Nhãn | `ketqua` (1 = dương tính) | Kết quả bác sĩ đọc phim theo phân loại ILO. Không sửa nhãn. |
| Chia dữ liệu | 5 **fold** của báo cáo fusion GAT (file fold hiện có, mục 3 đề cương) | Fold: dữ liệu chia 5 phần; mỗi lần học trên 4 phần, chấm trên phần còn lại. Mọi điểm số là **out-of-fold**, tức do mô hình chưa từng thấy người đó chấm. |
| Lô chụp | 18 lô; 16 lô có đủ cả ca dương lẫn ca âm | 2 lô bị loại khi tính "AUC trong lô" vì chỉ có ca dương: `20190827` (5 ảnh) và `unknown` (5 ảnh không xác định được ngày chụp). |
| Khoảng tin cậy | **KTC 95%**, **bootstrap** 2.000 lần **theo bệnh nhân** | Bootstrap: bốc ngẫu nhiên lại các bệnh nhân (có hoàn lại) 2.000 lần, mỗi lần tính lại chỉ số. KTC là khoảng chứa 95% các giá trị đó. Bốc theo người vì các lượt khám của cùng một người không độc lập. |
| Cấu hình khai báo trước | `config.json`, hash `2a4ad58122d08e57` | Mọi ngưỡng và tham số được ghi trước khi chạy. Hash là "dấu vân tay" của file cấu hình; nếu file bị sửa thì hash đổi. |
| Môi trường | Colab T4 (xử lý ảnh, trích đặc trưng), scikit-learn 1.6.1 | — |

**Kiểm tra tái hiện.** Em chạy lại mô hình L1 trên đặc trưng cũ và so với kết quả đã lưu ở báo cáo fusion GAT. AUC gần như trùng khớp: 0,76000 so với 0,75999 (control); 0,75889 so với 0,75892 (frozen). Xác suất từng ca lệch tối đa 0,0018. Mức lệch này do khác phiên bản thư viện: Colab dùng scikit-learn 1.6.1, máy local dùng 1.9.0 và trùng tuyệt đối. Như vậy code mới không làm sai số cũ.

**Hai nguồn đặc trưng ảnh.** Mọi chỉ số đều báo cho cả hai nguồn:
- **frozen**: dùng BioViL-T (mô hình học sẵn trên ảnh X-quang ngực) giữ nguyên trọng số;
- **control**: cùng kiến trúc nhưng đã được tinh chỉnh (**fine-tune**) trên tập ảnh ngoài SetA với nhãn `bnn`.

Cả hai lấy 256 chiều đầu của vector, giống hệt cách làm ở báo cáo fusion GAT.

---

## 1. Vấn đề cần kiểm tra

Mỗi **lô chụp** là một đợt chụp: cùng cơ sở, cùng ngày, cùng máy. Tỉ lệ bệnh giữa các lô rất khác nhau, ví dụ lô `20191115` có 7/143 ca dương (4,9%), còn lô `20181224` có 39/63 (62%). Riêng mã lô đã cho AUC gộp **0,705**.

Nếu pixel ảnh để lộ lô (qua chữ in trên phim, đặc tính máy chụp, cách xử lý ảnh), mô hình có thể đạt AUC cao chỉ nhờ "nhận ra lô", không cần đọc tổn thương. Đây là một **đường tắt (shortcut)**: kết quả đẹp nhưng không có giá trị lâm sàng.

**AUC** (diện tích dưới đường ROC) đo khả năng phân biệt. Hiểu đơn giản: lấy ngẫu nhiên một ca dương và một ca âm, AUC là xác suất mô hình chấm ca dương cao hơn. 0,5 là đoán bừa, 1,0 là hoàn hảo.

Ở bản 2, P0.1 đã chạy trên **ảnh gốc** (chưa che chữ, cắt theo kiểu CenterCrop) và cho AUC trong lô 0,716–0,719. Bản 3 yêu cầu làm lại sau khi **che chữ và cắt theo mask phổi**, để xem tín hiệu ảnh còn lại bao nhiêu khi đã loại các nguồn rò lô rõ ràng nhất.

---

## 2. Xử lý ảnh và QA

### 2.1. Quy trình (đúng mục 2 đề cương bản 3)

Đầu vào là ảnh gốc độ phân giải đầy đủ (khoảng 3.000 × 3.000 pixel), không dùng các bản 224/512 đã lưu trước đó.

| Bước | Việc làm | Giải thích |
|---|---|---|
| 1 | Đưa về ảnh xám; kiểm tra **đảo âm** | Đảo âm là ảnh lưu ngược màu (xương đen, phổi trắng). Phép kiểm tra: phim chuẩn có vùng bụng sáng hơn vùng phổi; nếu ngược lại thì coi là đảo âm và lật lại. |
| 2 | **Phân đoạn** phổi bằng torchxrayvision (mô hình PSPNet, kênh "Left Lung" và "Right Lung"); giữ **2 thành phần liên thông** lớn nhất | Phân đoạn: mô hình tô ra pixel nào là phổi, kết quả gọi là **mask**. Thành phần liên thông: một mảng pixel liền nhau. Giữ 2 mảng lớn nhất (2 lá phổi) để bỏ các đốm tô nhầm. Không dùng mask Set B của Silicodata. |
| 3 | Cắt theo **hộp bao** hai phổi, lề 5%, **pad** vuông, **resize** về 448 × 448 | Hộp bao: hình chữ nhật nhỏ nhất chứa hai phổi. Pad vuông: thêm viền đen cho thành hình vuông để ảnh không bị méo. 448 là kích thước đầu vào chuẩn của BioViL-T, giống đặc trưng cũ. **Không** dùng CenterCrop (cắt phần giữa ảnh), vì cách đó có thể làm mất đỉnh phổi. |
| 4 | **OCR** chế độ chỉ phát hiện, trên vùng đã cắt; tô đen các hộp chữ | OCR (nhận dạng ký tự quang học) ở chế độ chỉ tìm chỗ có chữ, không đọc nội dung, nên không trích xuất thông tin cá nhân. |
| 5 | Tạo hai biến thể: **V1** chỉ cắt (đã che chữ); **V2** = V1 và xóa mọi pixel ngoài mask phổi | V1 vẫn còn xương sườn, tim, cơ hoành. V2 chỉ còn hai lá phổi. So V1 với V2 cho biết tín hiệu nằm trong phổi hay ở vùng xung quanh. |

Ngoài ra còn tạo **V0** (toàn ảnh, đã che chữ), chỉ dùng làm vector dự phòng cho ảnh bị chấm "hỏng". Trong lần chạy này không có ảnh hỏng nên V0 không được dùng.

*Một sửa đổi trong quá trình chạy (để thầy nắm):* ở lần chạy thử 30 ảnh, em dùng cách kiểm tra đảo âm là "so tổng xác suất phổi giữa ảnh và ảnh lật". Cách này lật nhầm 3/30 phim bình thường. Em đã thay bằng phép đo độ sáng nêu ở bước 1: trên cả 1.835 ảnh, hiệu "bụng − phổi" đều dương (nhỏ nhất +37,7 trên thang 0–255), nên không ảnh nào bị lật. Lý do sửa được ghi trong `config.json`.

### 2.2. QA tự động

**QA** (Quality Assurance, kiểm tra chất lượng). Script tự gắn **cờ** (đánh dấu nghi có vấn đề) cho mọi ảnh theo 6 tiêu chí.

| Cờ | Ý nghĩa | Số ảnh (/1.835) |
|---|---|---|
| `dao_am` | Ảnh bị đảo âm (đã lật lại) | 0 |
| `khong_du_2_phoi` | Không tìm được đủ 2 lá phổi | 0 |
| `ty_le_trai_phai` | Diện tích phổi bên trái ảnh chia cho bên phải nằm ngoài [0,6; 1,6] | 0 |
| `cham_mep` | Mask chạm biên ảnh (phim có thể bị cắt thiếu) | 15 |
| `con_chu` | Sau khi tô đen vẫn phát hiện chữ | 5 |
| `dien_tich_bat_thuong` | Diện tích phổi ngoài khoảng trung vị ± 3·**MAD** | 97 |
| **Có ít nhất một cờ** | | **111 (6,0%)** |

**MAD** (median absolute deviation, độ lệch tuyệt đối trung vị) là một thước đo độ phân tán, ít bị ảnh hưởng bởi giá trị ngoại lai hơn độ lệch chuẩn.

Một số đặc điểm đo được trên 1.835 ảnh:
- phổi chiếm trung vị 29,6% diện tích ảnh;
- tỉ lệ trái/phải trung vị 1,01;
- 696 ảnh có hộp chữ bị phát hiện và tô đen bên trong vùng cắt. Chữ nằm ngoài vùng cắt thì đã bị loại khi cắt.

### 2.3. QA trực quan (chấm bằng mắt)

- **Mẫu:** 50 ảnh chọn ngẫu nhiên **phân tầng theo lô × nhãn**, cộng toàn bộ 111 ảnh bị cờ. Phân tầng nghĩa là mỗi tổ hợp (lô, dương/âm) đều có đại diện, nên mẫu rải đều chứ không dồn vào vài lô. File chấm **không** ghi nhãn, để người chấm không bị ảnh hưởng.
- **Mức chấm:** `đạt` (hai phổi trọn vẹn, mask ôm đúng, không còn chữ); `một phần` (dùng được nhưng có lỗi nhỏ, ví dụ hụt một góc sườn hoành); `hỏng` (mất phần phổi đáng kể hoặc mask sai hẳn).

| Nhóm | Số ảnh | Đạt | Một phần | Hỏng |
|---|---|---|---|---|
| 50 ảnh ngẫu nhiên | 50 | **48 (96%)** | 2 | 0 |
| Ảnh bị cờ | 111 | 95 | 16 | 0 |

- **Cổng QA:** cần ít nhất 90% ảnh ngẫu nhiên đạt; kết quả 96% nên **ĐẠT**. "Một phần" không được tính là đạt.
- **Ảnh hỏng:** 0 ảnh, nên không ảnh nào phải dùng vector toàn ảnh (`zone_missing` = 0). Không loại ảnh nào.

---

## 3. Bốn chỉ số của P0.1

Có 6 cấu hình, chạy trên cùng 1.571 lượt khám và cùng fold:
- **A_goc**: đặc trưng cũ, để đối chiếu;
- **V1** và **V2**;
- mỗi loại có 2 nguồn ảnh (control, frozen).

Mô hình kiểm tra là **L1**: hồi quy logistic chỉ dùng vector ảnh, giống báo cáo fusion GAT. **Hồi quy logistic (LR)** là mô hình đơn giản nhất cho bài toán có/không: nhân mỗi đặc trưng với một hệ số, cộng lại, rồi đổi thành xác suất.

### 3.1. Chỉ số 1 — Probe đoán lô

**Probe** là một mô hình nhỏ dùng để "dò" xem vector ảnh chứa thông tin gì. Ở đây em cho một hồi quy logistic đoán ảnh thuộc lô nào trong 18 lô. Đoán càng đúng thì ảnh càng mang nhiều dấu vết của lô.

| Cấu hình | Accuracy [KTC 95%] | Accuracy cân bằng | AUC OvR | Trong cùng tỉnh: accuracy (đa số) |
|---|---|---|---|---|
| A_goc · control | 0,43 [0,41; 0,45] | 0,35 | 0,89 | 0,50 (0,30) |
| A_goc · frozen | 0,47 [0,44; 0,49] | 0,40 | 0,90 | 0,53 (0,30) |
| V1 · control | 0,38 [0,35; 0,40] | 0,31 | 0,86 | 0,46 (0,30) |
| V1 · frozen | 0,41 [0,38; 0,43] | 0,33 | 0,86 | 0,48 (0,30) |
| V2 · control | 0,35 [0,33; 0,37] | 0,25 | 0,82 | 0,44 (0,30) |
| V2 · frozen | 0,35 [0,33; 0,37] | 0,27 | 0,84 | 0,42 (0,30) |

Cách đọc từng cột:
- **Accuracy**: tỉ lệ đoán đúng lô.
- Mốc so sánh 1, **lớp đa số**: luôn đoán lô đông nhất thì đúng **0,11**.
- Mốc so sánh 2, **nhãn hoán vị**: xáo trộn ngẫu nhiên mã lô rồi cho probe học lại thì chỉ đúng khoảng **0,08**. Đây là mức "không có thông tin".
- **Accuracy cân bằng**: trung bình tỉ lệ đúng của từng lô, không bị lô lớn lấn át.
- **AUC OvR** (one-vs-rest, một-chọi-phần-còn-lại): với mỗi lô tính AUC "lô này hay không phải lô này", rồi lấy trung bình. 0,5 là không phân biệt được lô.
- **Trong cùng tỉnh**: chỉ so các lô của cùng một tỉnh, để tách yếu tố máy chụp ra khỏi khác biệt dân cư giữa các tỉnh. Đây là phân tích khám phá.

Kết quả: che chữ và cắt làm giảm khả năng đoán lô (accuracy 0,43–0,47 → 0,35 ở V2), nhưng vẫn gấp khoảng 3 lần mức đa số. Nhận xét ở mục 6.

### 3.2. Chỉ số 2 — AUC trong lô của L1 (chỉ số quyết định)

**AUC trong lô** là AUC của L1 tính riêng trong từng lô, rồi lấy **trung bình có trọng số** theo cỡ lô (lô đông người được tính nặng hơn). Trong cùng một lô thì mọi ảnh có chung mã lô, nên "nhận ra lô" không còn giúp phân biệt dương với âm. Vì vậy chỉ số này đo phần tín hiệu ảnh không đến từ lô.

| Cấu hình | AUC gộp L1 | AUC trong lô [KTC 95%] (16 lô) | Nhạy cảm: chỉ lô có ≥ 10 dương và ≥ 10 âm (14 lô) |
|---|---|---|---|
| A_goc · control | 0,760 | 0,720 [0,687; 0,751] | 0,702 |
| A_goc · frozen | 0,759 | 0,716 [0,682; 0,748] | 0,697 |
| V1 · control | 0,750 | 0,712 [0,677; 0,744] | 0,702 |
| V1 · frozen | 0,749 | 0,711 [0,676; 0,742] | 0,691 |
| V2 · control | 0,741 | **0,697** [0,663; 0,731] | 0,686 |
| V2 · frozen | 0,742 | 0,715 [0,682; 0,747] | 0,696 |

- **AUC gộp** là AUC tính chung trên cả 1.571 lượt khám, không tách lô.
- **Bản nhạy cảm** bỏ thêm 2 lô có quá ít ca dương hoặc âm (ví dụ lô `20191113` chỉ có 3 ca dương), vì AUC tính trên vài ca rất dao động.
- AUC gộp luôn cao hơn AUC trong lô khoảng 0,03–0,04. Phần chênh này là phần nhờ phân biệt giữa các lô.

### 3.3. Chỉ số 3 — Giá trị gia tăng của ảnh: Δ (lô + ảnh so với lô)

Em so hai mô hình hồi quy logistic: LR(lô) chỉ biết mã lô, đạt AUC gộp **0,705**; LR(lô + ảnh) biết thêm vector ảnh. **Δ** (đọc là "delta") là hiệu AUC giữa hai mô hình này, tính **ghép cặp** trên cùng bệnh nhân.

| Cấu hình | AUC gộp LR(lô + ảnh) | Δ AUC [KTC 95%] |
|---|---|---|
| A_goc · control | 0,790 | +0,085 [+0,060; +0,110] |
| A_goc · frozen | 0,783 | +0,078 [+0,051; +0,105] |
| V1 · control | 0,790 | +0,084 [+0,060; +0,108] |
| V1 · frozen | 0,785 | +0,080 [+0,053; +0,106] |
| V2 · control | 0,773 | +0,068 [+0,045; +0,092] |
| V2 · frozen | 0,776 | +0,071 [+0,047; +0,097] |

Ở cả 6 cấu hình, **KTC nằm trọn trên 0**, tức ảnh thêm thông tin thật so với chỉ biết lô. "KTC nằm trọn trên 0" nghĩa là cả khoảng tin cậy đều dương, nên có thể kết luận có khác biệt. Nếu KTC "cắt qua 0" (chứa cả âm lẫn dương) thì chưa kết luận được.

### 3.4. Chỉ số 4 — LOBO

**LOBO** (Leave-One-Batch-Out, bỏ ra một lô): lần lượt bỏ hẳn một lô (và mọi bệnh nhân có ảnh trong lô đó) khỏi dữ liệu huấn luyện, rồi chấm trên chính lô đó. Mô hình chưa từng thấy lô đang được chấm, nên không thể dựa vào kiểu "lô này nhiều bệnh".

| Cấu hình | LOBO: AUC gộp [KTC 95%] | LOBO: AUC trong lô [KTC 95%] | Chênh AUC trong lô: LOBO − 5 fold [KTC 95%] |
|---|---|---|---|
| A_goc · control | 0,748 [0,719; 0,777] | 0,722 [0,690; 0,754] | +0,003 [−0,009; +0,014] |
| A_goc · frozen | 0,762 [0,734; 0,788] | 0,738 [0,706; 0,767] | +0,022 [+0,007; +0,038] |
| V1 · control | 0,729 [0,702; 0,756] | 0,707 [0,674; 0,739] | −0,005 [−0,017; +0,009] |
| V1 · frozen | 0,748 [0,721; 0,774] | 0,732 [0,699; 0,762] | +0,022 [+0,008; +0,036] |
| V2 · control | 0,734 [0,705; 0,763] | 0,699 [0,666; 0,732] | +0,002 [−0,011; +0,015] |
| V2 · frozen | 0,741 [0,714; 0,767] | 0,726 [0,695; 0,757] | +0,011 [−0,002; +0,024] |

Bỏ hẳn từng lô **không** làm AUC giảm. Với nguồn frozen, AUC còn tăng nhẹ. Như vậy L1 không phụ thuộc vào việc đã thấy lô đó khi học.

Phần tăng ở frozen nhiều khả năng do LOBO được học trên nhiều dữ liệu hơn (khoảng 17/18 số lô, so với 4/5 dữ liệu khi chia 5 fold). Em chưa kiểm chứng riêng điều này.

---

## 4. So với đặc trưng cũ (ghép cặp trên cùng bệnh nhân)

Mục đích là trả lời câu hỏi: che chữ và cắt theo phổi có làm mất tín hiệu ảnh không? Em so V1 và V2 với A_goc trên **cùng 1.571 lượt khám**, cùng các lần bootstrap. **P(Δ>0)** là tỉ lệ lần bootstrap có Δ dương. Đây **không phải p-value**, chỉ cho biết chiều của chênh lệch ổn định đến đâu.

| So sánh | Nguồn | Δ AUC trong lô [KTC 95%] | P(Δ>0) | Δ AUC gộp [KTC 95%] | P(Δ>0) |
|---|---|---|---|---|---|
| V1 − A_goc | control | −0,008 [−0,035; +0,018] | 29% | −0,010 [−0,030; +0,009] | 14% |
| V2 − A_goc | control | −0,023 [−0,054; +0,010] | 9% | −0,019 [−0,046; +0,008] | 8% |
| V1 − A_goc | frozen | −0,005 [−0,030; +0,018] | 32% | −0,010 [−0,029; +0,008] | 17% |
| V2 − A_goc | frozen | −0,001 [−0,031; +0,030] | 49% | −0,017 [−0,041; +0,008] | 9% |

Mọi KTC đều **cắt qua 0**, nên **chưa có bằng chứng** việc xử lý ảnh làm giảm tín hiệu trong lô. AUC gộp có xu hướng giảm 0,010–0,019, nhưng cũng không có ý nghĩa thống kê. Cần nói rõ là KTC khá rộng (khoảng ±0,03), nên phép so này cũng chỉ loại trừ được mức giảm lớn hơn khoảng 0,03–0,05.

---

## 5. Đối chiếu bảng quyết định của thầy

| AUC trong lô của L1 | Quyết định |
|---|---|
| ≥ 0,70 và KTC của Δ nằm trọn trên 0 | Tiếp tục; AUC trong lô là chỉ số chính cho mô hình dùng ảnh |
| 0,60–0,70 | Báo cáo PGS trước khi làm tiếp |
| < 0,60 | Dừng nhánh ảnh |

| Cấu hình | AUC trong lô | Cận dưới KTC của Δ | Ô tương ứng |
|---|---|---|---|
| A_goc · control | 0,720 | +0,060 | tiếp tục |
| A_goc · frozen | 0,716 | +0,051 | tiếp tục |
| V1 · control | 0,712 | +0,060 | tiếp tục |
| V1 · frozen | 0,711 | +0,053 | tiếp tục |
| V2 · control | 0,697 | +0,045 | **báo cáo PGS** |
| V2 · frozen | 0,715 | +0,047 | tiếp tục |


---

## 6. Nhận xét và các điểm xin thầy cho ý kiến

### 6.1. Nhận xét

1. **Kết quả sát ngưỡng, không phải rõ ràng.** KTC của AUC trong lô ở cả 6 cấu hình đều chứa 0,70, và bản nhạy cảm (14 lô) cho 0,686–0,702. Vì vậy em không thể khẳng định tín hiệu ảnh nằm chắc chắn trên hay dưới ngưỡng 0,70. Ô "báo cáo PGS" của V2 · control chỉ cách ngưỡng 0,003, nhỏ hơn rất nhiều so với độ rộng KTC (khoảng ±0,034).

2. **Che chữ và cắt không làm mất tín hiệu đáng kể.** Chênh lệch AUC trong lô so với ảnh cũ là −0,023 đến −0,001 và đều không có ý nghĩa thống kê. Ảnh vẫn thêm +0,068 đến +0,084 AUC so với chỉ biết lô, và LOBO không làm giảm AUC. Ba kết quả này cùng chỉ ra rằng phần tín hiệu ảnh trong lô **không phụ thuộc vào chữ in trên phim** hay vùng ngoài phổi.

3. **"Dấu vân tay" của lô vẫn còn trong vùng phổi.** Ngay cả V2 (chỉ còn pixel phổi), probe vẫn đoán đúng lô 35%, so với 11% (lớp đa số) và 8% (nhãn hoán vị). Trong cùng một tỉnh cũng vậy: 42–44% so với 30%. Chữ in không phải nguồn chính. Nguồn có thể là máy chụp, liều chiếu, thuật toán xử lý ảnh của máy, hoặc khác biệt dân cư giữa các đợt khám. Với dữ liệu hiện có em chưa tách được các nguồn này. Hệ quả là **AUC trong lô phải giữ làm chỉ số chính** cho mọi mô hình dùng ảnh, đúng như bảng quyết định. Không thể coi việc cắt theo phổi là đã loại hết ảnh hưởng của lô.

4. **V1 và V2 gần như tương đương.** Ở frozen, V2 (0,715) nhỉnh hơn V1 (0,711). Ở control thì ngược lại: V2 thấp hơn (0,697 so với 0,712), và phần chênh nằm trong KTC. Như vậy tín hiệu chủ yếu nằm trong vùng phổi, còn vùng xung quanh (xương, tim, cơ hoành) không đóng góp rõ rệt.

5. **QA đạt, nhưng mẫu 50 ảnh còn nhỏ.** Tỉ lệ đạt 96% (48/50) có KTC 95% (phương pháp Wilson) khoảng [86,5%; 98,9%]. Nghĩa là mẫu 50 ảnh chưa đủ để bảo đảm tỉ lệ đạt trên toàn bộ 1.835 ảnh là ≥ 90%. Ngoài ra, cờ `dien_tich_bat_thuong` gắn cho 97 ảnh nhưng khi chấm tay không có ảnh nào hỏng. Ngưỡng 3·MAD có vẻ hơi nhạy, nhưng điều này chỉ làm tăng số ảnh phải xem chứ không ảnh hưởng kết quả.

6. **Các số này còn có thể thay đổi sau P0.5.** Ghép nới lỏng (cùng họ tên, năm sinh, giới, tỉnh) cho thấy **1.366/8.030 lượt khám (17%)** thuộc các nhóm "nghi cùng một người" đang nằm ở các fold khác nhau, hoặc một bên dev một bên test. Con số này vượt ngưỡng 1% thầy đặt, nên phải chia lại fold. Khi chia lại, em sẽ chạy lại P0.1. Việc này chỉ chạy CPU, không cần trích lại đặc trưng ảnh.

### 6.2. Việc tiếp theo sau khi thầy duyệt

- P0.5: chia lại fold, chạy lại P0.1 trên fold mới, giữ kết quả fold cũ để đối chiếu.
- P0.4: vẽ boxplot điểm L1 của 57 ca (55 người) có `bnn=Có` nhưng `ketqua=2`, và làm phân tích độ nhạy khi bỏ 55 người này. Không sửa nhãn.
- Lập sổ kiểm toán (CSV) cho mọi quyết định xử lý dữ liệu.

---

## Phụ lục — File kết quả

| File | Nội dung |
|---|---|
| `p01_{A_goc,V1,V2}_{control,frozen}.json` / `.md` | Số liệu chi tiết từng cấu hình, gồm AUC của từng lô |
| `p01_oof_*.npz` | Điểm dự đoán out-of-fold, dùng cho phép so ghép cặp |
| `p01_mask_qa.csv` | QA tự động của từng ảnh (chỉ có img_id và số đo, không có tên) |
| `p01_qa_cham_tay.csv`, `p01_qa_tom_tat.json` | Kết quả chấm trực quan |
| `p01_zone_missing.csv` | Danh sách ảnh hỏng (rỗng) |
| `image_features_*_manifest.json` | Thông tin trích đặc trưng: phép biến đổi ảnh, mã băm của trọng số, số chiều |

Ảnh đã xử lý (phim X-quang) chỉ lưu trên Drive riêng tư, không đưa vào báo cáo.

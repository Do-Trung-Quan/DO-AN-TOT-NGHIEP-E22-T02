================================================================================
BÁO CÁO ĐÁNH GIÁ ĐẶC TRƯNG ẢNH TRÊN NHÃN: [BNN]
================================================================================
- Tổng số mẫu có ảnh   : 1835 lượt khám (1748 bệnh nhân)
- Số ca dương (bnn   = 1) : 99 ca
- Tỷ lệ dương tính (Prevalence): 5.40% (Mốc ngẫu nhiên PR-AUC = 0.0540)
- Phương pháp đánh giá  : 5-Fold CV (StratifiedGroupKFold theo patient_uid, 0% rò rỉ)

### BẢNG KẾT QUẢ CHÍNH (Out-of-fold kèm KTC 95% Bootstrap)

| Bộ đặc trưng | Mô tả nguồn gốc | ROC-AUC | PR-AUC | Accuracy | macro-F1 | Sensitivity | Specificity | PR-AUC vs Ngẫu nhiên |
|---|---|---|---|---|---|---|---|---|
| **control** | BioViL-T fine-tune tren SetA (bnn) | 0.880 [0.849–0.908] | 0.298 [0.225–0.390] | 0.922 [0.909–0.934] | 0.666 [0.625–0.705] | 0.434 [0.333–0.534] | 0.949 [0.939–0.960] | **5.5x** |
| **frozen** | BioViL-T goc Microsoft (pretrained) | 0.859 [0.826–0.889] | 0.232 [0.176–0.307] | 0.895 [0.880–0.909] | 0.614 [0.578–0.648] | 0.384 [0.292–0.480] | 0.925 [0.911–0.936] | **4.3x** |

### MỐC THAM CHIẾU (BASELINE)
- **Đoán ngẫu nhiên** : ROC-AUC = 0.500 | PR-AUC = 0.054 (tỷ lệ nền)
- **Chỉ dùng Tuổi**   : ROC-AUC = 0.823 | PR-AUC = 0.169 (Nếu vector ảnh vượt mốc này tức là ảnh thực sự mang tín hiệu bệnh học)


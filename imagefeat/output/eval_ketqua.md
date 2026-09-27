================================================================================
BÁO CÁO ĐÁNH GIÁ ĐẶC TRƯNG ẢNH TRÊN NHÃN: [KETQUA]
================================================================================
- Tổng số mẫu có ảnh   : 1835 lượt khám (1748 bệnh nhân)
- Số ca dương (ketqua= 1) : 462 ca
- Tỷ lệ dương tính (Prevalence): 25.18% (Mốc ngẫu nhiên PR-AUC = 0.2518)
- Phương pháp đánh giá  : 5-Fold CV (StratifiedGroupKFold theo patient_uid, 0% rò rỉ)

### BẢNG KẾT QUẢ CHÍNH (Out-of-fold kèm KTC 95% Bootstrap)

| Bộ đặc trưng | Mô tả nguồn gốc | ROC-AUC | PR-AUC | Accuracy | macro-F1 | Sensitivity | Specificity | PR-AUC vs Ngẫu nhiên |
|---|---|---|---|---|---|---|---|---|
| **control** | BioViL-T fine-tune tren SetA (bnn) | 0.761 [0.736–0.786] | 0.544 [0.496–0.591] | 0.744 [0.723–0.764] | 0.670 [0.645–0.695] | 0.537 [0.490–0.583] | 0.814 [0.792–0.834] | **2.2x** |
| **frozen** | BioViL-T goc Microsoft (pretrained) | 0.759 [0.733–0.784] | 0.533 [0.485–0.580] | 0.774 [0.754–0.794] | 0.679 [0.653–0.705] | 0.457 [0.409–0.503] | 0.881 [0.864–0.898] | **2.1x** |

### MỐC THAM CHIẾU (BASELINE)
- **Đoán ngẫu nhiên** : ROC-AUC = 0.500 | PR-AUC = 0.252 (tỷ lệ nền)
- **Chỉ dùng Tuổi**   : ROC-AUC = 0.636 | PR-AUC = 0.358 (Nếu vector ảnh vượt mốc này tức là ảnh thực sự mang tín hiệu bệnh học)


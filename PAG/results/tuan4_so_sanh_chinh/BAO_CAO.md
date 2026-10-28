# Tuần 4 — Kiểm định các giả thuyết khai báo trước (dev OOF)

Khai báo trước: `gia_thuyet_khai_bao_truoc_v2.json`. Biên không kém = −0,02 AUC. Holm (một phía, α = 0,025) trên 4 so sánh chính, dùng tỉ lệ lần bootstrap có Δ ≤ −0,02 làm đại lượng sắp thứ tự. H2 (PG vs PG-E6shuffle) và H4 (chú ý vs vùng tổn thương) KHÔNG đánh giá được.

### Bốn so sánh chính

| ma | so_sanh | nhom | n | auc_a | auc_b | delta | ci | P_delta_duong | ti_le_duoi_bien | dat_theo_KTC | p_Holm | dat_sau_Holm | delta_trong_lo | ci_trong_lo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1a | PG vs L1 | có ảnh | 1500 | 0.754 | 0.741 | 0.013 | [-0.008–0.035] | 0.878 | 0.000 | True | 0.000 | True | 0.005 | [-0.024–0.034] |
| H1b | PG vs L0 | không ảnh | 5183 | 0.679 | 0.708 | -0.028 | [-0.044–-0.012] | 0.000 | 0.853 | False | 0.853 | False | nan |  |
| H3a | PG vs T2 | có ảnh | 1500 | 0.754 | 0.761 | -0.007 | [-0.027–0.015] | 0.282 | 0.110 | False | 0.220 | False | nan |  |
| H3b | PG vs T2 | không ảnh | 5183 | 0.679 | 0.664 | 0.015 | [-0.002–0.032] | 0.957 | 0.000 | True | 0.000 | True | nan |  |

_Tệp: `gia_thuyet_chinh.csv`_

*Δ = AUC(PG) − AUC(đối chứng), ghép cặp trên cùng lượt khám. `ci` = KTC 95% bootstrap theo người. `P_delta_duong` = tỉ lệ lần bootstrap Δ>0 (mức chắc chắn PG tốt hơn; KHÔNG phải p-value). `ti_le_duoi_bien` = tỉ lệ lần bootstrap Δ ≤ −0,02 (càng nhỏ càng chắc PG không kém). `dat_theo_KTC`: cận dưới KTC > −0,02. `p_Holm`: tỉ lệ bootstrap dưới biên sau hiệu chỉnh Holm (dùng như p một phía cho giả thuyết “PG kém hơn quá 0,02”; đạt khi ≤ 0,025). `dat_sau_Holm`: vẫn đạt sau hiệu chỉnh đa so sánh Holm. `delta_trong_lo`: Δ tính bằng AUC trong lô chụp (H1a theo PDF yêu cầu báo cả chỉ số này).*

**Điểm dừng:** H1 KHÔNG đạt ở cả hai nhóm; H2 không đánh giá được → theo luật khai báo trước (H1 và H2 cùng không đạt) cần dừng và xem lại thiết kế trước khi đi tiếp.

### So sánh khám phá (không hiệu chỉnh đa so sánh, chỉ để tham khảo)

| so_sanh | nhom | n | auc_a | auc_b | delta | ci | P_delta_duong |
|---|---|---|---|---|---|---|---|
| PG vs PG-shuffle | có ảnh | 1500 | 0.754 | 0.751 | 0.003 | [-0.003–0.009] | 0.837 |
| PG vs PG-noedge | có ảnh | 1500 | 0.754 | 0.733 | 0.022 | [0.005–0.037] | 0.991 |
| PG vs L2 | có ảnh | 1500 | 0.754 | 0.711 | 0.043 | [0.019–0.069] | 1.000 |
| PG vs L2 | không ảnh | 5183 | 0.679 | 0.707 | -0.027 | [-0.043–-0.011] | 0.000 |
| PG vs GAT | có ảnh | 1500 | 0.754 | 0.704 | 0.050 | [0.026–0.078] | 1.000 |
| PG vs GAT | không ảnh | 5183 | 0.679 | 0.691 | -0.012 | [-0.029–0.006] | 0.074 |
| PG vs B3b | không ảnh | 5183 | 0.679 | 0.680 | -0.001 | [-0.021–0.018] | 0.439 |

_Tệp: `kham_pha.csv`_

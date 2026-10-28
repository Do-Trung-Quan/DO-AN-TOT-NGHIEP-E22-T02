# P0.4 — Ý nghĩa nhãn ketqua so với bnn (dev)

### Bảng chéo bnn × ketqua (lượt khám / người)

| nhom | luot | nguoi | co_anh | nguoi_co_anh |
|---|---|---|---|---|
| bnn=Có, ketqua=1 | 45 | 41 | 34 | 30 |
| bnn=Có, ketqua=2 | 128 | 97 | 45 | 42 |
| bnn=Không, ketqua=1 | 749 | 716 | 333 | 320 |
| bnn=Không, ketqua=2 | 5761 | 4635 | 1088 | 1007 |

_Tệp: `bang_cheo_bnn_ketqua.csv`_

### Điểm L1 (LR trên BioViL-T frozen, OOF) theo nhóm — chỉ ca có ảnh

| nhom | count | 25% | 50% | 75% |
|---|---|---|---|---|
| bnn=Có, ketqua=1 | 34.000 | 0.659 | 0.834 | 0.948 |
| bnn=Có, ketqua=2 | 45.000 | 0.214 | 0.454 | 0.578 |
| bnn=Không, ketqua=1 | 333.000 | 0.396 | 0.601 | 0.792 |
| bnn=Không, ketqua=2 | 1088.000 | 0.128 | 0.280 | 0.503 |

_Tệp: `diem_L1_theo_nhom.csv`_

### Trường ILO mật độ tổn thương ở nhóm bnn=Có, ketqua=2

| matdo | so_luot |
|---|---|
| (trống) | 128 |

_Tệp: `matdo_ILO_bnn_co_ketqua2.csv`_

### Các lượt khám khác của cùng người

| chi_so | gia_tri |
|---|---|
| Người bnn=Có, ketqua=2 | 97 |
| … trong đó có lượt khám KHÁC với ketqua=1 | 17 |
| … có > 1 lượt khám | 56 |
| Người bnn=Có, ketqua=2 có ảnh | 42 |

_Tệp: `luot_kham_khac.csv`_

**Diễn giải.** Nếu nhóm bnn=Có, ketqua=2 có điểm L1 gần nhóm ketqua=2 thì ảnh của họ trông bình thường. Khi đó ketqua có thể là kết luận phim *tại lần khám này*, không phải chẩn đoán bệnh nghề nghiệp, và hai nhãn đo hai thứ khác nhau. Cần PGS xác nhận.

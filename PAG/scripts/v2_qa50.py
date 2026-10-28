"""PAG_V2 QA: 50 anh ngau nhien SAU khi che chu burn-in (dung dung buoc 1 cua v2_image.py), de NGUOI xem bang mat.
Moi anh: ben trai = anh da che chu (inpaint), ben phai = cung anh, vien do = cho da che, vien xanh = phoi (phan doan lai).
Ten tep chi la so thu tu (qa_01.png ...). Bang tra so thu tu -> tep goc nam trong cache (PII).
Thu muc dau ra co the con sot chu neu che that bai -> KHONG dua len GitHub / khong chia se.
Script khong in gi ngoai so lieu gop.
"""
import os, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd, torch
from PIL import Image
import torchxrayvision as xrv
from v2_image import MERG, TEST_IMG, S0, S, DEV, xid, pad_square, burn_in_mask, largest_cc, outside_lung_blank
from v2_lib import CACHE, PAG

BG = "--bg" in sys.argv          # ban 2: to den moi thu ngoai bao loi 2 phoi; CUNG 50 anh, CUNG so thu tu de so voi ban 1
OUT = PAG / ("qa_50_anh_v2_xoa_ngoai_phoi_KHONG_CHIA_SE" if BG else "qa_50_anh_che_chu_KHONG_CHIA_SE"); OUT.mkdir(exist_ok=True)
files = []
for root in (MERG / "train", TEST_IMG):
    for sub in ("normal", "silicosis"):
        for f in sorted(os.listdir(root / sub)):
            if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png"): files.append(root / sub / f)
rng = np.random.default_rng(2026); sel = rng.choice(len(files), 50, replace=False)
seg = xrv.baseline_models.chestx_det.PSPNet().to(DEV).eval()
iL, iR = seg.targets.index("Left Lung"), seg.targets.index("Right Lung")

def segment(img, both=True):
    x = torch.from_numpy(xrv.datasets.normalize(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32), 255))[None, None].to(DEV)
    with torch.no_grad(): o = torch.sigmoid(seg(x))[0]
    up = lambda m: cv2.resize(m.astype(np.uint8), (S0, S0), interpolation=cv2.INTER_NEAREST).astype(bool)
    R, L = up(largest_cc((o[iR] > 0.5).cpu().numpy())), up(largest_cc((o[iL] > 0.5).cpu().numpy()))
    return (R | L) if both else (R, L)

rows, nchu = [], []
for j, i in enumerate(sel, 1):
    im = cv2.resize(pad_square(np.asarray(Image.open(files[i]).convert("L"))), (S0, S0), interpolation=cv2.INTER_AREA)
    tm, k = burn_in_mask(im, segment(im))
    imm = cv2.inpaint(im, tm.astype(np.uint8), 5, cv2.INPAINT_TELEA) if k else im
    R, L = segment(imm, both=False); lung = R | L
    if BG: imm, _ = outside_lung_blank(imm, R, L)
    over = cv2.cvtColor(imm, cv2.COLOR_GRAY2BGR)
    for m, col in ((tm, (0, 0, 255)), (lung, (0, 200, 0))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(over, cs, -1, col, 2)
    ok, buf = cv2.imencode(".png", np.concatenate([cv2.cvtColor(imm, cv2.COLOR_GRAY2BGR), over], 1))   # imwrite hong voi duong dan co dau
    assert ok; (OUT / f"qa_{j:02d}.png").write_bytes(buf.tobytes())
    rows.append(dict(stt=j, tep=str(files[i]))); nchu.append(k)
pd.DataFrame(rows).to_csv(CACHE / f"qa50_bang_tra{'_bg' if BG else ''}.csv", index=False, encoding="utf-8-sig")
(OUT / "HUONG_DAN.txt").write_text(
    "Kiem tra bang mat 50 anh da che chu (PAG_V2 QA).\n"
    "Moi tep: trai = anh sau khi che; phai = vien DO la cho da che, vien XANH la phoi.\n"
    "Ghi lai so thu tu (qa_XX) neu: (1) con doc duoc ten / ngay / chu nao; (2) vung che an vao phoi; (3) vien xanh lech phoi ro.\n"
    "Thu muc nay co the con thong tin ca nhan neu che that bai: KHONG dua len GitHub, KHONG chia se.\n", encoding="utf-8")
print(f"QA50 XONG: 50 anh | anh co che chu {sum(x > 0 for x in nchu)}/50 | so vung che trung vi {np.median(nchu):.0f}")

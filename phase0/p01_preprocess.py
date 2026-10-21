"""P0.1 — xu ly anh + QA tu dong (PG_update ban 3, muc 2 "Xu ly anh").

CHAY TREN COLAB GPU (can: torchxrayvision, opencv-python-headless, easyocr). Chay --limit 30 truoc.

Moi anh goc (data/train, data/test; KHONG dung ban 224/512 da luu):
  1. Dua ve anh xam; kiem tra dao am: chay mask phoi tren anh va anh lat (255 - x), chon chieu co
     tong xac suat phoi lon hon. Neu phai lat -> lat anh va gan co dao_am.
  2. Phan doan phoi bang torchxrayvision PSPNet (chestx_det, kenh Left/Right Lung); giu 2 thanh phan
     lien thong lon nhat. KHONG dung mask Set B cua Silicodata.
  3. Cat theo hop bao hai phoi (le 5%), dem vuong roi resize 448. Khong CenterCrop.
  4. OCR che do chi phat hien (easyocr Reader.detect, khong nhan dang) tren vung da cat; to den hop chu.
  5. Hai bien the: V1 = chi cat (da che chu); V2 = V1 va xoa pixel ngoai mask phoi.
     Them V0 = toan anh (da che chu) — chi dung lam vector du phong cho anh bi cham "hong" o QA.

QA tu dong (moi anh, cot co_* trong p01_mask_qa.csv): dien tich phoi bat thuong, khong du 2 phoi,
ty le trai/phai ngoai [0,6; 1,6], mask cham mep, dao am, con chu sau khi che. Khong loai anh nao.

Dau ra (--out-dir): shard_XXXX.npz, p01_processed.npz (img_id, V0, V1, V2, mask; uint8 448x448),
p01_mask_qa.csv (chi img_id va so lieu, KHONG ten file). Anh da xu ly van la phim X-quang that:
chi giu o Drive rieng tu, khong dua vao git, bao cao hay hinh minh hoa.

  python -m phase0.p01_preprocess --images-root /content/images --out-dir /content/p01_sample --limit 30
  python -m phase0.p01_preprocess --images-root /content/images --out-dir /content/p01_processed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "imagefeat"))

from phase0.common import load_config  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEG_SIZE = 512                # dau vao PSPNet chestx_det
OCR_LONG_SIDE = 1024          # canh dai cua ban thu nho dua vao OCR detect
SHARD = 100


# ------------------------------------------------------------------ hinh hoc
def pad_square(a: np.ndarray):
    """Dem vien den thanh hinh vuong. Tra ve (anh vuong, oy, ox)."""
    h, w = a.shape
    s = max(h, w)
    out = np.zeros((s, s), a.dtype)
    oy, ox = (s - h) // 2, (s - w) // 2
    out[oy:oy + h, ox:ox + w] = a
    return out, oy, ox


def resize(a: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    import cv2
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(a, (size, size), interpolation=interp)


# ------------------------------------------------------------------ buoc 1-2: mask phoi
def lung_prob(model, sq512: np.ndarray, device):
    """sq512: anh xam vuong 512x512 uint8. Tra ve xac suat (phoi trai, phoi phai) 512x512."""
    import torch
    import torchxrayvision as xrv
    x = xrv.datasets.normalize(sq512, 255)[None, None, ...].astype(np.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x).to(device)))[0].cpu().numpy()
    return prob[model.targets.index("Left Lung")], prob[model.targets.index("Right Lung")]


def keep_largest(mask: np.ndarray, k: int):
    """Giu k thanh phan lien thong lon nhat. Tra ve (mask, dien tich cac thanh phan da giu (px), so thanh phan ban dau)."""
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool), [], [], 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1][:k]
    keep = np.isin(lab, order + 1)
    return keep, [int(areas[i]) for i in order], [float(cent[i + 1][0]) for i in order], int(n - 1)


# ------------------------------------------------------------------ buoc 4: OCR chi phat hien
def text_boxes(img: np.ndarray, reader, margin: int):
    """Phat hien hop chu (KHONG doc chu). img: anh xam bat ky co. Tra ve list (x0,y0,x1,y1) tren toa do img."""
    if reader is None:
        return []
    h, w = img.shape
    f = min(1.0, OCR_LONG_SIDE / max(h, w))
    small = resize_keep(img, f)
    horiz, free = reader.detect(small)
    boxes = []
    for x0, x1, y0, y1 in horiz[0]:
        boxes.append((x0, y0, x1, y1))
    for poly in free[0]:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    out = []
    for x0, y0, x1, y1 in boxes:
        out.append((max(0, int((x0 - margin) / f)), max(0, int((y0 - margin) / f)),
                    min(w, int(np.ceil((x1 + margin) / f))), min(h, int(np.ceil((y1 + margin) / f)))))
    return out


def resize_keep(img: np.ndarray, f: float) -> np.ndarray:
    import cv2
    if f >= 1.0:
        return img
    h, w = img.shape
    return cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)


def blackout(img: np.ndarray, boxes) -> tuple[np.ndarray, float]:
    out = img.copy()
    m = np.zeros(img.shape, bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = True
    out[m] = 0
    return out, float(m.mean())


# ------------------------------------------------------------------ mot anh
def process_one(full: np.ndarray, model, reader, device, P: dict):
    """Tra ve (V0, V1, V2, mask448, qa)."""
    size, margin_ocr = P["kich_thuoc_dau_vao"], P["ocr_le_hop_px"]
    H, W = full.shape
    qa = {"H": H, "W": W}

    # --- buoc 1: dao am (chon chieu cho tong xac suat phoi lon hon)
    sq, oy, ox = pad_square(full)
    S = sq.shape[0]
    s512 = resize(sq, SEG_SIZE)
    pL, pR = lung_prob(model, s512, device)
    qL, qR = lung_prob(model, 255 - s512, device)
    inv = (qL + qR).sum() > (pL + pR).sum()
    if inv:
        full, sq, pL, pR = 255 - full, 255 - sq, qL, qR
        sq[:oy, :] = 0; sq[oy + H:, :] = 0; sq[:, :ox] = 0; sq[:, ox + W:] = 0   # vien dem van den
    qa["co_dao_am"] = int(inv)

    # --- buoc 2: mask phoi, giu 2 thanh phan lon nhat
    f = SEG_SIZE / S
    cy0, cy1 = int(oy * f), int(np.ceil((oy + H) * f))          # vung noi dung (khong tinh vien dem)
    cx0, cx1 = int(ox * f), int(np.ceil((ox + W) * f))
    content_area = (cy1 - cy0) * (cx1 - cx0)
    raw = (pL > 0.5) | (pR > 0.5)
    mask, areas, cxs, n_comp = keep_largest(raw, P["giu_thanh_phan"])
    qa["n_thanh_phan_truoc_loc"] = n_comp
    qa["dien_tich_phoi"] = float(mask.sum() / content_area)
    a_min = P["phoi_toi_thieu"] * SEG_SIZE ** 2
    du_hai = len(areas) == 2 and areas[1] >= a_min
    qa["co_khong_du_2_phoi"] = int(not du_hai)
    if du_hai:
        # thanh phan ben trai anh / ben phai anh (ben trai anh = phoi phai benh nhan)
        a_left_img, a_right_img = (areas[0], areas[1]) if cxs[0] < cxs[1] else (areas[1], areas[0])
        qa["ty_le_trai_phai"] = float(a_left_img / a_right_img)
    else:
        qa["ty_le_trai_phai"] = np.nan
    lo, hi = P["ty_le_trai_phai"]
    qa["co_ty_le_trai_phai"] = int(du_hai and not (lo <= qa["ty_le_trai_phai"] <= hi))
    e = P["cham_mep_px"]
    if mask.any():
        ys, xs = np.where(mask)
        touch = ys.min() <= cy0 + e or ys.max() >= cy1 - 1 - e or xs.min() <= cx0 + e or xs.max() >= cx1 - 1 - e
    else:
        touch = False
    qa["co_cham_mep"] = int(touch)

    # --- V0: toan anh, che chu (vector du phong)
    b0 = text_boxes(sq, reader, margin_ocr)
    sq0, _ = blackout(sq, b0)
    V0 = resize(sq0, size)

    # --- buoc 3: cat theo hop bao hai phoi + le, dem vuong, resize
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        my, mx = int((y1 - y0) * P["le_cat"]), int((x1 - x0) * P["le_cat"])
        y0, y1 = max(cy0, y0 - my), min(cy1, y1 + my)
        x0, x1 = max(cx0, x0 - mx), min(cx1, x1 + mx)
        g = S / SEG_SIZE
        Y0, Y1, X0, X1 = int(y0 * g), int(np.ceil(y1 * g)), int(x0 * g), int(np.ceil(x1 * g))
        crop = sq[Y0:Y1, X0:X1]
        mfull = resize(mask.astype(np.uint8), S, nearest=True)[Y0:Y1, X0:X1]
        qa["hop_bao_ty_le"] = float((y1 - y0) * (x1 - x0) / content_area)
    else:                                   # khong tim thay phoi: dung toan anh, da gan co khong_du_2_phoi
        crop, mfull = sq, np.zeros_like(sq, np.uint8)
        qa["hop_bao_ty_le"] = np.nan

    # --- buoc 4: OCR chi phat hien tren vung da cat, to den
    boxes = text_boxes(crop, reader, margin_ocr)
    crop_m, frac = blackout(crop, boxes)
    qa["n_hop_chu"] = len(boxes)
    qa["dien_tich_che"] = frac
    qa["co_con_chu"] = int(len(text_boxes(crop_m, reader, 0)) > 0)

    # --- buoc 5: V1, V2
    c_sq, _, _ = pad_square(crop_m)
    m_sq, _, _ = pad_square(mfull)
    V1 = resize(c_sq, size)
    M = resize(m_sq, size, nearest=True).astype(bool)
    V2 = np.where(M, V1, 0).astype(np.uint8)
    return V0, V1, V2, M, qa


# ------------------------------------------------------------------ chinh
def build_frame(images_root: Path) -> pd.DataFrame:
    from build_index import extract_id, scan_images
    fr = scan_images(images_root)
    fr["img_id"] = fr["fname"].map(extract_id)
    if fr["img_id"].isna().any():
        raise SystemExit(f"{int(fr['img_id'].isna().sum())} anh khong rut duoc id tu ten file")
    fr["img_id"] = fr["img_id"].astype(int)
    if fr["img_id"].duplicated().any():
        raise SystemExit("img_id trung nhau")
    return fr.sort_values("img_id").reset_index(drop=True)


def area_flag(qa: pd.DataFrame, k: float = 3.0) -> pd.Series:
    """Dien tich phoi bat thuong: ngoai trung vi +- k*MAD (tinh tren tap anh dang xu ly)."""
    a = qa["dien_tich_phoi"]
    med = a.median()
    mad = (a - med).abs().median()
    return ((a < med - k * mad) | (a > med + k * mad)).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="chi xu ly N anh ngau nhien (xem thu)")
    ap.add_argument("--no-ocr", action="store_true", help="bo OCR (CHI de thu toc do; khong dung cho bao cao)")
    args = ap.parse_args()

    cfg = load_config()
    P = {**cfg["ban3"]["xu_ly_anh"], **cfg["ban3"]["qa_tu_dong"]}

    import torch
    import torchxrayvision as xrv
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"P0.1 xu ly anh (ban 3) | config {cfg['_hash']} | thiet bi {device}")
    frame = build_frame(args.images_root)
    if args.limit:
        rng = np.random.default_rng(42)
        frame = frame.iloc[np.sort(rng.choice(len(frame), args.limit, replace=False))].reset_index(drop=True)
    print(f"{len(frame)} anh")

    reader = None
    if not args.no_ocr:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_lim" if args.limit else ""
    t0 = time.time()
    for s0 in range(0, len(frame), SHARD):
        shard = args.out_dir / f"shard_{s0 // SHARD:04d}{suffix}.npz"
        if shard.exists():
            z = np.load(shard, allow_pickle=False)
            if str(z["config_hash"]) != cfg["_hash"]:
                raise SystemExit(f"{shard.name} tao bang config khac ({z['config_hash']}): xoa thu muc out-dir roi chay lai")
            print(f"  [bo qua] {shard.name} da co")
            continue
        sub = frame.iloc[s0:s0 + SHARD]
        V0s, V1s, V2s, Ms, qas = [], [], [], [], []
        for _, r in sub.iterrows():
            full = np.asarray(Image.open(r["image_path"]).convert("L"), dtype=np.uint8)
            V0, V1, V2, M, qa = process_one(full, model, reader, device, P)
            qa["img_id"] = int(r["img_id"])
            V0s.append(V0); V1s.append(V1); V2s.append(V2); Ms.append(M); qas.append(qa)
        np.savez_compressed(shard, img_id=np.array([q["img_id"] for q in qas]), V0=np.stack(V0s),
                            V1=np.stack(V1s), V2=np.stack(V2s), mask=np.stack(Ms),
                            qa=json.dumps(qas), config_hash=cfg["_hash"])
        print(f"  {shard.name} xong | {min(s0 + SHARD, len(frame))}/{len(frame)} anh | {time.time() - t0:.0f}s")

    # gop cac shard
    pat = f"shard_*{suffix}.npz" if args.limit else "shard_[0-9][0-9][0-9][0-9].npz"
    parts = {k: [] for k in ("img_id", "V0", "V1", "V2", "mask")}
    Q = []
    for p in sorted(args.out_dir.glob(pat)):
        z = np.load(p, allow_pickle=False)
        for k in parts:
            parts[k].append(z[k])
        Q += json.loads(str(z["qa"]))
    arr = {k: np.concatenate(v) for k, v in parts.items()}
    o = np.argsort(arr["img_id"])
    arr = {k: v[o] for k, v in arr.items()}
    np.savez_compressed(args.out_dir / f"p01_processed{suffix}.npz", **arr, config_hash=cfg["_hash"])

    qa = pd.DataFrame(Q).set_index("img_id").loc[arr["img_id"]].reset_index()
    qa["co_dien_tich_bat_thuong"] = area_flag(qa, P["dien_tich_bat_thuong_k_mad"])
    flags = [c for c in qa.columns if c.startswith("co_")]
    qa["co_bat_ky"] = (qa[flags].sum(1) > 0).astype(int)
    qa = qa[["img_id"] + [c for c in qa.columns if c != "img_id"]]
    qa.to_csv(args.out_dir / f"p01_mask_qa{suffix}.csv", index=False)

    print("\n=== QA tu dong ===")
    print(f"  so anh: {len(qa)}" + ("  (mau --limit: nguong dien tich tinh tren mau nho, chi de xem thu)" if args.limit else ""))
    for c in flags + ["co_bat_ky"]:
        print(f"  {c:28s}: {int(qa[c].sum()):5d}  ({qa[c].mean():.1%})")
    print(f"  dien tich phoi / anh: trung vi {qa.dien_tich_phoi.median():.3f}")
    print(f"  ty le trai/phai: trung vi {qa.ty_le_trai_phai.median():.2f}")
    print(f"  anh co hop chu: {int((qa.n_hop_chu > 0).sum())} | dien tich che trung vi {qa.dien_tich_che.median():.4%}")
    print(f"ghi {args.out_dir / f'p01_processed{suffix}.npz'} va p01_mask_qa{suffix}.csv")


if __name__ == "__main__":
    main()

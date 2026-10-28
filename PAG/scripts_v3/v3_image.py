"""Xu ly anh + QA tu dong + dac trung theo PG_update (ban 3, 17/10) §2.

Tu ANH GOC (khong dung ban 224/512 da luu):
 1. Anh xam, pad vuong. Kiem tra dao am: phan doan phoi tren anh va tren anh dao (255 - x); neu anh dao cho phoi
    "chac" hon ro ret (diem = tong xac suat > 0.5, gap >= 1.5 lan) -> coi la anh dao am, dao lai, gan co.
 2. Phan doan phoi bang torchxrayvision PSPNet (khong dung mask Set B); giu thanh phan lon nhat moi ben (= 2 thanh phan).
 3. Cat theo hop bao hai phoi + le 5% moi phia, pad vuong, resize 512. Khong CenterCrop.
 4. OCR CHI PHAT HIEN (RapidOCR text_detector, khong nhan dang). Ban sua 18/10: chay tren ANH GOC (1024) thay vi khung cat
    phong to (o do OCR bat nham xuong suon/bo phoi thanh chu: dien tich to rai deu khung cat, khong don ra mep).
    Bo hop cham phoi (> 5% dien tich trong phoi noi ~1,5%), hop nghieng > 15 do, hop cao > 6% anh (chu tren phim nam ngang,
    o ngoai phoi). To (inpaint Telea) cac hop con lai tren anh goc, CHI phan nam ngoai hop bao 2 phoi (dai le 5% cua khung cat;
    trong truong phoi khong co chu in -> phat hien o do coi la nham, khong to), roi moi cat.
 5. V1 = anh cat (da to chu). V2 = V1 va xoa pixel ngoai mask phoi (noi 1% canh).
QA tu dong moi anh: dien tich phoi bat thuong (|z robust| > 3), khong du 2 phoi (< 1% anh), ti le trai/phai ngoai
[0.6; 1.6], mask cham mep anh goc, dao am, con chu (OCR chay lai tren V1 sau khi to van thay hop chu ngoai phoi).
Dac trung (V1 va V2): BioViL-T dong bang (CHINH, patch 512 x 16x16, img_embedding 512) va CheXFound (b, patch 1024 x 32x32, CLS).
 - vung: trung binh co trong so ban do dac trung trong mask 6 vung (1/3, 2/3 chieu cao moi phoi)
 - gpool: trung binh moi patch (vector toan anh cung chieu voi vung, dung thay vung khi zone_missing)
Dau ra: PAG/cache_v3/img_v3.npz (+ qa_thumbs.npz anh thu nho cho QA truc quan). Chua id + anh -> PII, KHONG dua len GitHub.
python v3_image.py [--n 20]   (--n: chay thu n anh dau, khong ghi cache chinh)
"""
import os, re, sys, time, json
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from scipy import ndimage
from v3_lib import MERG, BASE, CACHE3, REG

TEST_IMG = MERG.parent / "NEW DATA" / "test"
S, SD, T = 512, 1024, 256          # dau vao encoder, dau vao OCR, anh thu nho QA
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NTRY = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 0
MARGIN, GROW_V2 = 0.05, 0.01
BOX_IN_LUNG, BOX_ANGLE, BOX_H, LUNG_PAD = 0.05, 15, 0.06, 15   # loc hop chu: >5% trong phoi, nghieng >15 do, cao >6% anh; noi phoi 15/1024 px

RE_ID = [re.compile(r"^(\d+)_(?=\D)"), re.compile(r"^\d{8}_\d{6}_(\d+)_"), re.compile(r"^[A-Za-z][A-Za-z ]*_(\d+)(?:_\d{8})?$")]
def xid(f):
    s = Path(f).stem
    for p in RE_ID:
        m = p.match(s)
        if m: return int(m.group(1))


def pad_square(im):
    h, w = im.shape[:2]; n = max(h, w); y, x = (n - h) // 2, (n - w) // 2
    out = np.zeros((n, n) + im.shape[2:], im.dtype); out[y:y + h, x:x + w] = im
    return out, (y, x, h, w)


def largest_cc(m):
    lab, n = ndimage.label(m)
    if n == 0: return m
    sz = ndimage.sum(m, lab, range(1, n + 1)); return lab == (1 + int(np.argmax(sz)))


def thirds(m):
    rows = np.where(m.any(1))[0]; out = np.zeros((3,) + m.shape, bool)
    if len(rows) == 0: return out
    top, bot = rows.min(), rows.max() + 1; c1, c2 = top + (bot - top) / 3, top + 2 * (bot - top) / 3
    r = np.arange(m.shape[0])[:, None]
    out[0] = m & (r < c1); out[1] = m & (r >= c1) & (r < c2); out[2] = m & (r >= c2); return out


def biovil():
    from health_multimodal.image.model.model import ImageModel
    from health_multimodal.image.model.types import ImageEncoderType
    m = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE, joint_feature_size=128,
                   pretrained_model_path=str(BASE / "Data DGCNN" / "Pre-train BioViL-T" / "biovil_t_image_model_proj_size_128.pt"))
    return m.to(DEV).eval()


def main():
    files = []
    for root in (MERG / "train", TEST_IMG):
        for sub in ("normal", "silicosis"):
            for f in sorted(os.listdir(root / sub)):
                if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png"): files.append((xid(f), root / sub / f))
    ids = np.array([i for i, _ in files]); assert len(set(ids)) == len(ids) == 1835
    if NTRY: files, ids = files[:NTRY], ids[:NTRY]
    from rapidocr_onnxruntime import RapidOCR
    det = RapidOCR().text_detector
    seg = xrv.baseline_models.chestx_det.PSPNet().to(DEV).eval()
    iL, iR = seg.targets.index("Left Lung"), seg.targets.index("Right Lung")
    import chexfound
    enc = {"biovil": biovil(), "chexfound": chexfound.load(DEV)}
    DIM, GDIM = {"biovil": 512, "chexfound": 1024}, {"biovil": 512, "chexfound": 1024}
    N, V = len(files), ("V1", "V2")
    regf = {(e, v): np.zeros((N, 6, DIM[e]), np.float16) for e in enc for v in V}
    glob = {(e, v): np.zeros((N, GDIM[e]), np.float16) for e in enc for v in V}
    gpool = {(e, v): np.zeros((N, DIM[e]), np.float16) for e in enc for v in V}
    area = np.zeros((N, 6), np.float32); masks = np.zeros((N, 128, 128), np.uint8)
    qk = ["inv_ratio", "dao_am", "dao_am_heuristic", "phoi_R", "phoi_L", "ti_le_LR", "cham_mep", "pct_crop",
          "n_hop_chu", "pct_to", "hop_chu_trong_phoi", "n_chu_con_lai"]
    qa = {k: np.zeros(N, np.float32) for k in qk}
    th = {k: np.zeros((N, T, T), np.uint8) for k in ("goc", "V1")}; th["mask_goc"] = np.zeros((N, T, T), np.uint8)
    th["to"] = np.zeros((N, T, T), bool)

    def segment(img512):
        x = torch.from_numpy(xrv.datasets.normalize(img512.astype(np.float32), 255))[None, None].to(DEV)
        o = torch.sigmoid(seg(x))[0]
        return o[iR].cpu().numpy(), o[iL].cpu().numpy()

    def text_boxes(img, lung_d):
        """Hop chu tren anh goc vuong (OCR chi phat hien, o SD x SD). lung_d: mask phoi noi, SD x SD.
        Tra ve mask hop giu lai (SD x SD), danh sach hop bao cua hop giu (y0, y1, x0, x1; toa do 0-1), so hop bo vi cham phoi."""
        big = cv2.resize(img, (SD, SD), interpolation=cv2.INTER_AREA)
        boxes, _ = det(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
        m = np.zeros((SD, SD), np.uint8); cen, n_lung = [], 0
        for b in (boxes if boxes is not None else []):
            b = np.asarray(b, np.float32); (cx, cy), (w, h), ang = cv2.minAreaRect(b)
            one = np.zeros((SD, SD), np.uint8); cv2.fillPoly(one, [np.round(b).astype(np.int32)], 1); one = one.astype(bool)
            if lung_d[one].mean() > BOX_IN_LUNG: n_lung += 1; continue
            a = ang % 90
            if min(a, 90 - a) > BOX_ANGLE or min(w, h) > BOX_H * SD: continue
            ys, xs = b[:, 1] / SD, b[:, 0] / SD; m |= one; cen.append((ys.min(), ys.max(), xs.min(), xs.max()))
        return m.astype(bool), cen, n_lung

    t0 = time.time()
    with torch.no_grad():
        for i, (_, p) in enumerate(files):
            raw = np.asarray(Image.open(p).convert("L"))
            P, (oy, ox, oh, ow) = pad_square(raw); n = P.shape[0]
            p512 = cv2.resize(P, (S, S), interpolation=cv2.INTER_AREA)
            # 1) dao am
            pr, pl = segment(p512); qr, ql = segment(255 - p512)
            sc = lambda a, b: float(a[a > 0.5].sum() + b[b > 0.5].sum())
            s0, s1 = sc(pr, pl), sc(qr, ql); qa["inv_ratio"][i] = s1 / max(s0, 1.0)
            if s1 >= 1.5 * max(s0, 1.0):
                P = 255 - P; p512 = 255 - p512; pr, pl = qr, ql; qa["dao_am"][i] = 1
            # 2) phoi: thanh phan lon nhat moi ben
            R, L = largest_cc(pr > 0.5), largest_cc(pl > 0.5)
            k = S / n; ry0, rx0, ry1, rx1 = int(oy * k), int(ox * k), int(np.ceil((oy + oh) * k)), int(np.ceil((ox + ow) * k))
            img_area = max((ry1 - ry0) * (rx1 - rx0), 1)
            qa["phoi_R"][i], qa["phoi_L"][i] = R.sum() / img_area, L.sum() / img_area
            qa["ti_le_LR"][i] = L.sum() / max(R.sum(), 1)
            both = R | L
            inner = np.zeros_like(both); inner[ry0 + 2:ry1 - 2, rx0 + 2:rx1 - 2] = True
            qa["cham_mep"][i] = bool((both & ~inner).any())
            lung_mean = p512[both].mean() if both.any() else np.nan
            qa["dao_am_heuristic"][i] = bool(both.sum() > 0.02 * S * S and lung_mean > p512[ry0:ry1, rx0:rx1].mean())
            # 4) OCR chi phat hien tren anh goc, to hop chu ngoai phoi (truoc khi cat)
            lung_od = ndimage.binary_dilation(cv2.resize(both.astype(np.uint8), (SD, SD), interpolation=cv2.INTER_NEAREST).astype(bool), iterations=LUNG_PAD)
            tm, cen, qa["hop_chu_trong_phoi"][i] = text_boxes(P, lung_od)
            paint_o = cv2.resize((ndimage.binary_dilation(tm, iterations=2) & ~lung_od).astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST)
            if both.sum() >= 0.02 * S * S:                              # chi to ngoai hop bao 2 phoi: trong truong phoi khong co chu in
                ys, xs = np.where(both); paint_o[int(ys.min() / k):int(np.ceil((ys.max() + 1) / k)), int(xs.min() / k):int(np.ceil((xs.max() + 1) / k))] = 0
            if paint_o.any(): P = cv2.inpaint(P, paint_o, 5, cv2.INPAINT_TELEA)
            # 3) cat theo hop bao 2 phoi + le 5%, tren anh goc (da to chu)
            if both.sum() < 0.02 * S * S: y0, y1, x0, x1 = oy, oy + oh, ox, ox + ow
            else:
                ys, xs = np.where(both); ys = ys / k; xs = xs / k
                my, mx = MARGIN * (ys.max() - ys.min()), MARGIN * (xs.max() - xs.min())
                y0, y1 = int(max(ys.min() - my, 0)), int(min(ys.max() + 1 / k + my, n))
                x0, x1 = int(max(xs.min() - mx, 0)), int(min(xs.max() + 1 / k + mx, n))
            qa["pct_crop"][i] = 100 * (y1 - y0) * (x1 - x0) / max(oh * ow, 1)
            ci = cv2.resize(pad_square(P[y0:y1, x0:x1])[0], (S, S), interpolation=cv2.INTER_AREA)
            def crop_prob(pm):
                up = cv2.resize(pm.astype(np.float32), (n, n), interpolation=cv2.INTER_LINEAR)
                return largest_cc(cv2.resize(pad_square(up[y0:y1, x0:x1])[0], (S, S), interpolation=cv2.INTER_LINEAR) > 0.5)
            cR, cL = crop_prob(pr), crop_prob(pl); lung = cR | cL
            # chi dem hop chu chong len dai le cua khung cat (ngoai hop bao 2 phoi); OCR chay lai tren anh da to de dem chu con sot
            if both.sum() >= 0.02 * S * S:
                ys, xs = np.where(both); ly0, ly1, lx0, lx1 = ys.min() / k, (ys.max() + 1) / k, xs.min() / k, (xs.max() + 1) / k
            else: ly0 = ly1 = lx0 = lx1 = -1
            in_crop = lambda cs: sum(a * n < y1 and b * n > y0 and c * n < x1 and d * n > x0
                                     and not (a * n >= ly0 and b * n <= ly1 and c * n >= lx0 and d * n <= lx1) for a, b, c, d in cs)
            qa["n_hop_chu"][i] = in_crop(cen)
            paint = cv2.resize(pad_square(paint_o[y0:y1, x0:x1])[0], (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
            qa["pct_to"][i] = 100 * paint.mean()
            v1 = ci
            qa["n_chu_con_lai"][i] = in_crop(text_boxes(P, lung_od)[1])
            # 5) V2: xoa ngoai mask phoi (noi 1%)
            keep = ndimage.binary_dilation(lung, iterations=max(1, int(GROW_V2 * S))) if lung.any() else np.ones_like(lung)
            v2 = np.where(keep, v1, 0).astype(np.uint8)
            rmask = np.concatenate([thirds(cR), thirds(cL & ~cR)])
            area[i] = rmask.reshape(6, -1).mean(1)
            lab6 = np.zeros((S, S), np.uint8)
            for r in range(6): lab6[rmask[r]] = r + 1
            masks[i] = lab6[::4, ::4]
            # anh thu nho cho QA truc quan (chi luu cuc bo)
            th["goc"][i] = cv2.resize(p512, (T, T), interpolation=cv2.INTER_AREA)
            th["mask_goc"][i] = cv2.resize((R * 1 + L * 2).astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST)
            th["V1"][i] = cv2.resize(v1, (T, T), interpolation=cv2.INTER_AREA)
            th["to"][i] = cv2.resize(paint.astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST).astype(bool)
            # dac trung
            rmt = torch.from_numpy(rmask[:, None].astype(np.float32)).to(DEV)
            for v, img in (("V1", v1), ("V2", v2)):
                for e, m in enc.items():
                    if e == "chexfound":
                        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                            g, fm = m(chexfound.prep(img, DEV))
                        g, fm = g[0].float(), fm[0].float()
                    else:
                        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].expand(1, 3, S, S).to(DEV)
                        o = m(x); fm = o.patch_embeddings[0]; g = o.img_embedding[0]
                    w = F.interpolate(rmt, size=tuple(fm.shape[-2:]), mode="area")[:, 0]
                    ws = w.sum((1, 2))
                    f = torch.einsum("rhw,chw->rc", w, fm) / ws.clamp_min(1e-6)[:, None]; f[ws < 1e-6] = 0
                    regf[e, v][i] = f.cpu().numpy(); glob[e, v][i] = g.cpu().numpy(); gpool[e, v][i] = fm.mean((1, 2)).cpu().numpy()
            if (i + 1) % 100 == 0: print(f"{i+1}/{N} anh, {time.time()-t0:.0f}s", flush=True)
    print(f"Xong {N} anh, {time.time()-t0:.0f}s")
    for k in qk: print(f"  {k}: tb {np.nanmean(qa[k]):.3f} | >0: {int((qa[k] > 0).sum())}")
    tag = f"_try{NTRY}" if NTRY else ""
    np.savez_compressed(CACHE3 / f"img_v3{tag}.npz", img_id=ids, area=area, masks=masks, regions=np.array(REG), **qa,
                        **{f"regf_{e}_{v}": a for (e, v), a in regf.items()}, **{f"glob_{e}_{v}": a for (e, v), a in glob.items()},
                        **{f"gpool_{e}_{v}": a for (e, v), a in gpool.items()})
    np.savez_compressed(CACHE3 / f"qa_thumbs{tag}.npz", img_id=ids, **th)
    cfg = dict(thu_tu="anh goc xam -> pad vuong -> kiem tra dao am (PSPNet tren anh va anh dao) -> PSPNet torchxrayvision, giu thanh phan lon nhat moi ben "
               "-> cat hop bao 2 phoi + le 5% tren anh goc -> pad vuong -> 512. OCR chi phat hien (RapidOCR DBNet) tren ANH GOC 1024 TRUOC khi cat, bo hop cham phoi/nghieng/qua cao, "
               "to hop con lai (inpaint Telea) -> V1; V2 = V1 xoa ngoai mask phoi noi 1%", encoder_chinh="BioViL-T dong bang (patch 512 x 16x16; img_embedding 512)",
               encoder_b="CheXFound ViT-L/16 (patch 1024 x 32x32; CLS)", vung="6 vung: 1/3, 2/3 chieu cao moi phoi",
               nguong=dict(le=MARGIN, hop_cham_phoi=BOX_IN_LUNG, hop_nghieng_do=BOX_ANGLE, hop_cao=BOX_H, noi_phoi_ocr_px1024=LUNG_PAD, noi_V2=GROW_V2, dao_am="diem anh dao >= 1.5 x diem anh"),
               torch=torch.__version__, torchxrayvision=xrv.__version__)
    (CACHE3 / f"img_v3{tag}_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Xu ly 1 phim X-quang nguc thang -> dac trung 6 vung cho PG. CHEP tu PAG/scripts_v3/v3_image.py (vong lap moi anh),
chi giu nhanh PG dung: anh V2 + BioViL-T dong bang (regf_biovil_V2) va dien tich vung (area). Bo CheXFound, V1, anh thu nho QA.

Thu tu (giong luc hoc):
 1. anh xam (PIL convert 'L'), pad vuong, ban 512 (INTER_AREA)
 2. kiem tra dao am: PSPNet (torchxrayvision) tren anh va anh dao; anh dao "chac" hon >= 1,5 lan -> dao lai
 3. phoi = thanh phan lon nhat moi ben (xac suat > 0,5)
 4. OCR CHI PHAT HIEN (RapidOCR text_detector) tren anh goc 1024, bo hop cham phoi / nghieng > 15 do / cao > 6%;
    to (inpaint Telea) phan hop nam ngoai hop bao hai phoi
 5. cat theo hop bao hai phoi + le 5%, pad vuong, 512 -> V1; V2 = V1 xoa ngoai mask phoi (noi 1%)
 6. 6 vung = 1/3, 2/3 chieu cao moi phoi; dac trung vung = trung binh co trong so patch BioViL-T trong mask vung
 7. dac trung luu float16 (nhu img_v3.npz) roi doc lai float32
Can: file BioViL-T `biovil_t_image_model_proj_size_128.pt` (tham so duong_dan_biovil); trong so PSPNet do torchxrayvision
tu tai lan dau (~/.torchxrayvision) — may chu khong co mang thi chep san thu muc do.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

S, SD = 512, 1024
MARGIN, GROW_V2 = 0.05, 0.01
BOX_IN_LUNG, BOX_ANGLE, BOX_H, LUNG_PAD = 0.05, 15, 0.06, 15
REG = ["RU", "RM", "RL", "LU", "LM", "LL"]


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


def doc_anh_xam(anh):
    """duong dan / bytes / PIL.Image -> mang uint8 xam. Anh 8 bit: convert('L') nhu luc hoc.
    Anh 16 bit (khong co trong du lieu hoc): co gian min-max ve 0-255 truoc (convert('L') truc tiep se cat gia tri)."""
    import io
    if isinstance(anh, (bytes, bytearray)):
        anh = Image.open(io.BytesIO(anh))
    elif not isinstance(anh, Image.Image):
        anh = Image.open(anh)
    anh.load()
    if anh.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(anh, dtype=np.float64); lo, hi = a.min(), a.max()
        return np.round((a - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
    return np.asarray(anh.convert("L"))


class XuLyAnh:
    def __init__(self, duong_dan_biovil, thiet_bi=None, ocr=True):
        import torchxrayvision as xrv
        from health_multimodal.image.model.model import ImageModel
        from health_multimodal.image.model.types import ImageEncoderType
        self.dev = torch.device(thiet_bi or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.xrv = xrv
        self.seg = xrv.baseline_models.chestx_det.PSPNet().to(self.dev).eval()
        self.iL, self.iR = self.seg.targets.index("Left Lung"), self.seg.targets.index("Right Lung")
        self.enc = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE, joint_feature_size=128,
                              pretrained_model_path=str(duong_dan_biovil)).to(self.dev).eval()
        self.det = None
        if ocr:
            from rapidocr_onnxruntime import RapidOCR
            self.det = RapidOCR().text_detector

    def _segment(self, img512):
        x = torch.from_numpy(self.xrv.datasets.normalize(img512.astype(np.float32), 255))[None, None].to(self.dev)
        o = torch.sigmoid(self.seg(x))[0]
        return o[self.iR].cpu().numpy(), o[self.iL].cpu().numpy()

    def _text_boxes(self, img, lung_d):
        big = cv2.resize(img, (SD, SD), interpolation=cv2.INTER_AREA)
        boxes, _ = self.det(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
        m = np.zeros((SD, SD), np.uint8); n_lung = 0
        for b in (boxes if boxes is not None else []):
            b = np.asarray(b, np.float32); (cx, cy), (w, h), ang = cv2.minAreaRect(b)
            one = np.zeros((SD, SD), np.uint8); cv2.fillPoly(one, [np.round(b).astype(np.int32)], 1); one = one.astype(bool)
            if lung_d[one].mean() > BOX_IN_LUNG: n_lung += 1; continue
            a = ang % 90
            if min(a, 90 - a) > BOX_ANGLE or min(w, h) > BOX_H * SD: continue
            m |= one
        return m.astype(bool), n_lung

    @torch.no_grad()
    def __call__(self, anh):
        """-> dict(regf: (6, 512) float32, area: (6,) float32, qa: dict). Khong luu anh."""
        raw = doc_anh_xam(anh)
        P, (oy, ox, oh, ow) = pad_square(raw); n = P.shape[0]
        p512 = cv2.resize(P, (S, S), interpolation=cv2.INTER_AREA)
        qa = {}
        # 1) dao am
        pr, pl = self._segment(p512); qr, ql = self._segment(255 - p512)
        sc = lambda a, b: float(a[a > 0.5].sum() + b[b > 0.5].sum())  # noqa: E731
        s0, s1 = sc(pr, pl), sc(qr, ql)
        qa["dao_am"] = bool(s1 >= 1.5 * max(s0, 1.0))
        if qa["dao_am"]:
            P = 255 - P; p512 = 255 - p512; pr, pl = qr, ql
        # 2) phoi
        R, L = largest_cc(pr > 0.5), largest_cc(pl > 0.5)
        k = S / n; ry0, rx0, ry1, rx1 = int(oy * k), int(ox * k), int(np.ceil((oy + oh) * k)), int(np.ceil((ox + ow) * k))
        img_area = max((ry1 - ry0) * (rx1 - rx0), 1)
        qa["phoi_R"], qa["phoi_L"] = float(R.sum() / img_area), float(L.sum() / img_area)
        qa["ti_le_LR"] = float(L.sum() / max(R.sum(), 1))
        both = R | L
        inner = np.zeros_like(both); inner[ry0 + 2:ry1 - 2, rx0 + 2:rx1 - 2] = True
        qa["cham_mep"] = bool((both & ~inner).any())
        qa["du_phoi"] = bool(both.sum() >= 0.02 * S * S)
        # 4) OCR + to chu ngoai hop bao 2 phoi (truoc khi cat)
        lung_od = ndimage.binary_dilation(cv2.resize(both.astype(np.uint8), (SD, SD), interpolation=cv2.INTER_NEAREST).astype(bool), iterations=LUNG_PAD)
        qa["ocr"] = self.det is not None
        if self.det is not None:
            tm, _ = self._text_boxes(P, lung_od)
            paint_o = cv2.resize((ndimage.binary_dilation(tm, iterations=2) & ~lung_od).astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST)
            if both.sum() >= 0.02 * S * S:
                ys, xs = np.where(both); paint_o[int(ys.min() / k):int(np.ceil((ys.max() + 1) / k)), int(xs.min() / k):int(np.ceil((xs.max() + 1) / k))] = 0
            if paint_o.any(): P = cv2.inpaint(P, paint_o, 5, cv2.INPAINT_TELEA)
        # 3) cat theo hop bao 2 phoi + le 5%
        if both.sum() < 0.02 * S * S: y0, y1, x0, x1 = oy, oy + oh, ox, ox + ow
        else:
            ys, xs = np.where(both); ys = ys / k; xs = xs / k
            my, mx = MARGIN * (ys.max() - ys.min()), MARGIN * (xs.max() - xs.min())
            y0, y1 = int(max(ys.min() - my, 0)), int(min(ys.max() + 1 / k + my, n))
            x0, x1 = int(max(xs.min() - mx, 0)), int(min(xs.max() + 1 / k + mx, n))
        ci = cv2.resize(pad_square(P[y0:y1, x0:x1])[0], (S, S), interpolation=cv2.INTER_AREA)

        def crop_prob(pm):
            up = cv2.resize(pm.astype(np.float32), (n, n), interpolation=cv2.INTER_LINEAR)
            return largest_cc(cv2.resize(pad_square(up[y0:y1, x0:x1])[0], (S, S), interpolation=cv2.INTER_LINEAR) > 0.5)
        cR, cL = crop_prob(pr), crop_prob(pl); lung = cR | cL
        # 5) V2
        keep = ndimage.binary_dilation(lung, iterations=max(1, int(GROW_V2 * S))) if lung.any() else np.ones_like(lung)
        v2 = np.where(keep, ci, 0).astype(np.uint8)
        rmask = np.concatenate([thirds(cR), thirds(cL & ~cR)])
        area = rmask.reshape(6, -1).mean(1).astype(np.float32)
        qa["vung_rong"] = [REG[r] for r in range(6) if not rmask[r].any()]
        # 6) dac trung BioViL-T
        rmt = torch.from_numpy(rmask[:, None].astype(np.float32)).to(self.dev)
        x = torch.from_numpy(v2.astype(np.float32) / 255.0)[None, None].expand(1, 3, S, S).to(self.dev)
        fm = self.enc(x).patch_embeddings[0]
        w = F.interpolate(rmt, size=tuple(fm.shape[-2:]), mode="area")[:, 0]
        ws = w.sum((1, 2))
        f = torch.einsum("rhw,chw->rc", w, fm) / ws.clamp_min(1e-6)[:, None]; f[ws < 1e-6] = 0
        regf = f.cpu().numpy().astype(np.float16).astype(np.float32)   # nhu img_v3.npz (float16) -> load_base_v3 (float32)
        return dict(regf=regf, area=area, qa=qa)

"""Tien xu ly anh + dac trung vung theo PAG_V2 §5.

Thu tu (bat buoc):
 1. Pad vuong -> 1024. CHE CHU burn-in TRUOC moi buoc khac: top-hat sang + nguong cuong do -> noi ky tu theo chieu ngang
    -> giu thanh phan co hinh dang chu (cao 6-80 px, rong >= 0.8 x cao) nam NGOAI vung phoi -> inpaint (Telea).
    Ghi so vung che va dien tich che cho tung anh.
 2. Phan doan phoi bang mo hinh co san (torchxrayvision PSPNet) tren anh DA che; khong dung mask Set B.
 3. Cat theo khung bao 2 phoi + le 6% (thay Resize->CenterCrop), pad vuong -> 512.
 4. Chia moi phoi 3 vung o 1/3, 2/3 (RU RM RL LU LM LL).
 5. Dac trung vung = trung binh co trong so ban do dac trung 16x16 trong mask vung, cho 3 encoder:
      a) BioViL-T dong bang (CHINH) — patch_embeddings 512 chieu
      b) torchxrayvision ResNet50-res512 (thay cho EVA-X/CheXFound, khong co san offline) — layer4 2048 chieu
      c) BioViL-T control (fine-tune Silicodata SetA, best_model_ce.pth) — PHU
    + vector toan anh (tren anh da cat) cho tung encoder.
Dau ra: PAG/cache/img_v2.npz (id + dac trung -> PII, khong dua len GitHub). Chi in so lieu gop.
python v2_image.py [--qa]   (--qa: chi chay 50 anh, ghi luoi mask che chu do phan giai thap vao scratchpad de kiem tra)
"""
import os, re, sys, time, json
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from scipy import ndimage
from v2_lib import MERG, BASE, CACHE, REG

TEST_IMG = MERG.parent / "NEW DATA" / "test"
S0, S = 1024, 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QA = "--qa" in sys.argv
BG = "--bg" in sys.argv          # QA 07/10: 18/50 anh con sot chu -> xoa MOI THU ngoai bao loi 2 phoi (+2%) thay vi do tung chu
TAG = "_bg" if BG else ""


def outside_lung_blank(im, R, L, grow=0.02):
    """Giu phan trong bao loi cua 2 phoi (noi them grow*S0 px), to den phan con lai."""
    both = R | L
    if both.sum() < 0.02 * S0 * S0: return im, 0.0
    pts = cv2.findNonZero(both.astype(np.uint8)); keep = np.zeros_like(im, np.uint8)
    cv2.fillConvexPoly(keep, cv2.convexHull(pts), 1)
    g = int(grow * S0); keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))).astype(bool)
    return np.where(keep, im, 0).astype(im.dtype), 100 * keep.mean()

RE_ID = [re.compile(r"^(\d+)_(?=\D)"), re.compile(r"^\d{8}_\d{6}_(\d+)_"), re.compile(r"^[A-Za-z][A-Za-z ]*_(\d+)(?:_\d{8})?$")]
def xid(f):
    s = Path(f).stem
    for p in RE_ID:
        m = p.match(s)
        if m: return int(m.group(1))


def pad_square(im):
    h, w = im.shape; n = max(h, w)
    out = np.zeros((n, n), im.dtype); out[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = im
    return out


def burn_in_mask(im, lung=None):
    """im uint8 1024x1024. Tra ve mask chu (bool) + so thanh phan."""
    th = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    hi = np.percentile(im, 99.5)
    cand = (th > 50) & (im >= max(180, hi - 40))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand, bool); k = 0
    forbid = None if lung is None else ndimage.binary_dilation(lung, iterations=8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (6 <= h <= 80 and w >= 0.8 * h and w <= 600 and a >= 20): continue
        if w > 25 * h or a < 0.15 * w * h: continue                       # vet manh dai (mep xuong/vung chieu xa), khong phai chu
        box = lab[y:y + h, x:x + w] == i
        if forbid is not None and forbid[y:y + h, x:x + w][box].mean() > 0.3: continue
        keep[y:y + h, x:x + w] |= box; k += 1
    keep = ndimage.binary_dilation(keep, iterations=3)
    return keep, k


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


def biovil(ckpt=None):
    from health_multimodal.image.model.model import ImageModel
    from health_multimodal.image.model.types import ImageEncoderType
    m = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE, joint_feature_size=128,
                   pretrained_model_path=str(BASE / "Data DGCNN" / "Pre-train BioViL-T" / "biovil_t_image_model_proj_size_128.pt"))
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        miss, unexp = m.load_state_dict(sd, strict=False); assert len(miss) < 5 and len(unexp) < 5, (len(miss), len(unexp))
    return m.to(DEV).eval()


def main():
    files = []
    for root in (MERG / "train", TEST_IMG):
        for sub in ("normal", "silicosis"):
            for f in sorted(os.listdir(root / sub)):
                if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png"): files.append((xid(f), root / sub / f))
    ids = np.array([i for i, _ in files]); assert len(set(ids)) == len(ids) == 1835
    if QA:
        rng = np.random.default_rng(0); sel = rng.choice(len(files), 50, replace=False); files = [files[i] for i in sel]; ids = ids[sel]
    seg = xrv.baseline_models.chestx_det.PSPNet().to(DEV).eval()
    iL, iR = seg.targets.index("Left Lung"), seg.targets.index("Right Lung")
    enc = {} if QA else {"biovil_frozen": biovil(), "xrv": xrv.models.ResNet(weights="resnet50-res512-all").to(DEV).eval(),
                         "biovil_control": biovil(BASE / "Data DGCNN" / "Finetune CrossEntropy" / "best_model_ce.pth")}
    DIM = {"biovil_frozen": 512, "xrv": 2048, "biovil_control": 512}
    N = len(files)
    regf = {k: np.zeros((N, 6, DIM[k]), np.float16) for k in enc}; glob = {k: np.zeros((N, DIM[k]), np.float16) for k in enc}
    area = np.zeros((N, 6), np.float32); masks = np.zeros((N, 128, 128), np.uint8); txt = np.zeros((N, 128, 128), bool)
    qa = {k: np.zeros(N, np.float32) for k in ["n_chu", "pct_chu", "pct_crop", "empty_R", "empty_L", "chu_trong_phoi", "pct_giu"]}
    t0 = time.time()
    with torch.no_grad():
        for i, (_, p) in enumerate(files):
            im = pad_square(np.asarray(Image.open(p).convert("L")))
            im = cv2.resize(im, (S0, S0), interpolation=cv2.INTER_AREA)
            # 1) lan 1: phan doan so bo chi de biet cho nao la phoi (tranh xoa nham xuong/phoi), roi che chu
            def segment(img):
                x = torch.from_numpy(xrv.datasets.normalize(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32), 255))[None, None].to(DEV)
                o = torch.sigmoid(seg(x))[0]
                return {s: largest_cc((o[k] > 0.5).cpu().numpy()) for s, k in (("R", iR), ("L", iL))}
            l0 = segment(im); lung0 = cv2.resize((l0["R"] | l0["L"]).astype(np.uint8), (S0, S0), interpolation=cv2.INTER_NEAREST).astype(bool)
            tm, k = burn_in_mask(im, lung0)
            imm = cv2.inpaint(im, tm.astype(np.uint8), 5, cv2.INPAINT_TELEA) if k else im
            qa["n_chu"][i] = k; qa["pct_chu"][i] = 100 * tm.mean(); txt[i] = tm[::8, ::8]
            # 2) phan doan lai tren anh DA che
            lg = segment(imm)
            R = cv2.resize(lg["R"].astype(np.uint8), (S0, S0), interpolation=cv2.INTER_NEAREST).astype(bool)
            L = cv2.resize(lg["L"].astype(np.uint8), (S0, S0), interpolation=cv2.INTER_NEAREST).astype(bool)
            qa["empty_R"][i] = R.sum() < 0.01 * S0 * S0; qa["empty_L"][i] = L.sum() < 0.01 * S0 * S0
            qa["chu_trong_phoi"][i] = (tm & (R | L)).sum() > 0
            if BG: imm, qa["pct_giu"][i] = outside_lung_blank(imm, R, L)
            # 3) cat theo khung bao phoi + le
            both = R | L
            if both.sum() < 0.02 * S0 * S0: y0, y1, x0, x1 = 0, S0, 0, S0
            else:
                ys, xs = np.where(both); mg = int(0.06 * S0)
                y0, y1, x0, x1 = max(ys.min() - mg, 0), min(ys.max() + mg, S0), max(xs.min() - mg, 0), min(xs.max() + mg, S0)
            qa["pct_crop"][i] = 100 * (y1 - y0) * (x1 - x0) / S0 ** 2
            crop = lambda a: cv2.resize(pad_square(a[y0:y1, x0:x1]), (S, S), interpolation=cv2.INTER_AREA if a.dtype == np.uint8 and a.max() > 1 else cv2.INTER_NEAREST)
            ci = crop(imm); cR = crop(R.astype(np.uint8)).astype(bool); cL = crop(L.astype(np.uint8)).astype(bool)
            rm = np.concatenate([thirds(cR), thirds(cL)])                              # (6,512,512)
            area[i] = rm.reshape(6, -1).mean(1)
            lab = np.zeros((S, S), np.uint8)
            for r in range(6): lab[rm[r]] = r + 1
            masks[i] = lab[::4, ::4]
            # 5) dac trung vung
            w = F.interpolate(torch.from_numpy(rm[:, None].astype(np.float32)).to(DEV), size=(16, 16), mode="area")[:, 0]
            ws = w.sum((1, 2)).clamp_min(1e-6)
            for name, m in enc.items():
                if name == "xrv":
                    x = torch.from_numpy(xrv.datasets.normalize(ci.astype(np.float32), 255))[None, None].to(DEV)
                    mm = m.model; h = mm.maxpool(mm.relu(mm.bn1(mm.conv1(x))))
                    fm = mm.layer4(mm.layer3(mm.layer2(mm.layer1(h))))[0]; g = fm.mean((1, 2))
                else:
                    x = torch.from_numpy(ci.astype(np.float32) / 255.0)[None, None].expand(1, 3, S, S).to(DEV)
                    o = m(x); fm = o.patch_embeddings[0]; g = o.img_embedding[0]
                f = torch.einsum("rhw,chw->rc", w, fm) / ws[:, None]; f[w.sum((1, 2)) < 1e-6] = 0
                regf[name][i] = f.cpu().numpy(); glob[name][i] = g.cpu().numpy()
            if (i + 1) % 200 == 0: print(f"{i+1}/{N} anh, {time.time()-t0:.0f}s", flush=True)
    print(f"Xong {N} anh, {time.time()-t0:.0f}s")
    print("anh co che chu: %d/%d | so vung che trung vi %.0f | dien tich che trung vi %.2f%% (p95 %.2f%%) | chu cham phoi: %d"
          % ((qa["n_chu"] > 0).sum(), N, np.median(qa["n_chu"]), np.median(qa["pct_chu"]), np.percentile(qa["pct_chu"], 95), qa["chu_trong_phoi"].sum()))
    print("phoi rong R/L: %d/%d | khung cat chiem trung vi %.0f%% anh" % (qa["empty_R"].sum(), qa["empty_L"].sum(), np.median(qa["pct_crop"])))
    if QA:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        out = Path(os.environ.get("QA_DIR", "."))
        fig, ax = plt.subplots(5, 10, figsize=(20, 10))
        for a, t, m in zip(ax.flat, txt, masks):
            a.imshow(cv2.resize(m, (128, 128)) > 0, cmap="Blues", alpha=1.0); a.imshow(np.ma.masked_where(~t, t), cmap="autumn"); a.axis("off")
        fig.savefig(out / "qa_mask_chu_phoi.png", dpi=70, bbox_inches="tight"); return
    np.savez_compressed(CACHE / f"img_v2{TAG}.npz", img_id=ids, area=area, masks=masks, txt=txt, regions=np.array(REG),
                        **{f"regf_{k}": v for k, v in regf.items()}, **{f"glob_{k}": v for k, v in glob.items()}, **qa)
    cfg = dict(thu_tu="pad vuong 1024 -> che chu (top-hat + nguong, ngoai phoi, inpaint Telea) -> PSPNet tren anh da che -> "
               + ("TO DEN moi thu ngoai bao loi 2 phoi + 2% -> " if BG else "") + "cat khung bao phoi + le 6% -> pad vuong 512 -> 6 vung 1/3, 2/3",
               encoder_chinh="BioViL-T dong bang (patch_embeddings 512 x 16 x 16)", encoder_b="torchxrayvision ResNet50-res512-all (thay EVA-X/CheXFound — khong co offline)",
               encoder_phu="BioViL-T control (Finetune CrossEntropy/best_model_ce.pth, SetA)", dac_trung_vung="trung binh co trong so ban do 16x16 trong mask vung",
               input=S, torch=torch.__version__, torchxrayvision=xrv.__version__)
    (CACHE / f"img_v2{TAG}_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

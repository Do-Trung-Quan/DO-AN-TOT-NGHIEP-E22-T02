"""Chon nguong cho PG (ban khong dung tinh) CHI tren dev OOF — hai nhom: co phim / khong phim.

  python v3_chon_nguong.py                      # doc cache_v3/run_PG-noctx_nghe10.npz (khoa oof_cal)
  python v3_chon_nguong.py --ten PG             # ban co tinh (de so sanh)
  python v3_chon_nguong.py --npz DUONG_DAN      # file khac (chi de chay thu)

THAY DAN (chon tren dev, khong chon tren test). Script KHONG doc test_cal va KHONG dung nhan test.
Moi nhom tinh RIENG (mo hinh duoc hieu chuan rieng cho nhom co phim va khong phim).
QUY TAC CHINH (thay dong y): CAN BANG do nhay va do dac hieu = Youden (quy tac 2 ben duoi). Quy tac do nhay 80% chi la THAM KHAO
(y ban dau cua nhom), de thay thay cai gia cua viec uu tien bat benh.

 Quy tac 1 (THAM KHAO) — do nhay muc tieu 80% (sang loc):
   nguong t sao cho >= s* ca benh dev co diem >= t. s* khong dat dung 0,80 ma duoc nang len toi muc nho nhat ma
   can duoi (phan vi 5%) cua do nhay "chon nguong tren 4 fold, do tren fold con lai" van >= 0,80 — de bu sai so lay mau
   va thien lech do chon nguong. (Giong y "bien an toan 0.80" cua New Way.)
 Quy tac 2 (CHINH) — can bang (Youden): nguong lam cuc dai (do nhay + do dac hieu - 1); kem khoang dao dong bootstrap cua nguong.
 Bao cao: do nhay, do dac hieu, ti le nguoi bi bao "co nguy co" (cai gia that cua nguong), do nhay theo tinh (kiem tra,
 tinh KHONG la dau vao cua mo hinh khong tinh), bang danh doi 70-90%.
 "Co nguy co" khi xac_suat >= nguong.

Luu y: diem oof_cal tren dev la diem cua MOT mo hinh fold; xac suat app tra la TRUNG BINH 5 mo hinh fold -> phan bo hep hon
mot chut. Nguong chon tu OOF ap len diem trung binh chi gan dung; khong the kiem tren dev (moi nguoi dev da nam trong
tap hoc cua 4/5 mo hinh).

Dau ra: results_v3/<--ra>/BAO_CAO.md, nguong.csv, nguong_pg.json, run_config.json. Chi so lieu gop, khong co id.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_base import load_base_v3, CACHE3, FOLDS, run_config, TAG  # noqa: E402
from v3_lib import RES3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ten", default="PG-noctx", choices=["PG", "PG-noctx"])
ap.add_argument("--npz", type=Path, default=None)
ap.add_argument("--ra", default=None, help="thu muc trong results_v3 (mac dinh theo --ten)")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--muc-tieu", type=float, default=0.80)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

NPZ = args.npz or (CACHE3 / f"run_{args.ten}{TAG}.npz")
RA = args.ra or ("tuan8_nguong_noctx" if args.ten == "PG-noctx" else "tuan8_nguong_PG_co_tinh")
t0 = time.time()
B = load_base_v3()
y, hi, dev, fold = B["y"].astype(int), B["has_img"], B["dev"], B["fold"]
uid = np.asarray(B["uid"]).astype(str)
tinh = B["raw"]["tinh"].astype(str).to_numpy()
oof = np.load(NPZ, allow_pickle=False)["oof_cal"]          # CHI oof_cal (dev). test_cal khong duoc doc.
assert np.isfinite(oof[dev]).all(), "oof_cal thieu diem o hang dev"
print(f"[nguong] {NPZ.name} | dev {int(dev.sum())} | muc tieu do nhay {args.muc_tieu} | {args.n_boot} lan bootstrap theo nguoi")


# ------------------------------------------------------------------ ham
def thr_for_sens(pos, s):
    """Nguong lon nhat sao cho ti le ca benh co diem >= nguong la >= s."""
    t = float(np.quantile(pos, 1 - s, method="lower"))
    assert np.mean(pos >= t) >= s - 1e-12
    return t


def youden_thr(sc, yy):
    o = np.argsort(-sc); s, p = sc[o], yy[o]
    tp, fp = np.cumsum(p), np.cumsum(1 - p)
    j = tp / p.sum() - fp / (1 - p).sum()
    return float(s[int(np.argmax(j))])


def ss(sc, yy, t):
    f = sc >= t
    return float(f[yy == 1].mean()), float((~f)[yy == 0].mean()), float(f.mean())     # do nhay, dac hieu, ti le bi bao


def nested_flags(sc, yy, fd, rule, s=None):
    """Nguong tung fold chon tu 4 fold con lai; ap len fold do. Tra ve co bao nguy co (cho moi hang)."""
    fl = np.zeros(len(sc), bool)
    for k in np.unique(fd):
        tr, va = fd != k, fd == k
        t = thr_for_sens(sc[tr & (yy == 1)], s) if rule == "sens" else youden_thr(sc[tr], yy[tr])
        fl[va] = sc[va] >= t
    return fl


def boot_weights(uids, n_boot, seed):
    """Ma tran trong so (n_boot x n): lay mau lai THEO NGUOI (moi luot kham cua nguoi duoc chon tinh 1 lan moi lan chon)."""
    u, inv = np.unique(uids, return_inverse=True)
    rng = np.random.default_rng(seed)
    cnt = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(n_boot)])
    return cnt[:, inv].astype(np.float32)


def boot_sens_spec(W, yy, flags):
    """flags: (n,) hoac (n,G). Tra do nhay / dac hieu cho tung lan bootstrap."""
    F = flags.reshape(len(yy), -1).astype(np.float32)
    P, N = W * (yy == 1), W * (yy == 0)
    sens = (P @ F) / P.sum(1, keepdims=True)
    spec = (N @ (1 - F)) / N.sum(1, keepdims=True)
    return sens, spec


def boot_youden_thr(W, sc, yy):
    o = np.argsort(-sc); s = sc[o]; p, q = (yy[o] == 1).astype(np.float32), (yy[o] == 0).astype(np.float32)
    out = []
    for i in range(0, len(W), 100):
        w = W[i:i + 100][:, o]
        tp, fp = np.cumsum(w * p, 1), np.cumsum(w * q, 1)
        j = tp / tp[:, -1:] - fp / fp[:, -1:]
        out.append(s[np.argmax(j, 1)])
    return np.concatenate(out)


def pct(a, q):
    return float(np.percentile(a, q))


# ------------------------------------------------------------------ tinh theo nhom
rep, tab, J = [], [], {}
GRID = np.round(np.arange(args.muc_tieu, 0.9501, 0.005), 3)
for key, ten, g in (("co_phim", "Có phim", hi), ("khong_phim", "Không phim", ~hi)):
    m = dev & g
    sc, yy, fd, uu, tt = oof[m], y[m], fold[m], uid[m], tinh[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    auc = roc_auc_score(yy, sc)
    W = boot_weights(uu, args.n_boot, args.seed)
    J[key] = dict(n=int(m.sum()), ca_benh=npos, auc_oof=auc, ti_le_benh=npos / len(yy))
    print(f"\n=== {ten}: n={m.sum()} | ca benh {npos} ({npos / len(yy):.1%}) | AUC OOF {auc:.3f}")

    # ---- quy tac 1: do nhay muc tieu
    Fg = np.stack([nested_flags(sc, yy, fd, "sens", s) for s in GRID], 1)          # (n, G) co bao long nhau
    sens_b, _ = boot_sens_spec(W, yy, Fg)
    lo5 = np.percentile(sens_b, 5, axis=0)
    ok = np.where(lo5 >= args.muc_tieu)[0]
    gi = int(ok[0]) if len(ok) else len(GRID) - 1
    s_star = float(GRID[gi]); dat = bool(len(ok))
    t1 = thr_for_sens(sc[yy == 1], s_star)
    t80 = thr_for_sens(sc[yy == 1], args.muc_tieu)
    fl_n = Fg[:, gi]; fl_n80 = Fg[:, 0]
    sb, pb = boot_sens_spec(W, yy, np.stack([fl_n, fl_n80], 1))
    se_n, sp_n, _ = ss(np.where(fl_n, 1.0, 0.0), yy, 0.5)
    se80_n, sp80_n, _ = ss(np.where(fl_n80, 1.0, 0.0), yy, 0.5)
    se1, sp1, fr1 = ss(sc, yy, t1)
    se80, sp80, fr80 = ss(sc, yy, t80)
    J[key]["do_nhay_0.80"] = dict(
        nguong=t1, muc_tieu_hieu_chinh=s_star, dat_can_duoi_0_80=dat,
        do_nhay_dev=se1, do_dac_hieu_dev=sp1, ti_le_bao_dev=fr1,
        do_nhay_long_nhau=[se_n, pct(sb[:, 0], 2.5), pct(sb[:, 0], 97.5)], can_duoi_5pct=pct(sb[:, 0], 5),
        do_dac_hieu_long_nhau=[sp_n, pct(pb[:, 0], 2.5), pct(pb[:, 0], 97.5)],
        ghi_chu_khong_hieu_chinh=dict(nguong=t80, do_nhay_dev=se80, do_dac_hieu_dev=sp80, ti_le_bao_dev=fr80,
                                      do_nhay_long_nhau=[se80_n, pct(sb[:, 1], 2.5), pct(sb[:, 1], 97.5)]))
    print(f"  Quy tac 1 (do nhay): s*={s_star:.3f} {'(DAT)' if dat else '(KHONG DAT can duoi 0,80 trong luoi!)'} | nguong {t1:.4f}"
          f" | dev: nhay {se1:.3f} dac hieu {sp1:.3f} bi bao {fr1:.1%}"
          f" | long nhau: nhay {se_n:.3f} [{pct(sb[:, 0], 2.5):.3f}-{pct(sb[:, 0], 97.5):.3f}] dac hieu {sp_n:.3f}")
    print(f"     (khong nang muc tieu: nguong {t80:.4f} | nhay long nhau {se80_n:.3f} [{pct(sb[:, 1], 2.5):.3f}-{pct(sb[:, 1], 97.5):.3f}])")

    # ---- quy tac 2: Youden
    ty = youden_thr(sc, yy)
    tb = boot_youden_thr(W, sc, yy)
    fy = nested_flags(sc, yy, fd, "youden")
    syb, spb = boot_sens_spec(W, yy, fy)
    sey, spy, fry = ss(sc, yy, ty)
    sey_n, spy_n, _ = ss(np.where(fy, 1.0, 0.0), yy, 0.5)
    J[key]["youden"] = dict(
        nguong=ty, khoang_dao_dong_nguong_95=[pct(tb, 2.5), pct(tb, 97.5)],
        do_nhay_dev=sey, do_dac_hieu_dev=spy, ti_le_bao_dev=fry,
        do_nhay_long_nhau=[sey_n, pct(syb[:, 0], 2.5), pct(syb[:, 0], 97.5)],
        do_dac_hieu_long_nhau=[spy_n, pct(spb[:, 0], 2.5), pct(spb[:, 0], 97.5)])
    print(f"  Quy tac 2 (Youden): nguong {ty:.4f} [dao dong {pct(tb, 2.5):.4f}-{pct(tb, 97.5):.4f}] | dev: nhay {sey:.3f} dac hieu {spy:.3f} bi bao {fry:.1%}"
          f" | long nhau: nhay {sey_n:.3f} dac hieu {spy_n:.3f}")

    # ---- bang danh doi
    for s in (0.70, 0.75, 0.80, 0.85, 0.90):
        t = thr_for_sens(sc[yy == 1], s); a, b, c = ss(sc, yy, t)
        tab.append(dict(nhom=ten, do_nhay_dev=s, nguong=t, do_dac_hieu=b, ti_le_bi_bao=c))

    # ---- theo tinh (kiem tra; tinh khong la dau vao cua ban noctx)
    rows = []
    for tn in sorted(set(tt)):
        k = tt == tn; p_ = int((yy[k] == 1).sum())
        row = dict(nhom=ten, tinh=tn, n=int(k.sum()), ca_benh=p_)
        for rule, t in (("do_nhay_0.80", t1), ("youden", ty)):
            if p_ >= 20:
                a, b, c = ss(sc[k], yy[k], t); row.update({f"nhay_{rule}": a, f"dac_hieu_{rule}": b, f"ti_le_bao_{rule}": c})
            else:
                row.update({f"nhay_{rule}": np.nan, f"dac_hieu_{rule}": np.nan, f"ti_le_bao_{rule}": float((sc[k] >= t).mean())})
        rows.append(row)
    J[key]["theo_tinh"] = rows
    print("  Theo tinh (nhay chi tinh khi >= 20 ca benh):", {r["tinh"]: (round(r["nhay_do_nhay_0.80"], 2) if np.isfinite(r["nhay_do_nhay_0.80"]) else f"n={r['ca_benh']} ca") for r in rows})

# ------------------------------------------------------------------ ghi
d = RES3 / RA; d.mkdir(parents=True, exist_ok=True)
pd.DataFrame(tab).to_csv(d / "danh_doi.csv", index=False, encoding="utf-8-sig")
flat = []
for key, ten in (("co_phim", "Có phim"), ("khong_phim", "Không phim")):
    for rule, nm in (("youden", "**CHÍNH** — cân bằng (Youden)"), ("do_nhay_0.80", "tham khảo — độ nhạy ≥ 0,80 (có dự phòng)")):
        r = J[key][rule]
        flat.append(dict(nhom=ten, quy_tac=nm, nguong=r["nguong"], do_nhay_dev=r["do_nhay_dev"], do_dac_hieu_dev=r["do_dac_hieu_dev"],
                         ti_le_bao_dev=r["ti_le_bao_dev"], do_nhay_long_nhau=r["do_nhay_long_nhau"][0],
                         do_nhay_ktc=f"[{r['do_nhay_long_nhau'][1]:.3f}–{r['do_nhay_long_nhau'][2]:.3f}]",
                         do_dac_hieu_long_nhau=r["do_dac_hieu_long_nhau"][0]))
pd.DataFrame(flat).to_csv(d / "nguong.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([r for k in J for r in J[k]["theo_tinh"]]).to_csv(d / "theo_tinh.csv", index=False, encoding="utf-8-sig")
sha16 = hashlib.sha256(NPZ.read_bytes()).hexdigest()[:16]
(d / "nguong_pg.json").write_text(json.dumps(dict(
    mo_hinh=args.ten, nguon="dev out-of-fold (oof_cal); KHONG dung test", ma_bam_npz=sha16, quy_tac_bao="bao nguy co neu xac_suat >= nguong",
    quy_tac_chinh="youden", ghi_chu_quy_tac="thay dong y can bang do nhay/do dac hieu (Youden) chon tren dev; do_nhay_0.80 chi de tham khao",
    luu_y="oof_cal la diem cua mot mo hinh fold; app tra trung binh 5 fold nen phan bo hep hon mot chut",
    muc_tieu_do_nhay=args.muc_tieu, n_boot=args.n_boot, seed=args.seed,
    co_phim={k: v for k, v in J["co_phim"].items() if k != "theo_tinh"},
    khong_phim={k: v for k, v in J["khong_phim"].items() if k != "theo_tinh"}), ensure_ascii=False, indent=1))
run_config(f"{RA}", __file__, data=[NPZ, FOLDS], mo_hinh=args.ten, muc_tieu=args.muc_tieu, n_boot=args.n_boot, khong_dung_test=True)


def md(df, nd=3):
    f = lambda x: f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else ("—" if isinstance(x, float) else str(x))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)])


L = [f"# Chọn ngưỡng PG ({args.ten}) trên dev", "",
     f"- Nguồn: `{NPZ.name}` (khoá `oof_cal`, mã băm `{sha16}`). **Không dùng tập test**, không đọc `test_cal`.",
     f"- Mỗi nhóm tính riêng. Bootstrap {args.n_boot} lần theo người (seed {args.seed}). \"Có nguy cơ\" khi xác suất ≥ ngưỡng.",
     "- **Độ nhạy / độ đặc hiệu \"lồng nhau\"** là ước lượng trung thực: ngưỡng của mỗi fold được chọn từ 4 fold còn lại, rồi đo trên fold đó. "
     "Các cột `dev` là đo trên chính dữ liệu dùng để chọn ngưỡng nên lạc quan hơn.",
     "- Điểm `oof_cal` là của một mô hình fold, còn app trả trung bình 5 mô hình fold nên phân bố hẹp hơn một chút; ngưỡng chỉ gần đúng.", "",
     "- **Quy tắc chính là cân bằng độ nhạy và độ đặc hiệu (Youden), theo chỉ đạo của thầy.** Quy tắc độ nhạy ≥ 0,80 chỉ để tham khảo.", "",
     "## Ngưỡng đề xuất", "", md(pd.DataFrame(flat)), "",
     "## Cách đọc", "",
     "- `nguong`: ngưỡng đưa vào web. `do_nhay_dev`: tỉ lệ ca bệnh dev có điểm ≥ ngưỡng. `do_dac_hieu_dev`: tỉ lệ người không bệnh có điểm < ngưỡng.",
     "- `ti_le_bao_dev`: tỉ lệ **mọi người** bị gán \"có nguy cơ\", là cái giá thật của ngưỡng.", ""]
for key, ten in (("co_phim", "Có phim"), ("khong_phim", "Không phim")):
    r = J[key]; a = r["do_nhay_0.80"]; y_ = r["youden"]
    L += [f"### {ten} — n={r['n']}, {r['ca_benh']} ca bệnh ({r['ti_le_benh']:.1%}), AUC OOF {r['auc_oof']:.3f}", "",
          f"- **CHÍNH — Cân bằng (Youden):** ngưỡng **{y_['nguong']:.4f}** (dao động 95% {y_['khoang_dao_dong_nguong_95'][0]:.4f}–{y_['khoang_dao_dong_nguong_95'][1]:.4f}); "
          f"độ nhạy lồng nhau {y_['do_nhay_long_nhau'][0]:.3f} [{y_['do_nhay_long_nhau'][1]:.3f}–{y_['do_nhay_long_nhau'][2]:.3f}], "
          f"độ đặc hiệu lồng nhau {y_['do_dac_hieu_long_nhau'][0]:.3f} [{y_['do_dac_hieu_long_nhau'][1]:.3f}–{y_['do_dac_hieu_long_nhau'][2]:.3f}]; "
          f"bị báo \"có nguy cơ\" {y_['ti_le_bao_dev']:.1%} (dev, độ nhạy dev {y_['do_nhay_dev']:.3f}, độ đặc hiệu dev {y_['do_dac_hieu_dev']:.3f}).",
          f"- *Tham khảo — nếu ưu tiên bắt ≥ 80% ca bệnh:* nâng mục tiêu lên s* = {a['muc_tieu_hieu_chinh']:.3f} "
          f"({'đạt' if a['dat_can_duoi_0_80'] else '**KHÔNG đạt cận dưới 0,80 trong lưới tới 0,95**'}) → ngưỡng {a['nguong']:.4f}; "
          f"độ nhạy lồng nhau {a['do_nhay_long_nhau'][0]:.3f} [{a['do_nhay_long_nhau'][1]:.3f}–{a['do_nhay_long_nhau'][2]:.3f}]; "
          f"độ đặc hiệu lồng nhau {a['do_dac_hieu_long_nhau'][0]:.3f}; bị báo {a['ti_le_bao_dev']:.1%} (dev).",
          f"- *Tham khảo — nhắm đúng 0,80 không dự phòng:* ngưỡng {a['ghi_chu_khong_hieu_chinh']['nguong']:.4f}, độ nhạy lồng nhau "
          f"{a['ghi_chu_khong_hieu_chinh']['do_nhay_long_nhau'][0]:.3f} [{a['ghi_chu_khong_hieu_chinh']['do_nhay_long_nhau'][1]:.3f}–{a['ghi_chu_khong_hieu_chinh']['do_nhay_long_nhau'][2]:.3f}].", ""]
L += ["## Bảng đánh đổi (ngưỡng theo độ nhạy trên dev)", "", md(pd.DataFrame(tab)), "",
      "## Theo tỉnh (kiểm tra; độ nhạy chỉ tính khi ≥ 20 ca bệnh)", "", md(pd.DataFrame([r for k in J for r in J[k]["theo_tinh"]])), ""]
(d / "BAO_CAO.md").write_text("\n".join(L), encoding="utf-8")
print(f"\n[nguong] XONG {time.time() - t0:.0f}s -> results_v3/{RA}/ (BAO_CAO.md, nguong.csv, nguong_pg.json)")

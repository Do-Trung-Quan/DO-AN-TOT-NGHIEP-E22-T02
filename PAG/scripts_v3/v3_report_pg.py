"""Bao cao buoc 4 (ket qua mo ta PG) va buoc 5 (kiem dinh H1-H3 theo file khai bao truoc). Chay CPU, chi dev OOF.

  python v3_report_pg.py --buoc 4     # -> results_v3/tuan3_PG_nghe10/BAO_CAO.md  (chi so mo ta, CHUA kiem dinh)
  python v3_report_pg.py --buoc 5     # -> results_v3/tuan4_so_sanh_chinh_nghe10/BAO_CAO.md (theo khai_bao_truoc_v3.json)
  them --smoke de doc ket qua chay thu (*_smoke), chi de kiem tra script

Doc: cache_v3/run_<PG|PG-global|PG-shuffle>_nghe10.npz, run_baselines_v3_nghe10.npz (C, L0, L2, X0, X2, B3b, GAT),
run_T0_nghe10.npz, run_T2_nghe10.npz. Thieu file nao thi bo mo hinh do va ghi ro. Test KHONG cham. Chi ghi so lieu gop.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_base import load_base_v3, CACHE3, RES3, TAG, run_config  # noqa: E402
from v2_lib import Report, boot_ci, metrics, within_batch_auc, NBOOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--buoc", type=int, choices=[4, 5], required=True)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
SM = "_smoke" if args.smoke else ""
KB = RES3 / f"tuan4_so_sanh_chinh{TAG}" / "khai_bao_truoc_v3.json"

B = load_base_v3()
y, hi, uid, batch, dev = B["y"], B["has_img"], B["uid"], B["batch"], B["dev"]
dn = pd.Series(B["tinh"]).astype(str).str.lower().str.contains("nai").to_numpy()
BL_F = CACHE3 / f"run_baselines_v3{TAG}.npz"
BL = np.load(BL_F, allow_pickle=False) if BL_F.exists() else None
MISSING = []


def pred(name):
    """OOF da hieu chinh Platt trong fold; None neu chua co file."""
    if name in ("C", "L0", "L2", "X0", "X2", "B3b", "GAT"):
        if BL is None:
            MISSING.append(f"{name} ({BL_F.name})"); return None
        return BL[f"oof_cal_{name}"]
    if name == "PG-drop":
        f = CACHE3 / f"run_PG{TAG}{SM}.npz"; key = "oof_drop_cal"
    else:
        f = CACHE3 / f"run_{name}{TAG}{SM if name.startswith('PG') else ''}.npz"; key = "oof_cal"
    if not f.exists():
        MISSING.append(f"{name} ({f.name})"); return None
    return np.load(f, allow_pickle=False)[key]


GRP = {"có ảnh": dev & hi, "không ảnh": dev & ~hi, "không ảnh — Đồng Nai": dev & ~hi & dn,
       "không ảnh — tỉnh khác": dev & ~hi & ~dn}


def sens_at(yy, pp, spec):
    thr = np.quantile(pp[yy == 0], spec); return float((pp[yy == 1] > thr).mean())


def mrow(name, g, m, p):
    k = m & np.isfinite(p)
    if k.sum() < 20 or len(np.unique(y[k])) < 2:
        return None
    yy, pp = y[k], p[k]
    r = metrics(yy, pp, batch[k] if g == "có ảnh" else None)
    a, lo, hi_, _ = boot_ci(lambda i: roc_auc_score(yy[i], pp[i]), uid[k])
    out = dict(mo_hinh=name, nhom=g, n=r.pop("n"), duong=r.pop("duong"), roc_auc=a, roc_auc_ci=f"[{lo:.3f}–{hi_:.3f}]")
    out.update({kk: vv for kk, vv in r.items() if kk != "roc_auc"})
    out["nhay_tai_dh80"], out["nhay_tai_dh90"] = sens_at(yy, pp, 0.8), sens_at(yy, pp, 0.9)
    return out


def paired(a, b, nhom, chi_so, bien):
    pa, pb = pred(a), pred(b)
    if pa is None or pb is None:
        return None
    m = GRP[nhom]; k = m & np.isfinite(pa) & np.isfinite(pb)
    yy, A, Bb, bb = y[k], pa[k], pb[k], batch[k]
    if chi_so == "AUC trong lo":
        f = lambda i: within_batch_auc(yy[i], A[i], bb[i]) - within_batch_auc(yy[i], Bb[i], bb[i])
        va, vb = within_batch_auc(yy, A, bb), within_batch_auc(yy, Bb, bb)
    else:
        f = lambda i: roc_auc_score(yy[i], A[i]) - roc_auc_score(yy[i], Bb[i])
        va, vb = roc_auc_score(yy, A), roc_auc_score(yy, Bb)
    d, lo, hi_, v = boot_ci(f, uid[k])
    return dict(n=int(k.sum()), chi_so_a=va, chi_so_b=vb, delta=d, ktc=f"[{lo:+.3f}; {hi_:+.3f}]", ktc_duoi=lo, ktc_tren=hi_,
                P_delta_duong=float((v > 0).mean()), p_mot_phia=float((v <= bien).mean()))


def holm(ps, alpha):
    o = np.argsort(ps); ok = np.zeros(len(ps), bool)
    for r, i in enumerate(o):
        if ps[i] <= alpha / (len(ps) - r):
            ok[i] = True
        else:
            break
    return ok


NOTE = ("*`roc_auc`: xác suất mô hình chấm một ca bệnh cao hơn một ca khỏe (0,5 = đoán bừa), KTC 95% bootstrap theo người. "
        "`auc_trong_lo`: AUC tính riêng từng lô chụp rồi lấy trung bình theo cỡ lô (chỉ nhóm có ảnh). `pr_auc`: so với tỉ lệ dương của nhóm. "
        "`brier` (so với `brier_nen` = luôn đoán tỉ lệ nền), `ece`: xác suất có “thật thà” không, càng thấp càng tốt. "
        "`do_doc_hc` (1 là lý tưởng), `chan_hc` (0 là lý tưởng). `nhay_tai_dh80/90`: tỉ lệ ca bệnh bắt được khi chỉ báo nhầm 20% / 10% người khỏe. "
        "Mọi số là dự đoán out-of-fold trên dev, xác suất đã hiệu chỉnh Platt trong fold.*")

if args.buoc == 4:
    R = Report(f"tuan3_PG{TAG}{SM}", "Tuần 3 — Kết quả mô tả PG (bước 4, CHƯA kiểm định)")
    R.text("Kết quả **mô tả**: chưa phải kiểm định giả thuyết. Kiểm định H1–H3 làm ở bước 5 theo `khai_bao_truoc_v3.json`. "
           "Mốc so sánh chạy lại với cùng cách mã hoá 10 cấp nghề (`_nghe10`). Test không chấm.")
    names = ["PG-global", "PG", "PG-shuffle", "PG-drop", "C", "L0", "L2", "X0", "X2", "B3b", "GAT", "T0", "T2"]
    rows = [mrow(n, g, m, p) for n in names for p in [pred(n)] if p is not None for g, m in GRP.items()]
    T = pd.DataFrame([r for r in rows if r])
    R.table(T, "chi_so", "Chỉ số theo nhóm", note=NOTE)
    for nm in ("PG-global", "PG"):
        f = CACHE3 / f"selected_{nm}{TAG}{SM}.json"
        if f.exists():
            R.table(pd.DataFrame(json.loads(f.read_text())), f"cau_hinh_chon_{nm}", f"Cấu hình được chọn theo fold — {nm}")
else:
    kb_raw = KB.read_bytes(); kb = json.loads(kb_raw.decode("utf-8")); h = hashlib.sha256(kb_raw).hexdigest()[:16]
    R = Report(f"tuan4_so_sanh_chinh{TAG}{SM}", "Tuần 4 — Kiểm định giả thuyết khai báo trước (bước 5)")
    R.text(f"Khai báo trước: `{KB.name}` (mã băm `{h}`). Bootstrap {NBOOT} lần theo người, Holm một phía α = 0,025 trên "
           f"{len(kb['so_sanh_chinh'])} so sánh chính. Δ = chỉ số(PG) − chỉ số(đối chứng), ghép cặp trên cùng lượt khám. "
           "`P_delta_duong` = tỉ lệ lần bootstrap PG hơn (KHÔNG phải p-value). `p_mot_phia` = tỉ lệ lần bootstrap Δ ≤ biên.")
    rows = []
    for s in kb["so_sanh_chinh"]:
        a, b = [x.strip() for x in s["so_sanh"].split("vs")]
        nhom = "có ảnh" if s["nhom"] == "co_anh" else "không ảnh"
        r = paired(a, b, nhom, s["chi_so"], s["bien"])
        if r:
            rows.append(dict(ma=s["ma"], so_sanh=s["so_sanh"], nhom=nhom, chi_so=s["chi_so"], loai=s["loai"], bien=s["bien"], **r))
    T = pd.DataFrame(rows)
    if len(T) == len(kb["so_sanh_chinh"]):
        T["dat_theo_KTC"] = np.where(T.loai == "hon", T.ktc_duoi > 0, T.ktc_duoi > T.bien)
        T["dat_sau_Holm"] = holm(T.p_mot_phia.to_numpy(), 0.025) & T.dat_theo_KTC
        R.table(T.drop(columns=["ktc_duoi", "ktc_tren"]), "gia_thuyet_chinh", "Các so sánh chính")
        d = dict(zip(T.ma, T.dat_sau_Holm))
        H = {"H1": d["H1a"] and d["H1b"], "H2": d["H2"], "H3": d["H3a"] and d["H3b"]}
        R.text("## Kết luận\n\n" + "\n".join(f"- **{k}: {'ĐẠT' if v else 'KHÔNG ĐẠT'}** ({kb['ket_luan_gia_thuyet'][k]})" for k, v in H.items()))
        R.text("**Điểm duyệt 2:** " + ("H1 và H2 đều KHÔNG ĐẠT → theo khai báo trước phải DỪNG phát triển mô hình và báo PGS."
                                       if not H["H1"] and not H["H2"] else "chưa rơi vào điều kiện dừng (H1 hoặc H2 đạt)."))
    else:
        R.text("**Chưa đủ dữ liệu để kiểm định**: thiếu dự đoán của một số mô hình (xem cuối báo cáo). Không kết luận.")
        if len(T):
            R.table(T.drop(columns=["ktc_duoi", "ktc_tren"]), "gia_thuyet_chinh_thieu", "So sánh đã tính được (chưa đủ họ Holm)")
    ex = []
    for a, b, nhom, cs in [("PG-global", "C", "có ảnh", "AUC trong lo"), ("PG-global", "L0", "không ảnh", "ROC-AUC gop"),
                           ("PG-global", "T2", "có ảnh", "AUC trong lo"), ("PG-global", "T2", "không ảnh", "ROC-AUC gop"),
                           ("PG", "PG-global", "có ảnh", "AUC trong lo"), ("PG", "PG-global", "không ảnh", "ROC-AUC gop"),
                           ("PG", "L2", "có ảnh", "AUC trong lo"), ("PG", "L2", "không ảnh", "ROC-AUC gop"),
                           ("PG", "X2", "không ảnh", "ROC-AUC gop"), ("PG", "GAT", "không ảnh", "ROC-AUC gop"),
                           ("PG", "B3b", "không ảnh", "ROC-AUC gop"), ("PG", "L0", "không ảnh — Đồng Nai", "ROC-AUC gop"),
                           ("PG", "L0", "không ảnh — tỉnh khác", "ROC-AUC gop")]:
        r = paired(a, b, nhom, cs, 0.0)
        if r:
            ex.append(dict(so_sanh=f"{a} vs {b}", nhom=nhom, chi_so=cs, **{k: v for k, v in r.items() if k not in ("ktc_duoi", "ktc_tren", "p_mot_phia")}))
    if ex:
        R.table(pd.DataFrame(ex), "kham_pha", "So sánh khám phá (không hiệu chỉnh đa so sánh, chỉ tham khảo)")

if MISSING:
    R.text("**Thiếu dự đoán (chưa chạy hoặc chưa chép về):** " + ", ".join(sorted(set(MISSING))))
R.save()
if not args.smoke:
    run_config(R.d.relative_to(RES3).as_posix() + "/bao_cao", __file__,
               data=[f for f in [BL_F, KB if args.buoc == 5 else None] if f is not None and f.exists()], buoc=args.buoc)
print(f"da ghi {R.d / 'BAO_CAO.md'}" + (f" | thieu: {sorted(set(MISSING))}" if MISSING else ""))

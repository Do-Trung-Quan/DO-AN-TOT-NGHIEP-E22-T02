"""Cham tap test KHOA mot lan cho 9 mo hinh da bao cao 07/11 (lenh PGS 11/11: dung thuc nghiem, khoa cau hinh, cham test).

  python v3_test_mot_lan.py khoa   # ghi results_v3/tuan6_test_mot_lan/khoa_cau_hinh.json (ma bam file du doan + ke hoach cham)
  python v3_test_mot_lan.py thu    # chay DUNG code nay tren dev OOF (oof_cal) -> doi chieu voi bao cao 07/11; khong doc test
  python v3_test_mot_lan.py test   # cham test MOT LAN; tu choi chay neu da co ket qua test

Khong huan luyen lai: du doan test (test_cal = trung binh 5 mo hinh fold, da hieu chinh Platt trong fold) da duoc sinh
luc huan luyen 07/11 va chua tung duoc cham. Chi ghi so lieu gop.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_base import load_base_v3, CACHE3, RES3, TAG, run_config  # noqa: E402
from v2_lib import Report, boot_idx, load_core, within_batch_auc, NBOOT, SEED  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
assert MODE in ("khoa", "thu", "test"), "dung: khoa | thu | test"
OUT = RES3 / "tuan6_test_mot_lan"
LOCK = OUT / "khoa_cau_hinh.json"

MODELS = ["C", "L0", "L2", "X0", "X2", "GAT", "B3b", "T2", "PG"]
BASE = ("C", "L0", "L2", "X0", "X2", "GAT", "B3b")
FILES = [f"run_baselines_v3{TAG}.npz", f"run_T2{TAG}.npz", f"run_PG{TAG}.npz", f"selected_T2{TAG}.json",
         f"selected_PG{TAG}.json", f"chon_baselines_v3{TAG}.csv", "p05_folds.npz", "img_v3.npz"]
# so sanh ghep cap (a, b, nhom, chi so). 4 cai dau = H1a, H1b, H3a, H3b cua khai_bao_truoc_v3.json (H2 can PG-shuffle,
# khong thuoc 9 mo hinh khoa -> khong cham). Cac cai sau phuc vu cau hoi shortcut: loi da phuong thuc khi kiem soat lo.
PAIRS = [("PG", "C", "có ảnh", "AUC trong lô"), ("PG", "L0", "không ảnh", "ROC-AUC gộp"),
         ("PG", "T2", "có ảnh", "AUC trong lô"), ("PG", "T2", "không ảnh", "ROC-AUC gộp"),
         ("L2", "L0", "có ảnh", "ROC-AUC gộp"), ("L2", "L0", "có ảnh", "AUC trong lô"),
         ("X2", "X0", "có ảnh", "ROC-AUC gộp"), ("X2", "X0", "có ảnh", "AUC trong lô"),
         ("PG", "L0", "có ảnh", "ROC-AUC gộp"), ("PG", "L0", "có ảnh", "AUC trong lô"),
         ("C", "L0", "có ảnh", "ROC-AUC gộp"), ("C", "L0", "có ảnh", "AUC trong lô")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


if MODE == "khoa":
    assert not LOCK.exists(), f"{LOCK} da ton tai - khong ghi de"
    OUT.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({
        "ngay_khoa": datetime.now().isoformat(timespec="seconds"),
        "can_cu": "Lenh PGS 11/11/2026: khoa cau hinh 9 mo hinh bao cao 07/11, cham test mot lan, khong chay them mo hinh/ablation",
        "mo_hinh": MODELS,
        "du_doan": "test_cal (baselines: test_cal_<m>) = trung binh 5 mo hinh fold, Platt trong fold, sinh ngay 07/11, chua cham",
        "ma_bam_file": {f: sha(CACHE3 / f) for f in FILES},
        "ma_bam_script": sha(__file__),
        "nhom": {"có ảnh": "test & has_img", "không ảnh": "test & ~has_img", "toàn bộ": "test (mo hinh co du doan ca hai nhom)"},
        "chi_so": ["ROC-AUC gộp", "AUC trong lô (chỉ nhóm có ảnh; trung bình theo cỡ lô, bỏ lô một nhãn)", "PR-AUC", "Brier"],
        "ktc": f"95% phân vị, bootstrap {NBOOT} lần theo người (uid2), seed {SEED}",
        "so_sanh_ghep_cap": [dict(zip(("a", "b", "nhom", "chi_so"), p)) for p in PAIRS],
        "do_nhay": "bỏ các lượt bnn=Có & ketqua=2 & có ảnh (định nghĩa g57 trong v3_p045.py), tính lại nhóm có ảnh",
        "quy_tac": "Cham mot lan; khong sua mo hinh, nguong hay script sau khi xem test. Lech nao phai ghi vao lich_su_sua.",
        "lich_su_sua": []}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"da khoa: {LOCK}"); sys.exit()

# ------------------------------------------------------------------ kiem tra khoa (thu tu: thu -> khoa -> test)
DONE = OUT / "BAO_CAO.md"
if MODE == "test":
    L = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [f for f, h in L["ma_bam_file"].items() if sha(CACHE3 / f) != h]
    assert not bad, f"file du doan da thay doi sau khi khoa: {bad}"
    assert sha(__file__) == L["ma_bam_script"], "script da sua sau khi khoa"
    assert not DONE.exists(), "test DA duoc cham - khong cham lai"
else:
    L = {"ngay_khoa": "(chua khoa - chay thu)"}

B = load_base_v3(); core = load_core()
y, hi, uid, batch = B["y"], B["has_img"], np.asarray(B["uid"]), np.asarray(B["batch"])
ROWS = B["test"] if MODE == "test" else B["dev"]
g57 = ((core.bnn == 1) & (core.y == 0) & core.has_img).to_numpy()
BL = np.load(CACHE3 / f"run_baselines_v3{TAG}.npz", allow_pickle=False)
KEY = "test_cal" if MODE == "test" else "oof_cal"


def pred(m):
    p = BL[f"{KEY}_{m}"] if m in BASE else np.load(CACHE3 / f"run_{m}{TAG}.npz", allow_pickle=False)[KEY]
    p = np.asarray(p, float).copy(); p[~ROWS] = np.nan
    return p


P = {m: pred(m) for m in MODELS}
GRP = {"có ảnh": ROWS & hi, "không ảnh": ROWS & ~hi, "toàn bộ": ROWS}


def allm(yy, pp, bb):
    return dict(roc_auc=roc_auc_score(yy, pp), auc_trong_lo=within_batch_auc(yy, pp, bb) if bb is not None else np.nan,
                pr_auc=average_precision_score(yy, pp), brier=brier_score_loss(yy, pp))


def boot_all(yy, pp, bb, uu):
    base = allm(yy, pp, bb); rng = np.random.default_rng(SEED); V = []
    for _ in range(NBOOT):
        i = boot_idx(uu, rng)
        if len(np.unique(yy[i])) < 2: continue
        V.append(allm(yy[i], pp[i], bb[i] if bb is not None else None))
    V = pd.DataFrame(V)
    return {k: (v, *np.nanpercentile(V[k], [2.5, 97.5])) if np.isfinite(v) else (np.nan, np.nan, np.nan) for k, v in base.items()}


def mtable(excl=None):
    rows = []
    for m in MODELS:
        for g, msk in GRP.items():
            k = msk & np.isfinite(P[m])
            if excl is not None: k &= ~excl
            if g == "toàn bộ" and not (k == msk).all(): continue      # mo hinh chi co mot nhom
            if k.sum() < 20 or len(np.unique(y[k])) < 2: continue
            r = boot_all(y[k], P[m][k], batch[k] if g == "có ảnh" else None, uid[k])
            row = dict(mo_hinh=m, nhom=g, n=int(k.sum()), ti_le_benh=float(y[k].mean()), brier_nen=float(y[k].mean() * (1 - y[k].mean())))
            for kk, (v, lo, up) in r.items():
                row[kk] = v; row[kk + "_ktc"] = "" if not np.isfinite(v) else f"[{lo:.3f}–{up:.3f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def paired(a, b, g, cs):
    k = GRP[g] & np.isfinite(P[a]) & np.isfinite(P[b]); yy, A, Bb, bb = y[k], P[a][k], P[b][k], batch[k]
    f = (lambda i: within_batch_auc(yy[i], A[i], bb[i]) - within_batch_auc(yy[i], Bb[i], bb[i])) if cs == "AUC trong lô" else \
        (lambda i: roc_auc_score(yy[i], A[i]) - roc_auc_score(yy[i], Bb[i]))
    d = f(np.arange(k.sum())); rng = np.random.default_rng(SEED); v = []
    for _ in range(NBOOT):
        i = boot_idx(uid[k], rng)
        if len(np.unique(yy[i])) < 2: continue
        v.append(f(i))
    v = np.array([x for x in v if np.isfinite(x)]); lo, up = np.percentile(v, [2.5, 97.5])
    return dict(so_sanh=f"{a} − {b}", nhom=g, chi_so=cs, n=int(k.sum()), delta=d, ktc=f"[{lo:+.3f}; {up:+.3f}]",
                ti_le_boot_a_hon=float((v > 0).mean()))


T = mtable(); D = pd.DataFrame([paired(*p) for p in PAIRS]); S = mtable(excl=g57); S = S[S.nhom == "có ảnh"]
TEN = "tuan6_test_mot_lan" if MODE == "test" else "tuan6_test_mot_lan/thu_tren_dev_OOF"
R = Report(TEN, "Chấm tập test khoá — một lần" if MODE == "test" else "Chạy thử script chấm test trên dev OOF (không phải test)")
R.text(f"Khoá: `{LOCK.name}` (ngày khoá {L['ngay_khoa']}). 9 mô hình báo cáo 07/11, không huấn luyện lại. "
       f"{'Tập test khoá' if MODE == 'test' else 'Dev, dự đoán out-of-fold'}: {int(ROWS.sum())} lượt "
       f"({int((ROWS & hi).sum())} có ảnh, {int((ROWS & ~hi).sum())} không ảnh). KTC 95% bootstrap {NBOOT} lần theo người.")
NOTE = ("*`roc_auc` (ROC-AUC gộp): xác suất mô hình chấm một ca bệnh cao hơn một ca không bệnh, 0,5 = đoán bừa. "
        "`auc_trong_lo`: AUC tính riêng từng lô chụp rồi trung bình theo cỡ lô; loại phần mô hình 'nhận ra lô' thay vì nhận ra bệnh. "
        "`pr_auc`: độ chính xác trung bình khi bắt ca bệnh, so với `ti_le_benh`. `brier`: sai số bình phương của xác suất, "
        "càng thấp càng tốt, so với `brier_nen` (luôn đoán tỉ lệ nền).*")
R.table(T, "chi_so", "Chỉ số theo mô hình và nhóm", note=NOTE)
R.table(D, "so_sanh_ghep_cap", "So sánh ghép cặp (Δ = a − b, cùng lượt khám)",
        note="*4 dòng đầu = H1a, H1b, H3a, H3b của khai báo trước; H2 cần PG-shuffle (không thuộc 9 mô hình khoá) nên không chấm. "
             "`ti_le_boot_a_hon` là tỉ lệ lần bootstrap a hơn b, không phải p-value.*")
R.table(S, "do_nhay_bo_g57", "Phân tích độ nhạy: bỏ các lượt bnn=Có nhưng ketqua=2 (nhóm có ảnh)")
R.save()
run_config(R.d.relative_to(RES3).as_posix() + "/bao_cao", __file__, data=[LOCK] + [CACHE3 / f for f in FILES[:3]], che_do=MODE)
print(f"da ghi {R.d / 'BAO_CAO.md'} | g57 trong nhom: {int((g57 & ROWS).sum())}")

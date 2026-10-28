"""P0.5 ban 3 (PGS 27/10: "Hoan tat P0.5 (ghep nguoi, kiem tra vat fold) roi moi chay tiep cac buoc sau").
Ghep nguoi bi tach ma: khoa long (ho ten chuan hoa + nam sinh + gioi + tinh, union-find voi patient_uid) -> uid2.
Phuong an (a), user chon: GIU NGUYEN test; bo khoi dev cac luot cua nguoi da co trong test; chia lai 5 fold dev theo uid2
(StratifiedGroupKFold, phan tang y x co_anh, seed 42) -> dev2/fold2 (= v2_lib.add_person_merge).
Bao them khoa chat (khoa long + phan xuong) de thay bien dao dong. Kiem tra vat fold truoc / sau.
Test khong mo: luot test chi duoc dem. Khoa ghep chi dung cuc bo, khong luu. Chi ghi so lieu gop.
Dau ra: results_v3/tuan1_phase0/P0_5_ghep_nguoi (+ run_config.json); cache_v3/p05_folds.npz (id -> uid2/dev2/fold2, PII).
"""
import numpy as np, pandas as pd
from v3_lib import *
from v3_lib import CACHE3, run_config
from v2_lib import _norm_name, add_person_merge

OUT = "tuan1_phase0/P0_5_ghep_nguoi"
core = load_core()
# tinh lai tu dau (khong tin cot uid2/dev2/fold2 trong cache) va doi chieu
fresh = add_person_merge(core[["id", "y", "uid", "fold", "split", "has_img"]].copy())
for c in ("uid2", "dev2", "fold2"):
    assert (pd.factorize(fresh[c])[0] == pd.factorize(core[c])[0]).all() if c == "uid2" else (fresh[c].to_numpy() == core[c].to_numpy()).all(), c
core = core.drop(columns=["uid2", "dev2", "fold2"]).join(fresh[["uid2", "dev2", "fold2"]])

xl = pd.read_excel(XLSX, usecols=["id", "hoten", "namsinh", "gioitinh", "tinh", "pxuong"]).set_index("id").loc[core.id]
loose = (xl.hoten.map(_norm_name) + "|" + xl.namsinh.astype(str) + "|" + xl.gioitinh.astype(str) + "|" + xl.tinh.astype(str)).to_numpy()
strict = loose + "|" + xl.pxuong.astype(str).map(_norm_name).to_numpy()
del xl


def merge(key):
    """union-find patient_uid + khoa -> ma nhom."""
    par = {}
    def f(a):
        while par.setdefault(a, a) != a: par[a] = par[par[a]]; a = par[a]
        return a
    for u, k in zip(core.uid, key): par[f(("u", u))] = f(("k", k))
    return pd.factorize(pd.Series([str(f(("u", u))) for u in core.uid]))[0]


core["uid3"] = merge(strict)
fold_goc = np.where(core.split == "test", -1, core.fold)           # fold goc, test = -1
fold_moi = np.where(core.split == "test", -1, np.where(core.dev2, core.fold2, -9))   # -9 = luot dev bi bo


def vat(g, f, keep):
    """luot (trong keep) thuoc nhom g co > 1 gia tri fold (ke ca dev/test)."""
    d = pd.DataFrame(dict(g=g, f=f))[keep]
    nf = d.groupby("g").f.transform("nunique") > 1
    return int(nf.sum()), int(d.g[nf].nunique()), int((d.groupby("g").f.transform(lambda s: (s == -1).any() and (s != -1).any())).sum())


allr = np.ones(len(core), bool); kept = fold_moi != -9; img = core.has_img.to_numpy()
rows = []
for kn, g in (("Khóa lỏng (họ tên + năm sinh + giới + tỉnh) — DÙNG ĐỂ CHIA", core.uid2), ("Khóa chặt (khóa lỏng + phân xưởng) — chỉ để so", core.uid3)):
    for tn, f, k in (("Trước (fold gốc)", fold_goc, allr), ("Sau (fold mới, đã bỏ lượt dev trùng test)", fold_moi, kept)):
        n, ng, nt = vat(g.to_numpy(), f, k)
        ni, _, _ = vat(g.to_numpy(), f, k & img)
        rows.append(dict(khoa=kn, thoi_diem=tn, luot_xet=int(k.sum()), luot_vat_fold=n, pct=round(100 * n / k.sum(), 2),
                         nhom_vat=ng, luot_vat_dev_test=nt, luot_co_anh_vat=ni, pct_co_anh=round(100 * ni / (k & img).sum(), 2)))
T1 = pd.DataFrame(rows)

n_uid, n2, n3 = core.uid.nunique(), core.uid2.nunique(), core.uid3.nunique()
drop = (core.split == "dev") & ~core.dev2
T2 = pd.DataFrame([
    dict(muc="Lượt khám (dev / test)", gia_tri=f"{len(core)} ({int((core.split=='dev').sum())} / {int((core.split=='test').sum())})"),
    dict(muc="Người theo patient_uid gốc", gia_tri=n_uid),
    dict(muc="Người sau ghép khóa lỏng (uid2) / khóa chặt", gia_tri=f"{n2} / {n3}"),
    dict(muc="Nhóm gộp ≥ 2 patient_uid — khóa lỏng / khóa chặt",
         gia_tri=f"{int((core.groupby('uid2').uid.nunique() > 1).sum())} / {int((core.groupby('uid3').uid.nunique() > 1).sum())}"),
    dict(muc="Lượt dev bị bỏ (người đã có trong test)", gia_tri=f"{int(drop.sum())} (có ảnh: {int((drop & core.has_img).sum())}; dương: {int(core.y[drop].sum())})"),
    dict(muc="Tập test", gia_tri="giữ nguyên, không mở (chỉ đếm)"),
    dict(muc="dev mới (dev2)", gia_tri=f"{int(core.dev2.sum())} lượt, {int(core.y[core.dev2].sum())} dương, {core.uid2[core.dev2].nunique()} người"),
    dict(muc="dev mới có ảnh (tập dùng cho P0.1 / việc (1))",
         gia_tri=f"{int((core.dev2 & core.has_img).sum())} lượt, {int(core.y[core.dev2 & core.has_img].sum())} dương, "
                 f"{core.uid2[core.dev2 & core.has_img].nunique()} người, {core.batch[core.dev2 & core.has_img].nunique()} lô"),
])
d2 = core[core.dev2]
T3 = d2.groupby("fold2").agg(luot=("y", "size"), duong=("y", "sum"), co_anh=("has_img", "sum"),
                             duong_co_anh=("y", lambda s: int(s[d2.loc[s.index, "has_img"]].sum())), nguoi=("uid2", "nunique")).reset_index()
T3["ti_le_duong"] = (T3.duong / T3.luot).round(3); T3["ti_le_duong_co_anh"] = (T3.duong_co_anh / T3.co_anh).round(3)
g57 = (core.bnn == 1) & (core.y == 0) & core.has_img
T4 = pd.DataFrame([dict(muc="Nhóm bnn=Có & ketqua=2 có ảnh: lượt / người theo patient_uid / người theo uid2",
                        gia_tri=f"{int(g57.sum())} / {core.uid[g57].nunique()} / {core.uid2[g57].nunique()}"),
                   dict(muc="  lượt của nhóm này trong dev mới có ảnh", gia_tri=int((g57 & core.dev2).sum()))])

R = Report(OUT, "P0.5 — Ghép người bị tách mã và chia lại fold")
R.text("Khóa ghép lỏng: họ tên chuẩn hóa (bỏ dấu, chữ thường) + năm sinh + giới + tỉnh, gộp với `patient_uid` bằng union-find "
       "(hai lượt chung mã hoặc chung khóa → cùng người). Khóa chặt thêm phân xưởng, chỉ để thấy biên dao động. Khóa chỉ dùng cục bộ, không lưu. "
       "Luật PG_update §2: lượt thuộc nhóm vắt fold hoặc vắt dev/test ≥ 1% → chia lại fold. "
       "Phương án đã chọn: **giữ nguyên test**, bỏ khỏi dev các lượt của người đã có trong test, chia lại 5 fold dev theo người đã ghép "
       f"(StratifiedGroupKFold, phân tầng theo nhãn × có ảnh, seed {SEED}).")
R.table(T1, "1_vat_fold_truoc_sau", "Kiểm tra vắt fold — trước và sau khi chia lại",
        note="*`luot_vat_fold`: lượt thuộc nhóm \"cùng người\" có lượt nằm ở ≥ 2 fold khác nhau (test tính là một fold). "
             "`luot_vat_dev_test`: phần trong đó vắt giữa dev và test. `luot_co_anh_vat`: chỉ tính lượt có ảnh. "
             "Khóa lỏng gộp nhiều hơn khóa chặt (có thể gộp nhầm người trùng tên), nên con số theo khóa lỏng là mức trên.*")
R.table(T2, "2_tom_tat", "Tóm tắt")
R.table(T3, "3_fold_moi", "Cân bằng 5 fold mới (dev2)")
R.table(T4, "4_nhom_55_nguoi", "Nhóm bnn=Có & ketqua=2 (dùng cho phân tích độ nhạy) sau ghép người")
R.save()
np.savez(CACHE3 / "p05_folds.npz", id=core.id.to_numpy(), uid2=core.uid2.to_numpy(), dev2=core.dev2.to_numpy(), fold2=core.fold2.to_numpy())
run_config(OUT, __file__, data=[FUS / "fusion_dataset_ketqua_frozen.npz", XLSX],
           khoa_long="ho ten chuan hoa + namsinh + gioitinh + tinh", khoa_chat="khoa long + pxuong",
           phuong_an="(a) giu test, bo luot dev trung nguoi test, chia lai 5 fold dev",
           chia_fold="StratifiedGroupKFold(5, shuffle=True, random_state=SEED), strat = y*2 + has_img, group = uid2")
pd.set_option("display.width", 250)
print(T1.to_string(index=False)); print(T2.to_string(index=False)); print(T3.to_string(index=False)); print(T4.to_string(index=False))

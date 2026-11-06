"""Tao lai cache_pg/clin_raw.pkl (bang lam sang tho cho pag_lib.load_base) tu Main_data_Merged_1835_Images.xlsx.

File goc (Minh tao) khong co trong repo va khong co script sinh ra. Dung lai theo dung cach pag_lib doc no:
  - thu tu hang = core.id (cache_v2/core.pkl)
  - cot = moi bien trong pag_lib.GROUPS + `tinh` + `nam` (load_base drop `nam`); bmi va bnn do load_base tu tinh
  - fev1_fvc = fev1tt / fvctt (PG_update: dung gia tri tuyet doi va ti so tinh lai)
Khong co ho ten / sdt / sobh (PII). Doi chieu voi results_v2/tuan2_du_lieu_do_thi/bien_vao_node.csv: so cot ma hoa
va % thieu tung nhom phai khop.

  python PAG/scripts_v3/v3_make_clin_raw.py [--xlsx data/Main_data_Merged_1835_Images.xlsx]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_lib import MERG, RES2, load_core  # noqa: E402
from pag_lib import GROUPS, ClinEncoder, GNAMES  # noqa: E402

COMPUTED = {"bmi", "bnn", "fev1_fvc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=MERG / "data" / "Main_data_Merged_1835_Images.xlsx")
    ap.add_argument("--out", type=Path, default=MERG / "cache_pg" / "clin_raw.pkl")
    args = ap.parse_args()

    core = load_core()
    want = [c for g in GROUPS.values() for c in g if c not in COMPUTED]
    cols = list(dict.fromkeys(want + ["fvctt", "fev1tt", "tinh", "nam"]))
    xl = pd.read_excel(args.xlsx)
    miss = [c for c in cols if c not in xl.columns]
    if miss:
        raise SystemExit(f"xlsx thieu cot: {miss}")
    assert xl["id"].is_unique
    raw = xl.set_index("id").loc[core.id, cols].reset_index(drop=True)
    f1, fv = pd.to_numeric(raw["fev1tt"], errors="coerce"), pd.to_numeric(raw["fvctt"], errors="coerce")
    raw["fev1_fvc"] = (f1 / fv).where(fv > 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(args.out)
    print(f"da ghi {args.out}: {raw.shape[0]} hang x {raw.shape[1]} cot (gom nam; load_base bo nam va them bmi, bnn)")

    # ---- doi chieu voi bang bien -> node ban 2 (ma hoa fit tren toan dev, chi de mo ta)
    raw2 = raw.drop(columns=["nam"]).copy()
    h, w = pd.to_numeric(raw2["cao"], errors="coerce") / 100, pd.to_numeric(raw2["can"], errors="coerce")
    raw2["bmi"] = (w / h ** 2).where((h > 1.2) & (h < 2.1) & (w > 25) & (w < 150))
    raw2["bnn"] = core["bnn"].to_numpy()
    dev = core.dev2.to_numpy()
    ce = ClinEncoder().fit(raw2, dev)
    out, names, _ = ce.transform(raw2)
    ref = pd.read_csv(RES2 / "tuan2_du_lieu_do_thi" / "bien_vao_node.csv").set_index("node")
    ok = True
    print("\nnhom | so cot ma hoa (moi / ban 2) | % thieu tb moi bien (moi / ban 2)")
    for g in GNAMES + ["Bối cảnh"]:
        cs = GROUPS[g] if g in GROUPS else ["tinh"]
        pct = float(raw2.loc[dev, cs].isna().mean().mean() * 100)
        r = ref.loc[g]
        same = out[g].shape[1] == int(r.so_cot_ma_hoa) and abs(pct - r.pct_thieu_tb_moi_bien) < 0.05
        ok &= same
        print(f"  {g}: {out[g].shape[1]} / {int(r.so_cot_ma_hoa)} | {pct:.2f} / {r.pct_thieu_tb_moi_bien:.2f}  {'OK' if same else 'LECH'}")
    print("\nKET LUAN:", "khop bang ban 2" if ok else "LECH bang ban 2 — kiem tra truoc khi dung")


if __name__ == "__main__":
    main()

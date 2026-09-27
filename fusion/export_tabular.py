"""
================================================================================
XUAT DAC TRUNG LAM SANG DANG BANG CHO FUSION
================================================================================
VI SAO KHONG DUNG VECTOR 32 CHIEU CUA NHANH TS NUA

  Nhanh ts sinh vector bang lop `fusion_dense` cua 5 model fold. Co hai cach lay,
  ca hai deu hong:
    - Trung binh 5 model cho moi node  -> cung mot khong gian, NHUNG voi node dev
      thi 4/5 model da hoc chinh no => vector mang san thong tin nhan (ro ri).
      Do duoc: MLP tren vector nay dat ROC 0,752 tren dev trong khi chinh nhanh ts
      chi dat 0,641.
    - Lay out-of-fold (node fold k lay tu model k) -> sach, NHUNG 5 fold nam o
      5 khong gian vector khac nhau. Do duoc: probe tuyen tinh ROC 0,487 —
      gan nhu vo dung. Dung loi da gap o bo anh `crossfit`.

  => Dung thang BANG THO da ma hoa (139 cot). Khong qua mo hinh nao nen khong the
     ro ri, moi node cung mot khong gian, va do duoc manh hon: ROC 0,663 so voi
     0,641 cua vector 32 chieu.

  Diem du doan cua nhanh ts van duoc giu lai lam MOT dac trung phu (1 chieu, sach,
  cung y nghia o moi fold) — xem fusion/baselines.py.

DAU VAO   output/Main_data_Processed_Encoded.xlsx   (chi co o may local, khong commit)
DAU RA    fusion/output/tabular_ketqua.npz          (commit duoc: khong ten, khong sdt)

CHAY      python fusion/export_tabular.py
================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output" / "Main_data_Processed_Encoded.xlsx"
META = ROOT / "output" / "ketqua" / "fusion_node_meta.parquet"
OUT = ROOT / "fusion" / "output" / "tabular_ketqua.npz"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PII = {"hoten", "ten", "sdt", "id", "sobh", "file_name", "bnncuthe"}
LEAK = {"bnn", "ketqua", "chatluongphim", "matdotonthuong", "kichthuoctt",
        "tonthuongkhac", "vungtonthuong"}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Thieu {SRC} — file nay chi co o may local (khong commit).")
    df = pd.read_excel(SRC)
    meta = pd.read_parquet(META)

    drop = [c for c in df.columns if str(c).strip().lower() in (PII | LEAK)]
    X = df.drop(columns=drop)
    print(f"Da loai {len(drop)} cot (dinh danh + doc phim): {drop}")

    assert len(X) == len(meta) == 8030, f"Lech so hang: {len(X)} vs {len(meta)}"
    assert X.select_dtypes(exclude="number").empty, "Con cot khong phai so"
    arr = X.to_numpy(np.float32)
    assert np.isfinite(arr).all(), "Co NaN/Inf trong bang"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X_tab=arr, cols=np.array(list(X.columns), dtype="U32"),
                        id=meta["id"].to_numpy(np.int64))
    print(f"Da ghi -> {OUT.relative_to(ROOT).as_posix()}  {arr.shape}  "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")
    print("Khoa `id` lay tu fusion_node_meta nen thu tu hang khop tuyet doi voi cac npz khac.")


if __name__ == "__main__":
    main()

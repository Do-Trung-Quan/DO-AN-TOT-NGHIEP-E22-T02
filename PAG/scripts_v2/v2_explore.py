"""Tham do cot Excel cho Phase 0 ban 2 — CHI in ten cot + thong ke gop, khong in gia tri ca nhan."""
import numpy as np, pandas as pd
from pg_lib import XLSX, FUS, load_base

xl = pd.read_excel(XLSX)
PII = {"hoten", "sdt", "sobh", "bnncuthe", "diachi"}
print(len(xl), "dong |", len(xl.columns), "cot")
print([c for c in xl.columns if c not in PII])
for c in ["ketqua", "bnn", "chatluongphim", "matdotonthuong", "kichthuoctt", "tonthuongkhac", "tinh", "nam"]:
    if c in xl.columns:
        print(c, xl[c].astype(str).value_counts(dropna=False).head(8).to_dict())
if {"ketqua", "bnn"} <= set(xl.columns):
    print(pd.crosstab(xl["bnn"].astype(str), xl["ketqua"].astype(str)))

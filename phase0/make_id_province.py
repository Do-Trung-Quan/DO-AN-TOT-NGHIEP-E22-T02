"""Tao bang id -> tinh (khong chua ho ten/sdt) de dua len Colab cho probe lo trong cung tinh.

Chay LOCAL (doc xlsx co PII), chi ghi hai cot id, tinh.
"""
from __future__ import annotations

import pandas as pd

from phase0.common import OUT, read_merged_xlsx

df = read_merged_xlsx(usecols=["id", "tinh"])
assert df["id"].is_unique, "id khong duy nhat trong xlsx"
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "id_province.csv", index=False)
print("da ghi id_province.csv:", len(df), "dong;", df["tinh"].nunique(), "tinh;",
      df["tinh"].value_counts().to_dict())

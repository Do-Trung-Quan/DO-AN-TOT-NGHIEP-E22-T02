"""P0.5(a) — gop ket qua chay lai B3b/GAT tren do thi da sua voi k chon chi tren dev.

Moi fold bao cao f lay du doan tu lan chay co k = k_f (k chon bang nhan cua 4 fold con lai, xem
phase0/p05_select_k.py). Ban "toan dev" (mot k chung) in kem de doi chieu.

Cau truc thu muc chay (tao boi colab/phase0_p05_graph.ipynb):
  {runs}/{source}_k{K}/phase3_oof_{source}.npz            (fusion/baselines.py, ca 8 mo hinh)
  {runs}/{source}_k{K}/phase4_oof_gat_{source}.npz        (fusion/gnn.py)
  {runs}/{source}_k{K}/phase4_oof_gat_nograph_{source}.npz (tuy chon)

Moi so sanh ghep cap tren cung benh nhan: delta, KTC 95% bootstrap theo benh nhan, P(delta>0).
KHONG goi P(delta>0) la p-value. Chi ket luan "khac" khi KTC khong cat 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion.baselines import paired_gap, score_block
from phase0.common import OUT, Nodes, load_config, write_json

MODELS = ["B0", "B1", "B2", "B3", "B3b", "L1", "L2", "L3b"]
# (a, b, nhom, mo ta) — khai bao truoc; moi so sanh khac se ghi la kham pha
PAIRS = [
    ("B3b", "B0", "khong_anh", "hang xom (diem) co giup nhom khong anh khong"),
    ("L3b", "L2", "khong_anh", "diem hang xom trong hoi quy logistic"),
    ("GAT", "GAT_nograph", "khong_anh", "dong gop thuan cua truyen tin (khong anh)"),
    ("GAT", "B3b", "khong_anh", "GAT so voi gop trung binh co dinh"),
    ("GAT", "L2", "khong_anh", "GAT so voi hoi quy logistic"),
    ("GAT", "B2", "co_anh", "GAT so voi noi chay (co anh)"),
    ("GAT", "L1", "co_anh", "GAT so voi hoi quy logistic chi anh (co anh)"),
]


def load_run(runs: Path, source: str, k: int):
    d = runs / f"{source}_k{k}"
    out = {}
    p3 = d / f"phase3_oof_{source}.npz"
    if p3.exists():
        z = np.load(p3)
        out.update({m: z[m] for m in z.files})
    for key, fn in (("GAT", f"phase4_oof_gat_{source}.npz"), ("GAT_nograph", f"phase4_oof_gat_nograph_{source}.npz")):
        f = d / fn
        if f.exists():
            out[key] = np.load(f)["GAT"]
    return out


def assemble(nd: Nodes, runs: Path, source: str, sel_by_fold: dict[int, int]):
    cache = {}
    def get(k):
        if k not in cache:
            cache[k] = load_run(runs, source, k)
        return cache[k]
    names = set(MODELS) | {"GAT", "GAT_nograph"}
    out = {}
    for name in names:
        arr = np.zeros(len(nd.y))
        ok = True
        for f, k in sel_by_fold.items():
            r = get(k)
            if name not in r:
                ok = False
                break
            m = nd.dev & (nd.fold == f)
            arr[m] = r[name][m]
        if ok:
            out[name] = arr
    return out


def evaluate(nd: Nodes, oof: dict, n_boot: int):
    groups = {"toan_dev": nd.dev, "co_anh": nd.dev & nd.has, "khong_anh": nd.dev & ~nd.has}
    res = {}
    for name, s in oof.items():
        res[name] = {}
        for g, m in groups.items():
            if name == "B1" and g != "co_anh":
                continue
            res[name][g] = score_block(nd.y[m], s[m], nd.uid[m])
    pairs = []
    for a, b, g, desc in PAIRS:
        if a in oof and b in oof:
            m = groups[g]
            gap, lo, hi, p = paired_gap(nd.y[m], oof[a][m], oof[b][m], nd.uid[m], n=n_boot)
            pairs.append({"a": a, "b": b, "nhom": g, "mo_ta": desc, "delta": gap, "ci": [lo, hi],
                          "p_pos": p, "ktc_khong_cat_0": bool(lo > 0 or hi < 0)})
    return res, pairs


def md(title, res, pairs) -> str:
    L = [f"## {title}", "", "| Mô hình | Có ảnh | Không ảnh | Toàn dev |", "|---|---|---|---|"]
    def cell(r):
        return "—" if r is None else f"{r['roc']:.3f} [{r['roc_ci'][0]:.3f}; {r['roc_ci'][1]:.3f}]"
    for name, r in res.items():
        L.append(f"| {name} | {cell(r.get('co_anh'))} | {cell(r.get('khong_anh'))} | {cell(r.get('toan_dev'))} |")
    L += ["", "| So sánh | Nhóm | Δ ROC | KTC 95% | P(Δ>0) | KTC không cắt 0 |", "|---|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {p['a']} − {p['b']} | {p['nhom']} | {p['delta']:+.4f} | [{p['ci'][0]:+.4f}; {p['ci'][1]:+.4f}] | "
                 f"{p['p_pos']:.0%} | {'có' if p['ktc_khong_cat_0'] else 'không'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=OUT / "runs")
    ap.add_argument("--sources", nargs="+", default=["control", "frozen"])
    ap.add_argument("--selection", type=Path, default=OUT / "p05_k_selection.json")
    args = ap.parse_args()
    cfg = load_config()
    sel = json.loads(args.selection.read_text(encoding="utf-8"))
    by_fold = {int(f): int(k) for f, k in sel["k_chon_theo_fold"].items()}
    k_dev = int(sel["k_chon_toan_dev"])
    report, text = {"config_hash": cfg["_hash"], "k_theo_fold": by_fold, "k_toan_dev": k_dev}, []
    for source in args.sources:
        nd = Nodes(source)
        for label, mapping in (("theo-fold (ban chinh)", by_fold), (f"toan-dev k={k_dev} (doi chieu)", {f: k_dev for f in range(5)})):
            oof = assemble(nd, args.runs, source, mapping)
            if not oof:
                print(f"[{source}/{label}] thieu ket qua chay — bo qua")
                continue
            res, pairs = evaluate(nd, oof, cfg["n_boot"])
            report[f"{source}|{label}"] = {"ket_qua": res, "so_sanh": pairs}
            text.append(md(f"Nguồn {source} — chọn k {label}", res, pairs))
            print(f"[{source}/{label}] xong: {sorted(oof)}")
    write_json(OUT / "p05_rerun_gat_b3b.json", report)
    (OUT / "p05_rerun_gat_b3b.md").write_text("# P0.5(a) — chạy lại GAT và B3b\n\n" + "\n".join(text), encoding="utf-8")
    print("da ghi p05_rerun_gat_b3b.json|md")


if __name__ == "__main__":
    main()

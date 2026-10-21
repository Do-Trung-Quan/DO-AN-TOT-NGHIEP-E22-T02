"""P0.1 — tong hop bao cao gui thay (PG_update ban 3, muc 2).

Doc phase0/output/p01_{tag}_{source}.json (+ p01_oof_*.npz, p01_qa_tom_tat.json) va ghi
phase0/output/p01_bao_cao_ban3.md:
  - QA (tu dong + truc quan, so anh zone_missing)
  - 4 chi so cho moi bien the x nguon: probe lo, AUC trong lo cua L1, delta (lo + anh vs lo), LOBO
  - so voi dac trung cu (A_goc): chenh ghep cap AUC trong lo va AUC gop, bootstrap theo benh nhan
  - doi chieu bang quyet dinh (nguon co AUC trong lo THAP HON, de xuat cho PGS duyet)
Day chi la doi chieu bang; quyet dinh thuoc ve PGS.

  python -m phase0.p01_decision
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion.baselines import paired_gap
from phase0.common import OUT, Nodes, boot_lot_auc, decision_v3, fmt_ci, load_config, weighted_lot_auc

TEN = {"A_goc": "đặc trưng cũ (ảnh chưa che chữ, CenterCrop)",
       "V1": "V1 — cắt theo hộp bao phổi + che chữ",
       "V2": "V2 — V1 + xóa pixel ngoài mask phổi"}


def ci_str(d, ci, nd=3):
    return f"{d:+.{nd}f} [{ci[0]:+.{nd}f}; {ci[1]:+.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["A_goc", "V1", "V2"])
    ap.add_argument("--sources", nargs="+", default=["control", "frozen"])
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--dir", type=Path, default=OUT, help="thu muc chua p01_*.json va ghi bao cao")
    args = ap.parse_args()
    cfg = load_config()
    n_boot = args.n_boot or cfg["n_boot"]

    res = {}
    for tag in args.tags:
        for s in args.sources:
            p = args.dir / f"p01_{tag}_{s}.json"
            if p.exists():
                res[(tag, s)] = json.loads(p.read_text(encoding="utf-8"))
            else:
                print(f"[thieu] {p.name}")
    if not res:
        raise SystemExit("chua co ket qua p01_*.json nao")

    L = ["# P0.1 — Nhiễu theo lô chụp, chạy lại trên ảnh đã xử lý (bản 3)", "",
         f"Config `{cfg['_hash']}` · chỉ dev có ảnh · bootstrap {n_boot} lần theo bệnh nhân · "
         "file fold hiện có (mục 3).", "",
         "Ghi chú P0.5: ghép nới lỏng (họ tên + năm sinh + giới + tỉnh) cho 1.366/8.030 lượt khám (17%) nằm trong "
         "nhóm vắt qua fold hoặc dev/test, vượt ngưỡng 1% → sẽ chia lại fold theo khóa ghép nới lỏng sau khi thầy "
         "duyệt P0.1, rồi chạy lại các số dưới đây trên fold mới (chỉ phần CPU, không trích lại ảnh).", ""]

    # ---------------- QA
    qp = args.dir / "p01_qa_tom_tat.json"
    if qp.exists():
        q = json.loads(qp.read_text(encoding="utf-8"))
        a, t = q["qa_tu_dong"], q["qa_truc_quan"]
        L += ["## 1. Xử lý ảnh và QA", "",
              "Ảnh gốc → ảnh xám, kiểm tra đảo âm → mask phổi torchxrayvision (giữ 2 thành phần lớn nhất) → cắt theo "
              "hộp bao hai phổi (lề 5%), pad vuông, resize 448, không CenterCrop → OCR chỉ phát hiện trên vùng cắt, "
              "tô đen hộp chữ → V1 (chỉ cắt), V2 (cắt + xóa ngoài mask).", "",
              f"QA tự động trên {a['so_anh']} ảnh:", "",
              "| Cờ | Số ảnh |", "|---|---|"]
        for k, v in a["so_anh_theo_co"].items():
            L.append(f"| {k[3:]} | {v} |")
        r = t["ngau_nhien"]
        f = t["anh_bi_co"]
        L += ["", f"QA trực quan: {r['n']} ảnh ngẫu nhiên phân tầng lô × nhãn — đạt {r['dat']}, một phần "
              f"{r['mot_phan']}, hỏng {r['hong']} → **tỉ lệ đạt {r['ty_le_dat']:.0%}** (ngưỡng {r['nguong']:.0%}: "
              f"{'ĐẠT' if r['dat_cong'] else 'KHÔNG ĐẠT'}). {f['n']} ảnh bị cờ — đạt {f['dat']}, một phần "
              f"{f['mot_phan']}, hỏng {f['hong']}.", "",
              f"Ảnh hỏng: {q['zone_missing']['n']} ảnh dùng vector toàn ảnh (đã che chữ) và gắn cờ `zone_missing`; "
              "không loại ảnh nào.", ""]
    else:
        L += ["## 1. Xử lý ảnh và QA", "", "(chưa có p01_qa_tom_tat.json)", ""]

    # ---------------- 4 chi so
    L += ["## 2. Bốn chỉ số", "",
          "| Đặc trưng | Nguồn | Probe lô: accuracy (lớp đa số) | AUC trong lô của L1 | Δ AUC (lô + ảnh so với lô) | "
          "LOBO: AUC gộp | LOBO: AUC trong lô | AUC gộp L1 |",
          "|---|---|---|---|---|---|---|---|"]
    for (tag, s), o in res.items():
        p, a, g, b = o["probe_lo"], o["auc_trong_lo"]["chinh"], o["gia_tri_gia_tang"]["delta_auc_gop"], o["leave_batch_out"]
        lobo_gop = fmt_ci(b["auc_gop"]["auc"], b["auc_gop"]["ci"]) if "auc_gop" in b else "—"
        L.append(f"| {tag} | {s} | {fmt_ci(p['acc'], p['acc_ci'], 2)} ({p['lop_da_so']:.2f}) | "
                 f"{fmt_ci(a['auc'], a['ci'])} | {ci_str(g['delta'], g['ci'])} | {lobo_gop} | "
                 f"{fmt_ci(b['auc_trong_lo']['auc'], b['auc_trong_lo']['ci'])} | {o['L1_gop']['roc']:.3f} |")
    L += ["", "AUC trong lô: trung bình có trọng số (theo cỡ lô) của AUC trong từng lô có đủ hai nhãn. "
          "Δ: AUC gộp của LR(lô + ảnh) trừ LR(lô), KTC 95% bootstrap theo bệnh nhân. "
          "LOBO: bỏ hẳn từng lô khỏi huấn luyện rồi chấm trên lô đó.", ""]

    # ---------------- so voi dac trung cu
    if any(t == "A_goc" for t, _ in res) and any(t != "A_goc" for t, _ in res):
        L += ["## 3. So với đặc trưng cũ (ghép cặp trên cùng bệnh nhân)", "",
              "| So sánh | Nguồn | Δ AUC trong lô [KTC 95%] | P(Δ>0) | Δ AUC gộp [KTC 95%] | P(Δ>0) |",
              "|---|---|---|---|---|---|"]
        for s in args.sources:
            if ("A_goc", s) not in res:
                continue
            nd = Nodes(s)
            m = nd.m
            ym, um, lm = nd.y[m], nd.uid[m], nd.lot[m]
            base = np.load(args.dir / f"p01_oof_A_goc_{s}.npz")
            assert (base["id"] == nd.id).all()
            for tag in args.tags:
                if tag == "A_goc" or (tag, s) not in res:
                    continue
                z = np.load(args.dir / f"p01_oof_{tag}_{s}.npz")
                assert (z["id"] == nd.id).all()
                sa, sb = z["L1"][m], base["L1"][m]
                _, dci = boot_lot_auc(ym, {"v": sa, "a": sb}, lm, um, n_boot, cfg["seed"], pairs=[("v", "a")])
                lo, hi, pp = dci[("v", "a")]
                dlot = weighted_lot_auc(ym, sa, lm) - weighted_lot_auc(ym, sb, lm)
                g = paired_gap(ym, sa, sb, um, n=n_boot)
                L.append(f"| {tag} − A_goc | {s} | {ci_str(dlot, (lo, hi))} | {pp:.0%} | "
                         f"{ci_str(g[0], (g[1], g[2]))} | {g[3]:.0%} |")
        L += ["", "P(Δ>0) là tỉ lệ lần bootstrap có Δ dương, không phải p-value.", ""]

    # ---------------- bang quyet dinh
    L += ["## 4. Đối chiếu bảng quyết định của thầy", "",
          "Quy tắc đề xuất (chờ thầy duyệt): với mỗi biến thể, dùng nguồn có AUC trong lô thấp hơn.", "",
          "| Đặc trưng | Nguồn dùng | AUC trong lô | Cận dưới KTC của Δ | Hạng mục theo bảng |", "|---|---|---|---|---|"]
    for tag in args.tags:
        rows = {s: res[(tag, s)] for s in args.sources if (tag, s) in res}
        if not rows:
            continue
        s = min(rows, key=lambda k: rows[k]["auc_trong_lo"]["chinh"]["auc"])
        auc = rows[s]["auc_trong_lo"]["chinh"]["auc"]
        dlo = rows[s]["gia_tri_gia_tang"]["delta_auc_gop"]["ci"][0]
        L.append(f"| {TEN.get(tag, tag)} | {s} | {auc:.3f} | {dlo:+.3f} | **{decision_v3(auc, dlo)}** |")
    L += ["", "Bảng: AUC trong lô ≥ 0,70 và KTC của Δ nằm trọn trên 0 → tiếp tục; 0,60–0,70 → báo cáo PGS "
          "trước khi làm tiếp; < 0,60 → dừng nhánh ảnh. Đây chỉ là đối chiếu; quyết định thuộc về thầy."]

    path = args.dir / "p01_bao_cao_ban3.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nghi {path}")


if __name__ == "__main__":
    main()

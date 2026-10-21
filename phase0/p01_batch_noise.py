"""P0.1 — Nhieu theo lo chup: 4 chi so (PG_update ban 3, muc 2).

Chay cho MOT bien the dac trung va MOT nguon anh (control | frozen):
  A_goc = embedding cu (anh chua che chu, Resize512 -> CenterCrop448), mac dinh khi khong co --features
  V1    = anh cat theo hop bao phoi + che chu        (--features image_features_{src}_V1.parquet)
  V2    = V1 + xoa pixel ngoai mask phoi              (--features image_features_{src}_V2.parquet)
Anh cham "hong" o QA: --zone-missing p01_zone_missing.csv --fallback image_features_{src}_V0.parquet
(dung vector toan anh, gan co, khong loai anh nao).

  1. Probe lo: du doan lo (18 lop) tu embedding, so voi ty le lop da so + nhan lo bi hoan vi
  2. AUC trong lo cua L1: AUC rieng tung lo co du hai nhan, trung binh co trong so theo co lo
  3. Gia tri gia tang cua anh: delta AUC giua LR(lo) va LR(lo + anh)
  4. Do nhay leave-batch-out: bo han tung lo (loai benh nhan co dong thuoc lo do khoi train)

Chi dung dev co anh. Ma lo chi la chan doan va phep chia nhom khi danh gia, khong phai
dac trung cua bat ky mo hinh nao duoc bao cao.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusion.baselines import paired_gap, score_block, sk_oof
from phase0.common import (OUT, ROOT, Nodes, assert_no_test, boot_lot_auc, cluster_boot_indices,
                           decision_v3, fmt_ci, lib_versions, load_config, lot_table, write_json)


def make_lr(C=0.1, class_weight="balanced", max_iter=5000):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter))


# ------------------------------------------------------------------ buoc 1
def probe_lot(X, lot, fold, seed=0):
    """Hoi quy logistic da lop du doan lo, CV theo fold_id (cung fold voi L1)."""
    classes = np.unique(lot)
    cidx = {c: i for i, c in enumerate(classes)}
    yy = np.array([cidx[c] for c in lot])
    pred = np.zeros(len(lot), int)
    proba = np.zeros((len(lot), len(classes)))
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        full = np.zeros((va.sum(), len(classes)))
        if len(np.unique(yy[tr])) < 2:     # tap train chi con mot lo -> du doan lo do
            full[:, np.unique(yy[tr])[0]] = 1.0
        else:
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            mdl.fit(X[tr], yy[tr])
            full[:, mdl[-1].classes_] = mdl.predict_proba(X[va])
        proba[va] = full
        pred[va] = full.argmax(1)
    return yy, pred, proba


def probe_metrics(yy, pred, proba):
    aucs = []
    for c in range(proba.shape[1]):
        pos = yy == c
        if 0 < pos.sum() < len(yy):
            aucs.append(roc_auc_score(pos, proba[:, c]))
    return {"acc": float((pred == yy).mean()),
            "bal_acc": float(balanced_accuracy_score(yy, pred)),
            "auc_ovr_macro": float(np.mean(aucs)),
            "lop_da_so": float(np.bincount(yy).max() / len(yy))}


def boot_acc(yy, pred, uid, n, seed):
    vals = [(pred[i] == yy[i]).mean() for i in cluster_boot_indices(uid, n, seed)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ OOF LR
def lr_oof(design, y, fold):
    oof = np.zeros(len(y))
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        oof[va] = make_lr().fit(design[tr], y[tr]).predict_proba(design[va])[:, 1]
    return oof


def onehot(lot):
    cols = sorted(np.unique(lot))
    return (lot[:, None] == np.array(cols)[None, :]).astype(np.float32), cols


# ------------------------------------------------------------------ chinh
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["control", "frozen"])
    ap.add_argument("--tag", default="A_goc", help="ten bien the dac trung (A_goc | B_che_chu_cat_mask ...)")
    ap.add_argument("--features", type=Path, default=None,
                    help="parquet embedding thay the (img_id + img_feat_*). Mac dinh: embedding goc trong dataset npz")
    ap.add_argument("--fallback", type=Path, default=None, help="parquet embedding V0 (toan anh) cho anh hong")
    ap.add_argument("--zone-missing", type=Path, default=None, help="p01_zone_missing.csv (img_id anh cham 'hong')")
    ap.add_argument("--folds", type=Path, default=None,
                    help="parquet fold thay the (giai doan 2). Mac dinh: file fold hien co")
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--province", type=Path, default=OUT / "id_province.csv")
    ap.add_argument("--oof-dir", type=Path, default=ROOT / "fusion/output",
                    help="thu muc chua phase3_oof_{source}.npz de kiem tra tai hien L1")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    n_boot = args.n_boot or cfg["n_boot"]
    n_perm = args.n_perm if args.n_perm is not None else cfg["p01"]["probe_lo"]["n_hoan_vi"]
    sens = cfg["p01"]["auc_trong_lo"]["nhay_cam_phu"]
    t0 = time.time()

    nd = Nodes(args.source, dataset_dir=args.dataset_dir, features=args.features, folds=args.folds,
               fallback=args.fallback, zone_missing=args.zone_missing)
    m = nd.m
    assert_no_test(m, nd)
    if args.folds is None:      # file fold hien co: phai dung mau cua bao cao fusion GAT
        assert m.sum() == 1571 and nd.y[m].sum() == 393, (m.sum(), nd.y[m].sum())
    ym, um, fm, lm, Xm = nd.y[m], nd.uid[m], nd.fold[m], nd.lot[m], nd.X_img[m]
    assert (fm >= 0).all()
    lots = np.unique(lm)
    print("=" * 78)
    print(f"P0.1 — {args.tag} | nguon {args.source} | config {cfg['_hash']}")
    print("=" * 78)
    print(f"dev co anh: {m.sum()} dong, {ym.sum()} duong, {len(np.unique(um))} benh nhan, "
          f"{len(lots)} lo (gom 'unknown' neu co)")

    out = {"tag": args.tag, "source": args.source, "config_hash": cfg["_hash"],
           "file_fold": nd.folds_file, "zone_missing_dev_co_anh": int(nd.zone_missing[m].sum()),
           "thu_vien": lib_versions(), "n_boot": n_boot,
           "mau": {"n": int(m.sum()), "duong": int(ym.sum()), "benh_nhan": int(len(np.unique(um))),
                   "so_lo": int(len(lots))}}

    # ---------------- L1 (OOF, cung fold, cung cau hinh voi baselines.py) ----------------
    l1_full = sk_oof(["img"], [False], True, m, nd.fold, {"img": nd.X_img}, {"img": nd.has},
                     nd.y.astype(np.float32))
    l1 = l1_full[m]
    if args.features is None and args.folds is None:
        ref = np.load(args.oof_dir / f"phase3_oof_{args.source}.npz")["L1"]
        diff = float(np.abs(ref[m] - l1).max())
        out["kiem_tai_hien_L1_so_voi_phase3_oof"] = {"max_abs_diff": diff}
        print(f"[kiem tra] L1 tai hien so voi phase3_oof_{args.source}.npz: max|diff| = {diff:.2e}")
        assert diff < 1e-6, "L1 khong tai hien duoc OOF da luu — dung lai"
    pooled = score_block(ym, l1, um)
    out["L1_gop"] = pooled
    print(f"L1 AUC gop = {fmt_ci(pooled['roc'], pooled['roc_ci'])}")

    # ---------------- buoc 1: probe lo ----------------
    print("\n[1] Probe lo (du doan lo tu embedding)")
    yy, pred, proba = probe_lot(Xm, lm, fm)
    pm = probe_metrics(yy, pred, proba)
    pm["acc_ci"] = boot_acc(yy, pred, um, n_boot, seed)
    rng = np.random.default_rng(seed)
    null_acc = []
    for _ in range(n_perm):
        # nhan lo hoan vi giua cac dong: accuracy ky vong xap xi ty le lop da so
        yp, pp, _ = probe_lot(Xm, rng.permutation(lm), fm)
        null_acc.append(float((pp == yp).mean()))
    pm["acc_nhan_hoan_vi_tb"] = float(np.mean(null_acc)) if null_acc else None
    pm["acc_nhan_hoan_vi_toi_da"] = float(np.max(null_acc)) if null_acc else None
    out["probe_lo"] = pm
    print(f"  accuracy {fmt_ci(pm['acc'], pm['acc_ci'])} | lop da so {pm['lop_da_so']:.3f} | "
          f"hoan vi: tb {pm['acc_nhan_hoan_vi_tb']} toi da {pm['acc_nhan_hoan_vi_toi_da']}")
    print(f"  accuracy can bang {pm['bal_acc']:.3f} | AUC OvR macro {pm['auc_ovr_macro']:.3f}")

    # probe lo trong cung tinh (kham pha): tach ky thuat chup khoi dan so theo tinh
    if args.province.exists():
        prov = pd.read_csv(args.province).set_index("id")["tinh"]
        pv = prov.reindex(nd.id[m]).to_numpy()
        assert not pd.isna(pv).any(), "id_province.csv khong phu het id"
        rows = []
        tot_c = tot_n = tot_maj = 0
        for t in np.unique(pv):
            sel = pv == t
            sub_lots = np.unique(lm[sel])
            if len(sub_lots) < 2:
                continue
            y2, p2, _ = probe_lot(Xm[sel], lm[sel], fm[sel])
            acc = float((p2 == y2).mean())
            maj = float(np.bincount(y2).max() / len(y2))
            rows.append({"tinh": t, "so_lo": int(len(sub_lots)), "n": int(sel.sum()),
                         "acc": acc, "lop_da_so": maj})
            tot_c += (p2 == y2).sum(); tot_n += len(y2); tot_maj += np.bincount(y2).max()
        out["probe_lo_trong_tinh"] = {"theo_tinh": rows,
                                      "acc_gop": float(tot_c / tot_n),
                                      "lop_da_so_gop": float(tot_maj / tot_n)}
        print(f"  [kham pha] trong cung tinh: acc {tot_c / tot_n:.3f} so voi lop da so {tot_maj / tot_n:.3f}")
    else:
        print("  (bo qua probe trong tinh: thieu id_province.csv)")

    # ---------------- buoc 2: AUC trong lo cua L1 ----------------
    print("\n[2] AUC trong lo cua L1")
    tab, agg, dropped = lot_table(ym, l1, lm)
    _, agg_sens, dropped_sens = lot_table(ym, l1, lm, sens["min_duong"], sens["min_am"])
    scores = {"L1": l1}
    ci_all, _ = boot_lot_auc(ym, scores, lm, um, n_boot, seed)
    ci_sens, _ = boot_lot_auc(ym, scores, lm, um, n_boot, seed, sens["min_duong"], sens["min_am"])
    out["auc_trong_lo"] = {"chinh": {"auc": agg, "ci": ci_all["L1"], "lo_bi_loai": dropped,
                                     "so_lo_dung": int(len(lots) - len(dropped))},
                           "nhay_cam_min10": {"auc": agg_sens, "ci": ci_sens["L1"],
                                              "lo_bi_loai": dropped_sens,
                                              "so_lo_dung": int(len(lots) - len(dropped_sens))},
                           "bang_theo_lo": tab.to_dict("records")}
    print(f"  chinh: {fmt_ci(agg, ci_all['L1'])} | loai {len(dropped)} lo ({', '.join(dropped)})")
    print(f"  nhay cam (>=10 duong, >=10 am): {fmt_ci(agg_sens, ci_sens['L1'])} | "
          f"loai {len(dropped_sens)} lo")
    print(f"  (so voi AUC gop {pooled['roc']:.3f})")

    # ---------------- buoc 3: gia tri gia tang cua anh ----------------
    print("\n[3] Gia tri gia tang cua anh: LR(lo) vs LR(lo + anh)")
    oh, _ = onehot(lm)
    s_lot = lr_oof(oh, ym, fm)
    s_lot_img = lr_oof(np.hstack([oh, Xm]), ym, fm)
    d_pooled = paired_gap(ym, s_lot_img, s_lot, um, n=n_boot)
    auc_lot, auc_lot_img = roc_auc_score(ym, s_lot), roc_auc_score(ym, s_lot_img)
    _, agg_lot_img, _ = lot_table(ym, s_lot_img, lm)
    # Mo hinh chi biet lo khong co thong tin TRONG lo, nen moc dung la hang so (AUC = 0,5).
    # Khong dung OOF cua LR(lo) lam moc: diem cua no doi theo fold (ty le duong cua lo duoc
    # uoc luong lai tren tap train moi fold) nen AUC trong lo cua no lech khoi 0,5.
    _, agg_lot_oof, _ = lot_table(ym, s_lot, lm)
    ci3, dci3 = boot_lot_auc(ym, {"lot_img": s_lot_img, "hang_so": np.zeros(len(ym))}, lm, um,
                             n_boot, seed, pairs=[("lot_img", "hang_so")])
    dl = dci3[("lot_img", "hang_so")]
    out["gia_tri_gia_tang"] = {
        "auc_gop_LR_lo": auc_lot, "auc_gop_LR_lo_anh": auc_lot_img,
        "delta_auc_gop": {"delta": d_pooled[0], "ci": [d_pooled[1], d_pooled[2]], "p_pos": d_pooled[3]},
        "auc_trong_lo_LR_lo_anh": {"auc": agg_lot_img, "ci": ci3["lot_img"]},
        "auc_trong_lo_OOF_LR_lo": agg_lot_oof,
        "delta_auc_trong_lo": {"delta": agg_lot_img - 0.5, "ci": [dl[0], dl[1]], "p_pos": dl[2],
                               "ghi_chu": "so voi moc hang so 0,5; OOF cua LR(lo) khong dung lam moc vi diem doi theo fold"}}
    print(f"  AUC gop: LR(lo) {auc_lot:.3f} | LR(lo+anh) {auc_lot_img:.3f}")
    print(f"  delta AUC gop      = {d_pooled[0]:+.4f} [{d_pooled[1]:+.4f}; {d_pooled[2]:+.4f}] P(delta>0)={d_pooled[3]:.0%}")
    print(f"  AUC trong lo cua LR(lo+anh) so voi moc 0,5: {agg_lot_img - 0.5:+.4f} "
          f"[{dl[0]:+.4f}; {dl[1]:+.4f}] P(delta>0)={dl[2]:.0%}  "
          f"(OOF LR(lo) tu no cho {agg_lot_oof:.3f} trong lo: nhieu do fold, khong phai moc)")

    # ---------------- buoc 4: leave-batch-out ----------------
    print("\n[4] Leave-batch-out (bo han tung lo)")
    s_lolo = np.zeros(len(ym))
    removed = {}
    for L in lots:
        te = lm == L
        te_pat = set(um[te])
        tr = (~te) & np.array([u not in te_pat for u in um])
        removed[L] = int((~te).sum() - tr.sum())
        if ym[tr].min() == ym[tr].max():
            s_lolo[te] = 0.5
            continue
        s_lolo[te] = make_lr().fit(Xm[tr], ym[tr]).predict_proba(Xm[te])[:, 1]
    tab4, agg4, drop4 = lot_table(ym, s_lolo, lm)
    pooled4 = score_block(ym, s_lolo, um)
    ci4, dci4 = boot_lot_auc(ym, {"lolo": s_lolo, "infold": l1}, lm, um, n_boot, seed,
                             pairs=[("lolo", "infold")])
    d4 = dci4[("lolo", "infold")]
    out["leave_batch_out"] = {
        "auc_trong_lo": {"auc": agg4, "ci": ci4["lolo"], "lo_bi_loai": drop4},
        "auc_gop": {"auc": pooled4["roc"], "ci": pooled4["roc_ci"]},
        "chenh_so_voi_in_fold": {"delta": agg4 - agg, "ci": [d4[0], d4[1]], "p_pos": d4[2]},
        "benh_nhan_bi_loai_khoi_train_theo_lo": removed,
        "bang_theo_lo": tab4.to_dict("records")}
    print(f"  AUC gop (leave-batch-out)      = {fmt_ci(pooled4['roc'], pooled4['roc_ci'])}")
    print(f"  AUC trong lo (leave-batch-out) = {fmt_ci(agg4, ci4['lolo'])}")
    print(f"  chenh so voi in-fold = {agg4 - agg:+.4f} [{d4[0]:+.4f}; {d4[1]:+.4f}] P(delta>0)={d4[2]:.0%}")
    print(f"  dong bi loai khoi train (theo lo): toi da {max(removed.values())}")

    # ---------------- doi chieu bang quyet dinh (chi trinh bay) ----------------
    d_lo = d_pooled[1]
    cat = decision_v3(agg, d_lo)
    out["doi_chieu_bang_quyet_dinh"] = {
        "auc_trong_lo": agg, "delta_auc_gop_ktc_duoi": d_lo, "hang_muc": cat,
        "ghi_chu": "chi doi chieu voi bang cua thay; quyet dinh thuoc ve PGS"}
    print(f"\n[doi chieu bang quyet dinh] AUC trong lo = {agg:.3f}; can duoi KTC cua delta AUC gop = {d_lo:+.4f} "
          f"-> {cat}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"p01_{args.tag}_{args.source}" + ("" if args.folds is None else f"_{args.folds.stem}")
    write_json(args.out_dir / f"{stem}.json", out)
    np.savez_compressed(args.out_dir / f"{stem.replace('p01_', 'p01_oof_', 1)}.npz",
                        L1=_scatter(m, l1), LR_lo=_scatter(m, s_lot),
                        LR_lo_anh=_scatter(m, s_lot_img), L1_lolo=_scatter(m, s_lolo), id=nd.id)
    write_md(args.out_dir / f"{stem}.md", out)
    print(f"\nxong sau {time.time() - t0:.0f}s -> {args.out_dir / stem}.json|md")


def _scatter(mask, arr):
    out = np.zeros(len(mask))
    out[mask] = arr
    return out


def write_md(path: Path, o: dict) -> None:
    L = []
    a = L.append
    a(f"# P0.1 — {o['tag']} / nguồn {o['source']}")
    a(f"Config `{o['config_hash']}` · fold: {o['file_fold']} · ảnh zone_missing (dev): "
      f"{o['zone_missing_dev_co_anh']} · {o['mau']['n']} dòng dev có ảnh ({o['mau']['duong']} dương, "
      f"{o['mau']['benh_nhan']} bệnh nhân) · {o['mau']['so_lo']} lô · bootstrap {o['n_boot']} lần theo bệnh nhân.\n")
    p = o["probe_lo"]
    a("## 1. Probe lô")
    a(f"Accuracy {fmt_ci(p['acc'], p['acc_ci'])}; lớp đa số {p['lop_da_so']:.3f}; nhãn hoán vị trung bình "
      f"{p['acc_nhan_hoan_vi_tb']}. Accuracy cân bằng {p['bal_acc']:.3f}; AUC OvR macro {p['auc_ovr_macro']:.3f}.")
    if "probe_lo_trong_tinh" in o:
        t = o["probe_lo_trong_tinh"]
        a(f"\nKhám phá, trong cùng một tỉnh: accuracy {t['acc_gop']:.3f} so với lớp đa số {t['lop_da_so_gop']:.3f}.")
    r = o["auc_trong_lo"]
    a("\n## 2. AUC trong lô của L1")
    a(f"AUC gộp {o['L1_gop']['roc']:.3f}. AUC trong lô (chính): {fmt_ci(r['chinh']['auc'], r['chinh']['ci'])}, "
      f"dùng {r['chinh']['so_lo_dung']} lô, loại {len(r['chinh']['lo_bi_loai'])} lô ({', '.join(r['chinh']['lo_bi_loai'])}). "
      f"Nhạy cảm (≥10 dương và ≥10 âm): {fmt_ci(r['nhay_cam_min10']['auc'], r['nhay_cam_min10']['ci'])}, "
      f"dùng {r['nhay_cam_min10']['so_lo_dung']} lô.\n")
    a("| Lô | n | dương | âm | AUC |\n|---|---|---|---|---|")
    for x in r["bang_theo_lo"]:
        au = "—" if x["auc"] is None else f"{x['auc']:.3f}"
        a(f"| {x['lo']} | {x['n']} | {x['duong']} | {x['am']} | {au} |")
    g = o["gia_tri_gia_tang"]
    a("\n## 3. Giá trị gia tăng của ảnh")
    a(f"AUC gộp LR(lô) {g['auc_gop_LR_lo']:.3f}; LR(lô+ảnh) {g['auc_gop_LR_lo_anh']:.3f}. "
      f"ΔAUC gộp {g['delta_auc_gop']['delta']:+.4f} [{g['delta_auc_gop']['ci'][0]:+.4f}; "
      f"{g['delta_auc_gop']['ci'][1]:+.4f}], P(Δ>0)={g['delta_auc_gop']['p_pos']:.0%}. "
      f"ΔAUC trong lô {g['delta_auc_trong_lo']['delta']:+.4f} [{g['delta_auc_trong_lo']['ci'][0]:+.4f}; "
      f"{g['delta_auc_trong_lo']['ci'][1]:+.4f}], P(Δ>0)={g['delta_auc_trong_lo']['p_pos']:.0%}.")
    b = o["leave_batch_out"]
    a("\n## 4. Leave-batch-out (LOBO)")
    a(f"AUC gộp {fmt_ci(b['auc_gop']['auc'], b['auc_gop']['ci'])}; AUC trong lô {fmt_ci(b['auc_trong_lo']['auc'], b['auc_trong_lo']['ci'])}; chênh so với in-fold "
      f"{b['chenh_so_voi_in_fold']['delta']:+.4f} [{b['chenh_so_voi_in_fold']['ci'][0]:+.4f}; "
      f"{b['chenh_so_voi_in_fold']['ci'][1]:+.4f}], P(Δ>0)={b['chenh_so_voi_in_fold']['p_pos']:.0%}.")
    d = o["doi_chieu_bang_quyet_dinh"]
    a("\n## Đối chiếu bảng quyết định của thầy")
    a(f"AUC trong lô {d['auc_trong_lo']:.3f}; cận dưới KTC của ΔAUC gộp {d['delta_auc_gop_ktc_duoi']:+.4f} → "
      f"**{d['hang_muc']}**. {d['ghi_chu']}.")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

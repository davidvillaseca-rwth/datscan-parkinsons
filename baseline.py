"""Feature baseline: hand-crafted striatal features -> logistic regression / XGBoost, repeated stratified CV.

Features come from output/manifest.csv (per-side SBR) plus shape features computed here from the
crops. The key clinical sign is that DaT loss starts in the posterior putamen (the "comma tail")
and spreads forward to the caudate, so we measure posterior vs anterior uptake per side.

  python baseline.py               # main run
  python baseline.py --scanner     # also test adding scanner/acquisition columns (shortcut check)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OUT = Path(__file__).resolve().parent / "output"


def side_shape_features(half: np.ndarray, prefix: str) -> dict:
    """Shape features for one hemisphere's half-crop (x, y, z), y = posterior->anterior.

    Striatal region = voxels above 40% of this side's 99.5th percentile (or SBR>0.5, whichever is
    higher). Split it along y at its own extent: anterior third ~ caudate head, posterior third ~
    putamen tail."""
    p = np.percentile(half, 99.5)
    thr = max(0.4 * p, 0.5)
    reg = half > thr
    f = {}
    if reg.sum() < 10:
        for k in ("ant", "mid", "post", "post_ant_ratio", "vol_ml", "y_extent_mm", "elong"):
            f[f"{prefix}_{k}"] = 0.0
        return f
    idx = np.argwhere(reg)
    vals = half[reg]
    y = idx[:, 1]
    y0, y1 = y.min(), y.max()
    ext = max(y1 - y0, 1)
    t1, t2 = y0 + ext / 3, y0 + 2 * ext / 3
    post = vals[y <= t1].mean() if (y <= t1).any() else 0.0
    mid = vals[(y > t1) & (y < t2)].mean() if ((y > t1) & (y < t2)).any() else 0.0
    ant = vals[y >= t2].mean() if (y >= t2).any() else 0.0
    # elongation: ratio of largest to smallest principal axis (comma = elongated, dot = round)
    cov = np.cov((idx * 2.0).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    f.update({
        f"{prefix}_ant": float(ant), f"{prefix}_mid": float(mid), f"{prefix}_post": float(post),
        f"{prefix}_post_ant_ratio": float(post / ant) if ant > 0 else 0.0,
        f"{prefix}_vol_ml": float(reg.sum() * 8 / 1000),
        f"{prefix}_y_extent_mm": float((y1 - y0 + 1) * 2.0),
        f"{prefix}_elong": float(np.sqrt(ev[0] / max(ev[-1], 1e-6))),
    })
    return f


def crop_features(uid: str) -> dict:
    c = np.load(OUT / "crops" / f"{uid}.npy")
    mid = c.shape[0] // 2
    f = {**side_shape_features(c[mid:], "R"), **side_shape_features(c[:mid], "L")}
    # Summaries by worse/better side rather than left/right, since the affected side varies by patient.
    for k in ("post", "post_ant_ratio", "elong", "y_extent_mm", "vol_ml"):
        a, b = f[f"R_{k}"], f[f"L_{k}"]
        f[f"min_{k}"], f[f"max_{k}"] = min(a, b), max(a, b)
    return f


def build_table() -> pd.DataFrame:
    m = pd.read_csv(OUT / "manifest.csv")
    cf = pd.DataFrame([{"uid": u, **crop_features(u)} for u in m.uid])
    df = m.merge(cf, on="uid")
    for k in ("top50", "top500"):
        a, b = df[f"sbr_R_{k}"], df[f"sbr_L_{k}"]
        df[f"sbr_max_{k}"] = np.maximum(a, b)
        df[f"sbr_mean_{k}"] = (a + b) / 2
    df["min_tail_ratio"] = df[["sbr_R_top500", "sbr_L_top500"]].min(axis=1) / \
        df[["sbr_R_top50", "sbr_L_top50"]].max(axis=1).clip(lower=1e-3)
    return df


SIDE_AGNOSTIC = [
    "sbr_min_top50", "sbr_max_top50", "sbr_mean_top50",
    "sbr_min_top500", "sbr_max_top500", "sbr_mean_top500",
    "asym_top50", "asym_top500", "min_tail_ratio", "hot_ml",
    "min_post", "max_post", "min_post_ant_ratio", "max_post_ant_ratio",
    "min_elong", "max_elong", "min_y_extent_mm", "max_y_extent_mm", "min_vol_ml", "max_vol_ml",
]
SCANNER = ["n_levels_log", "zoom0", "dim0", "dim2", "ref_value_log"]


def add_scanner_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_levels_log"] = np.log10(df.n_levels)
    z = df.orig_zooms.str.strip("()").str.split(",", expand=True).astype(float)
    s = df.orig_shape.str.strip("()").str.split(",", expand=True).astype(float)
    df["zoom0"], df["dim0"], df["dim2"] = z[0], s[0], s[2]
    df["ref_value_log"] = np.log10(df.ref_value.clip(lower=1e-3))
    return df


def models():
    return {
        "logreg_sbr2": (["sbr_min_top500", "asym_top500"],
                        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0))),
        "logreg_all": (SIDE_AGNOSTIC,
                       lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))),
        "xgb_all": (SIDE_AGNOSTIC,
                    lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                          reg_lambda=1.0, eval_metric="logloss", n_jobs=4)),
    }


def cv_eval(df, feats, make, n_splits=5, n_repeats=5, seed=0):
    X, y = df[feats].to_numpy(float), df.label.to_numpy(int)
    oof = np.zeros((n_repeats, len(y)))
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        mdl = make().fit(X[tr], y[tr])
        oof[i // n_splits, te] = mdl.predict_proba(X[te])[:, 1]
    per_rep = [(roc_auc_score(y, p), log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)), accuracy_score(y, p > 0.5))
               for p in oof]
    r = np.array(per_rep)
    return {"auc": r[:, 0].mean(), "auc_sd": r[:, 0].std(), "logloss": r[:, 1].mean(),
            "acc": r[:, 2].mean()}, oof.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scanner", action="store_true")
    args = ap.parse_args()

    df = build_table()
    df.to_csv(OUT / "features.csv", index=False)
    print(f"{len(df)} scans, {df.label.mean():.1%} pathologic, {len(SIDE_AGNOSTIC)} features\n")

    rows, oofs = [], {}
    for name, (feats, make) in models().items():
        res, oof = cv_eval(df, feats, make)
        rows.append({"model": name, **res})
        oofs[name] = oof
    if args.scanner:
        d2 = add_scanner_cols(df)
        for name in ("logreg_all", "xgb_all"):
            feats, make = models()[name]
            res, _ = cv_eval(d2, feats + SCANNER, make)
            rows.append({"model": name + "+scanner", **res})
        # can scanner columns alone predict the label? (shortcut size)
        res, _ = cv_eval(d2, SCANNER, models()["xgb_all"][1])
        rows.append({"model": "xgb_scanner_only", **res})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    best = max(rows[:3], key=lambda r: r["auc"])["model"]
    pd.DataFrame({"uid": df.uid, "label": df.label, "oof": oofs[best]}).to_csv(OUT / f"oof_{best}.csv", index=False)

    # importance from a full-data XGB fit (indicative only)
    feats, make = models()["xgb_all"]
    mdl = make().fit(df[feats].to_numpy(float), df.label.to_numpy(int))
    imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nXGB feature importance (gain share):")
    print(imp.head(10).round(3).to_string())


if __name__ == "__main__":
    main()

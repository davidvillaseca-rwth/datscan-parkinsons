"""Batch runner: preprocess scans, save crops + sanity PNGs + manifest, log failures.

Examples:
  python run_preprocess.py --ids 01nouhtc 0224wk0y        # specific scans
  python run_preprocess.py --n 16 --seed 0                  # random sample
  python run_preprocess.py --all --no-plots                 # everything
"""

from __future__ import annotations

import argparse
import os
import random
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from preprocess import CROP_SHAPE, Result, process_scan  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def find_data(data_dir: Path) -> tuple[Path, Path]:
    """Locate the scan folder and label CSV inside the extracted DrivenData download.

    The download names them with a random suffix (e.g. niftis_utCGpHE/, train_labels_JNDlMjr.csv),
    so match on the prefix instead of hardcoding the names."""
    niftis = sorted(p for p in data_dir.glob("niftis*") if p.is_dir())
    labels = sorted(data_dir.glob("train_labels*.csv"))
    if not niftis or not labels:
        raise SystemExit(f"Couldn't find niftis*/ and train_labels*.csv in {data_dir}. "
                         "Point --data-dir (or DAT_DATA_DIR) at the folder you extracted the data into.")
    return niftis[0], labels[0]


def sanity_plot(uid: str, label, r: Result, path: Path) -> None:
    d = r.debug
    vol, c = d["vol"], d["center"]
    half = np.array(CROP_SHAPE) / 2
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    vmax = np.percentile(vol[d["head"]], 99.9)
    # (projection axis, displayed axes (h, v), title)
    views = [(2, (0, 1), "axial MIP (x→R, y→A)"), (1, (0, 2), "coronal MIP (x→R, z→S)"),
             (0, (1, 2), "sagittal MIP (y→A, z→S)")]
    for j, (pa, (h, v), title) in enumerate(views):
        a = ax[0, j]
        a.imshow(vol.max(axis=pa).T, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        a.contour(d["head"].any(axis=pa).T, levels=[0.5], colors="cyan", linewidths=0.6)
        a.contour(d["ref_mask"].any(axis=pa).T, levels=[0.5], colors="lime", linewidths=0.8)
        a.add_patch(patches.Rectangle((c[h] - half[h], c[v] - half[v]), 2 * half[h], 2 * half[v],
                                      fill=False, ec="white", lw=1.2))
        a.plot(d["seed"][h], d["seed"][v], "c+", ms=10)
        a.set_title(title, fontsize=9)
        a.axis("off")
    crop = r.crop
    # slice through the hottest voxel of the crop (centre x is the midline, which is empty)
    hx, hy, hz = np.unravel_index(np.argmax(crop), crop.shape)
    slices = [(crop[:, :, hz].T, f"crop axial z={hz}"), (crop[:, hy, :].T, f"crop coronal y={hy}"),
              (crop[hx, :, :].T, f"crop sagittal x={hx}")]
    for j, (s, t) in enumerate(slices):
        im = ax[1, j].imshow(s, origin="lower", cmap="turbo", vmin=-0.5, vmax=4)
        ax[1, j].set_title(t, fontsize=9)
        ax[1, j].axis("off")
    fig.colorbar(im, ax=ax[1, :], shrink=0.8, label="SBR (I/ref - 1)")
    i = r.info
    fig.suptitle(
        f"{uid} | label={label} | orig {i['orig_shape']} @ {i['orig_zooms']} mm | "
        f"ref={i['ref_method']} ({i['ref_voxels']} vox) | SBR R={i['sbr_R_top50']:.2f} "
        f"L={i['sbr_L_top50']:.2f} | clipped={i['crop_clipped']}",
        fontsize=9)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", nargs="+")
    g.add_argument("--n", type=int)
    g.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("DAT_DATA_DIR", Path(__file__).resolve().parent.parent)),
                   help="folder holding niftis*/ and train_labels*.csv (default: $DAT_DATA_DIR or the parent folder)")
    args = p.parse_args()

    nifti_dir, labels_csv = find_data(args.data_dir)
    labels = pd.read_csv(labels_csv).set_index("uid")["is_pathologic"]
    all_ids = sorted(f.name.removesuffix(".nii.gz") for f in nifti_dir.glob("*.nii.gz"))
    if args.ids:
        ids = args.ids
    elif args.n:
        ids = random.Random(args.seed).sample(all_ids, args.n)
    else:
        ids = all_ids

    (args.out / "crops").mkdir(parents=True, exist_ok=True)
    (args.out / "sanity").mkdir(parents=True, exist_ok=True)
    rows, fails = [], []
    t0 = time.time()
    for k, uid in enumerate(ids, 1):
        label = labels.get(uid, np.nan)
        try:
            r = process_scan(nifti_dir / f"{uid}.nii.gz")
            np.save(args.out / "crops" / f"{uid}.npy", r.crop.astype(np.float32))
            if not args.no_plots:
                sanity_plot(uid, label, r, args.out / "sanity" / f"{uid}.png")
            rows.append({"uid": uid, "label": label, **r.info})
            print(f"[{k}/{len(ids)}] {uid} ok  ref={r.info['ref_method']} "
                  f"R={r.info['sbr_R_top50']:.2f} L={r.info['sbr_L_top50']:.2f} label={label}")
        except Exception as e:  # noqa: BLE001
            fails.append({"uid": uid, "error": repr(e), "trace": traceback.format_exc()})
            print(f"[{k}/{len(ids)}] {uid} FAILED: {e!r}")
    print(f"done in {time.time() - t0:.0f}s: {len(rows)} ok, {len(fails)} failed")

    man = args.out / "manifest.csv"
    new = pd.DataFrame(rows)
    if man.exists() and not args.all:  # merge with earlier partial runs
        old = pd.read_csv(man)
        new = pd.concat([old[~old.uid.isin(new.uid)], new], ignore_index=True)
    new.to_csv(man, index=False)
    pd.DataFrame(fails, columns=["uid", "error", "trace"]).to_csv(args.out / "failures.csv", index=False)


if __name__ == "__main__":
    main()

"""DaT-SPECT preprocessing: raw NIfTI -> fixed-size striatal crop normalized to a reference region.

Python port + redesign of the old MATLAB DataSetMaker pipeline. Only uses numpy/scipy/nibabel,
all of which exist in the competition runtime, so the same code runs at inference time.

Steps per scan:
  1. Load, reorient to closest canonical RAS.
  2. Crop to the head bounding box (fast: 256^3 scans are mostly air), resample to 2 mm isotropic.
  3. Head mask -> eroded "inner head" search region (drops scalp, glands near the surface, ring artifacts).
  4. Find the striatum: brightest point of the smoothed image inside the search region, restricted
     to near the head midline; then refine the centre from the hot voxels around it.
  5. Reference region: occipital box posterior to the striatum -> denominator.
  6. Output = image / reference - 1 (specific binding ratio map; background ~0, healthy striatum ~2+).
  7. Fixed crop of CROP_SHAPE voxels around the striatal centre, padded with 0 (= reference level).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

VOXEL_MM = 2.0
CROP_SHAPE = (48, 48, 32)  # x (L-R), y (P-A), z (I-S) voxels at 2 mm -> 96 x 96 x 64 mm
# Harmonizing Gaussian (sigma, mm) applied after resampling. Scanners differ in reconstruction
# sharpness, and ~23% of scans are stored with <64 integer levels (some only 4-10); this evens
# both out. Set to 0 to disable.
SMOOTH_MM = 2.0


@dataclass
class Result:
    crop: np.ndarray  # float32, CROP_SHAPE, SBR map (I/ref - 1)
    info: dict = field(default_factory=dict)  # scalar diagnostics + features
    debug: dict = field(default_factory=dict)  # arrays for the sanity plot


def _mm(v: float) -> float:
    """mm -> voxels at the working resolution."""
    return v / VOXEL_MM


def load_resampled(path) -> tuple[np.ndarray, dict]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0
    n_levels = int(len(np.unique(data)))
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)

    # Crop to head bounding box at native resolution (generous threshold, plus margin).
    sm = ndi.gaussian_filter(data, sigma=6.0 / zooms)
    thr = 0.05 * np.percentile(sm[sm > 0], 99.5) if np.any(sm > 0) else 0
    nz = np.argwhere(sm > thr)
    if len(nz) == 0:
        raise ValueError("empty image")
    margin = np.ceil(10.0 / zooms).astype(int)
    lo = np.maximum(nz.min(0) - margin, 0)
    hi = np.minimum(nz.max(0) + margin + 1, data.shape)
    data = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    res = ndi.zoom(data, zooms / VOXEL_MM, order=1, prefilter=False)
    res[res < 0] = 0
    if SMOOTH_MM > 0:
        res = ndi.gaussian_filter(res, sigma=_mm(SMOOTH_MM))
    meta = {
        "n_levels": n_levels,
        "orig_shape": tuple(int(s) for s in img.shape[:3]),
        "orig_zooms": tuple(round(float(z), 3) for z in zooms),
        "res_shape": tuple(int(s) for s in res.shape),
    }
    return res, meta


def head_mask(vol: np.ndarray) -> np.ndarray:
    # Threshold relative to typical *brain* intensity (not the 99th percentile, which is the
    # striatum and would make the mask depend on the very signal we measure).
    sm = ndi.gaussian_filter(vol, sigma=_mm(6))
    t_otsu = threshold_otsu(sm[sm > 0])
    brain_med = np.median(sm[sm > t_otsu])
    m = sm > 0.35 * brain_med
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        raise ValueError("no head found")
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (np.argmax(sizes) + 1)
    # fill holes slice-wise (axial) and in 3D
    m = ndi.binary_fill_holes(m)
    for z in range(m.shape[2]):
        m[:, :, z] = ndi.binary_fill_holes(m[:, :, z])
    return m


def find_striatum(vol: np.ndarray, head: np.ndarray) -> dict:
    sm = ndi.gaussian_filter(vol, sigma=_mm(4))

    # Midline: centre of mass of the head mask in x (L-R).
    com = np.array(ndi.center_of_mass(head))
    x_mid = com[0]

    # Search region: inner head (eroded ~20 mm), within +/-40 mm of midline.
    inner = ndi.binary_erosion(head, iterations=int(round(_mm(20))))
    if inner.sum() < 100:  # tiny / truncated FOV -> erode less
        inner = ndi.binary_erosion(head, iterations=int(round(_mm(8))))
    xs = np.arange(vol.shape[0])[:, None, None]
    search = inner & (np.abs(xs - x_mid) <= _mm(40))

    # Vertical prior: across a 150-scan sample the striatum sits 87 +/- 10 mm below the top of
    # the head (range 70-113 when correct). When the head top is inside the FOV, only search
    # 55-125 mm below it. Stops faint (severe-PD) striata losing to the skull base / glands.
    zs = np.where(head.any(axis=(0, 1)))[0]
    top_in_fov = bool(zs.max() < vol.shape[2] - 2)
    if top_in_fov:
        zi = np.arange(vol.shape[2])[None, None, :]
        dz = (zs.max() - zi) * VOXEL_MM
        prior = search & (dz >= 55) & (dz <= 125)
        if prior.any():
            search = prior
    if not search.any():
        search = head

    masked = np.where(search, sm, -np.inf)
    seed = np.array(np.unravel_index(np.argmax(masked), sm.shape))
    peak = sm[tuple(seed)]

    # Refine: hot voxels (>50% of peak) within 45 mm of the seed, intensity-weighted centroid in y/z.
    zz = np.indices(sm.shape, dtype=np.float32)
    dist = np.sqrt(sum((zz[i] - seed[i]) ** 2 for i in range(3))) * VOXEL_MM
    hot = (sm > 0.5 * peak) & (dist <= 45) & head
    w = np.where(hot, sm, 0)
    centroid = np.array([(zz[i] * w).sum() / w.sum() for i in range(3)])

    # x centre: midline between the two striata. Only trust the hot-voxel x extent when hot
    # voxels exist on BOTH sides of the head midline; with one dim side (unilateral loss or
    # asymmetry) the extent covers one striatum only, so fall back to the head midline.
    hx = np.where(hot.any(axis=(1, 2)))[0]
    n_left = int(hot[: int(x_mid)].sum())
    n_right = int(hot[int(x_mid) + 1:].sum())
    x_c = x_mid
    if n_left >= 20 and n_right >= 20:
        x_c = 0.5 * (hx.min() + hx.max())
        if abs(x_c - x_mid) > _mm(15):
            x_c = x_mid
    center = np.array([x_c, centroid[1], centroid[2]])
    return {
        "seed": seed,
        "center": center,
        "x_mid": float(x_mid),
        "top_in_fov": top_in_fov,
        "dz_top_mm": float((zs.max() - center[2]) * VOXEL_MM),
        "peak_smoothed": float(peak),
        "hot_mask": hot,
        "search": search,
    }


def reference_value(vol: np.ndarray, head: np.ndarray, center: np.ndarray) -> tuple[float, dict]:
    """Mean of an occipital box: 50-90 mm posterior of the striatum, +/-35 mm L-R, -5..+35 mm S-I,
    inside the head eroded by 12 mm (to avoid scalp/skull edge). Falls back to the inner-head median."""
    inner = ndi.binary_erosion(head, iterations=int(round(_mm(12))))
    idx = np.indices(vol.shape, dtype=np.float32)
    d = [(idx[i] - center[i]) * VOXEL_MM for i in range(3)]
    box = (np.abs(d[0]) <= 35) & (d[1] <= -50) & (d[1] >= -90) & (d[2] >= -5) & (d[2] <= 35)
    region = box & inner
    info = {"ref_voxels": int(region.sum())}
    if region.sum() >= 200:
        ref = float(vol[region].mean())
        info["ref_method"] = "occipital"
    else:
        # fallback: inner head excluding a 40 mm sphere around the striatum
        far = np.sqrt(sum(di ** 2 for di in d)) > 40
        region = inner & far if (inner & far).sum() > 200 else head & far
        ref = float(np.median(vol[region]))
        info["ref_method"] = "inner_median"
        info["ref_voxels"] = int(region.sum())
    return ref, {**info, "ref_mask": region}


def fixed_crop(vol: np.ndarray, center: np.ndarray, shape=CROP_SHAPE, pad=0.0) -> np.ndarray:
    out = np.full(shape, pad, dtype=np.float32)
    start = np.round(center).astype(int) - np.array(shape) // 2
    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + np.array(shape), vol.shape)
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    if np.all(src_hi > src_lo):
        out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
            vol[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def striatal_features(crop: np.ndarray, hot: np.ndarray) -> dict:
    """Simple per-side features from the SBR crop (for a non-deep baseline).

    Each side is measured in its own half of the crop (not via the >50%-of-peak hot mask, which
    is empty on a strongly affected side and used to give SBR=0 there). Crop x centre is the
    midline; RAS so +x = right. top50 ~ peak (caudate-dominated), top500 ~ whole striatum (4 ml)."""
    mid = crop.shape[0] // 2
    feats = {}
    for side, half in (("R", crop[mid:]), ("L", crop[:mid])):
        vals = np.sort(half.ravel())[::-1]
        feats[f"sbr_{side}_top50"] = float(vals[:50].mean())
        feats[f"sbr_{side}_top500"] = float(vals[:500].mean())
    feats["hot_ml"] = float(hot.sum() * VOXEL_MM ** 3 / 1000)
    a, b = feats["sbr_R_top50"], feats["sbr_L_top50"]
    feats["asym_top50"] = float(abs(a - b) / (abs(a + b) / 2 + 1e-6))
    a, b = feats["sbr_R_top500"], feats["sbr_L_top500"]
    feats["asym_top500"] = float(abs(a - b) / (abs(a + b) / 2 + 1e-6))
    feats["sbr_min_top50"] = min(feats["sbr_R_top50"], feats["sbr_L_top50"])
    feats["sbr_min_top500"] = min(feats["sbr_R_top500"], feats["sbr_L_top500"])
    return feats


def process_scan(path) -> Result:
    vol, meta = load_resampled(path)
    head = head_mask(vol)
    st = find_striatum(vol, head)
    ref, rinfo = reference_value(vol, head, st["center"])
    if ref <= 0:
        raise ValueError("non-positive reference value")
    sbr = vol / ref - 1.0
    crop = fixed_crop(sbr, st["center"])

    edge = np.round(st["center"]).astype(int) - np.array(CROP_SHAPE) // 2
    clipped = bool(np.any(edge < 0) or np.any(edge + np.array(CROP_SHAPE) > vol.shape))
    info = {
        **meta,
        "center_vox": tuple(round(float(c), 1) for c in st["center"]),
        "ref_value": ref,
        "ref_method": rinfo["ref_method"],
        "ref_voxels": rinfo["ref_voxels"],
        "crop_clipped": clipped,
        "crop_max_sbr": float(crop.max()),
        "top_in_fov": st["top_in_fov"],
        "dz_top_mm": round(st["dz_top_mm"], 1),
        **striatal_features(crop, st["hot_mask"]),
    }
    debug = {"vol": vol, "head": head, "search": st["search"], "seed": st["seed"],
             "center": st["center"], "ref_mask": rinfo["ref_mask"], "hot": st["hot_mask"]}
    return Result(crop=crop, info=info, debug=debug)

"""
runme_inpainting.py  -  Q3: ADMM-PnP inpainting with the BM3D denoiser
=====================================================================
Observation model:  y = x_gt(M),   where M is a random set of indices covering
OBSERVED_RATIO (= 20%) of the pixels; the other 80% are unobserved. There is no
additive noise in Q3. The random seed is fixed to 0 before drawing the indices
(set once in build_observations before iterating images in canonical order), so a
given image gets the same mask across tuning and reporting.

Missing pixels are first filled by the prior-free MEDIAN-INPAINTING initialisation
(median_inpainting.py, from CM4IR), which ADMM-PnP then refines. Without a sensible
init the denoiser has nothing to propagate from the mostly-empty observation.

There is no cyclic convolution here, so PSNR uses no crop (unlike Q2).

Each invocation runs tune then report:

  1. TUNE   - grid-search a small hyper-parameter grid on TUNE_NAMES (5 diverse
              images: 3x256^2 + 2x512^2), scored by average PSNR, print the full
              ranking, and pick the best config. Tuned-subset results are cached so
              the winner is not recomputed in step 2.
  2. REPORT - apply that single best config to all 8 test images; print/save a
              PSNR table (PSNR_y masked input vs PSNR_xhat inpainted, per image +
              average) and a visual grid (x_gt, y, x_hat) for VISUAL_NAMES.

Hyper-parameter grid (see BM3D_GRID below): 8 combos (n_iter, rho, sigma_d).

Outputs (under outputs/inpainting/):
  q3_inpainting_BM3D.txt  - PSNR table + tuning ranking footer
  q3_inpainting_BM3D.png  - VISUAL_NAMES rows x (gt, y, x_hat)

Usage:
  .venv/bin/python runme_inpainting.py                       # Q3 BM3D
  .venv/bin/python runme_inpainting.py --bm3d-mode full
"""
import argparse
import itertools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils import load_all_images, psnr
from bm3d import bm3d, BM3DStages
from admm_pnp import admm_pnp_inpainting
from median_inpainting import median_inpainting

# BM3D stage map. 'hard' (hard-thresholding only) is ~2x faster on 512^2 images
# and is the standard choice for PnP-ADMM inner iterations.
BM3D_STAGE = {
    "hard": BM3DStages.HARD_THRESHOLDING,
    "full": BM3DStages.ALL_STAGES,
}

# Configuration
SEED           = 0
# Fraction of pixels that are observed (the indices in M); the rest are missing.
OBSERVED_RATIO = 0.2

# Diverse subset used for tuning: 3 small (256x256) + 2 large (512x512) images so
# the sweep stays fast (BM3D is called every ADMM iteration and costs ~4x more on
# 512x512). The chosen config is then evaluated on all 8 images in the report.
TUNE_NAMES   = ["cameraman", "house", "peppers", "lena", "barbara"]
# 4 examples shown in the report visual: a 256x256, two 512x512, texture + structure.
VISUAL_NAMES = ["cameraman", "lena", "barbara", "boat"]

OUT_DIR = Path("outputs") / "inpainting"

# Hyper-parameter grid. Sweep tuple order is (n_iter, rho, sigma_d). rho is small
# because the data term constrains only 20% of the pixels, so we lean on the
# denoiser; sigma_d controls how aggressively BM3D smooths the inpainted fill.
BM3D_GRID = dict(n_iter=[10, 20], rho=[0.02, 0.05],
                 sigma_d=[0.02, 0.04])               # 2 * 2 * 2 = 8 combos


def make_denoiser(bm3d_mode="hard"):
    """Return a denoiser(img, sigma) callable backed by BM3D."""
    stage = BM3D_STAGE[bm3d_mode]
    def denoiser(img, sigma):
        return bm3d(z=img, sigma_psd=sigma, stage_arg=stage)
    return denoiser


def median_init(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """numpy adapter around CM4IR's torch median_inpainting (grayscale, square).

    Feeds the masked image as a (1, 1, H, W) tensor and the flat mask, then
    reshapes the flattened result back to a 2-D [0,1] float64 image.
    """
    h, w = y.shape
    assert h == w, "median_inpainting assumes square images (image_size = H = W)"
    y_t    = torch.from_numpy(np.ascontiguousarray(y)).float().view(1, 1, h, w)
    mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().reshape(1, h * w)
    out = median_inpainting(y_t, mask_t, in_channels=1, image_size=h)
    return out.view(h, w).cpu().numpy().astype(np.float64)


def build_observations(images: dict) -> dict:
    """Generate y = x_gt(M), the binary mask M, and the median init (no noise)."""
    base = {}
    # Seed once, then iterate in canonical order so each image gets a reproducible
    # set of observed indices shared between tuning and reporting.
    np.random.seed(SEED)
    for name, x_gt in images.items():
        n_obs = int(round(x_gt.size * OBSERVED_RATIO))
        idx = np.random.choice(x_gt.size, size=n_obs, replace=False)
        mask = np.zeros(x_gt.size)
        mask[idx] = 1.0
        mask = mask.reshape(x_gt.shape)
        y = mask * x_gt
        base[name] = {"gt": x_gt, "noisy": y, "mask": mask,
                      "init": median_init(y, mask)}
    return base


def format_combo(ni, r, sd) -> str:
    return f"n_iter={ni}, rho={r}, sigma_d={sd}"


def build_combos():
    g = BM3D_GRID
    return list(itertools.product(g["n_iter"], g["rho"], g["sigma_d"]))


def q3(bm3d_mode: str = "hard"):
    """Tune on the diverse subset, then report the best config on all 8 images."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label        = "BM3D"
    images       = load_all_images()
    base         = build_observations(images)
    combos       = build_combos()
    tune_names   = TUNE_NAMES
    visual_names = VISUAL_NAMES
    denoiser     = make_denoiser(bm3d_mode)

    # ---- Step 1: tune on the diverse subset --------------------------------
    print(f"\nQ3 inpainting  -  denoiser={label}  (observed {OBSERVED_RATIO:.0%})")
    print(f"Tuning {len(combos)} combos on {len(tune_names)} images: {tune_names}\n")

    cache   = {}   # (combo_idx, name) -> x_hat, so the winning combo isn't recomputed
    results = []   # (avg, idx, ni, r, sd)
    for idx, (ni, r, sd) in enumerate(tqdm(combos, desc="tune combos")):
        per_image = []
        for name in tune_names:
            d = base[name]
            x_hat = admm_pnp_inpainting(d["noisy"], d["mask"], denoiser,
                                        sd, r, n_iter=ni, x_init=d["init"])
            cache[(idx, name)] = x_hat
            per_image.append(psnr(x_hat, d["gt"]))
        avg = float(np.mean(per_image))
        results.append((avg, idx, ni, r, sd))
        tqdm.write(f"  {format_combo(ni, r, sd)}  avg PSNR = {avg:.2f}")

    results.sort(key=lambda t: -t[0])
    best_avg, best_idx, b_ni, b_r, b_sd = results[0]
    best_cfg = format_combo(b_ni, b_r, b_sd)

    rank_lines = [f"Tuning ranking (avg PSNR over {tune_names}):", "",
                  f"{'avg':>8}   config", "-" * 60]
    for avg, idx, ni, r, sd in results:
        rank_lines.append(f"{avg:>8.2f}   {format_combo(ni, r, sd)}")
    ranking = "\n".join(rank_lines)
    print("\n" + ranking)
    print(f"\nBest config: avg={best_avg:.2f} dB  -  {best_cfg}\n")

    # ---- Step 2: report best config on ALL 8 images ------------------------
    for name in tqdm(list(base.keys()), desc="report (best config)"):
        d = base[name]
        # reuse the tuned result for subset images; compute held-out images fresh
        d["inpaint"] = cache.get((best_idx, name))
        if d["inpaint"] is None:
            d["inpaint"] = admm_pnp_inpainting(d["noisy"], d["mask"], denoiser,
                                               b_sd, b_r, n_iter=b_ni,
                                               x_init=d["init"])

    psnr_y, psnr_x, body = [], [], []
    for name, d in base.items():
        py = psnr(d["noisy"], d["gt"])
        px = psnr(d["inpaint"], d["gt"])
        psnr_y.append(py)
        psnr_x.append(px)
        body.append(f"{name:<12}{py:>14.2f}{px:>14.2f}")

    sep = "-" * 40
    table = "\n".join(
        [f"{'Image':<12}{'PSNR_y':>14}{'PSNR_xhat':>14}", sep]
        + body
        + [sep,
           f"{'Average':<12}{np.mean(psnr_y):>14.2f}{np.mean(psnr_x):>14.2f}",
           "",
           f"Denoiser: {label}  |  best config: {best_cfg}  |  "
           f"tuned on {len(tune_names)} images"]
    )
    print("\n" + table + "\n")

    txt_path = OUT_DIR / f"q3_inpainting_{label}.txt"
    txt_path.write_text(table + "\n\n" + ranking + "\n", encoding="utf-8")
    print(f"Table saved to {txt_path.name}")

    # ---- visual examples (x_gt, y, x_hat) ----------------------------------
    fig, axes = plt.subplots(len(visual_names), 3, figsize=(12, 4 * len(visual_names)))
    fig.suptitle(f"Q3 Inpainting - {label}  (best: {best_cfg})", fontsize=11, y=1.01)
    for row, name in enumerate(visual_names):
        d  = base[name]
        py = psnr(d["noisy"], d["gt"])
        px = psnr(d["inpaint"], d["gt"])
        for col, (arr, title) in enumerate([
            (d["gt"],      "Ground truth"),
            (d["noisy"],   f"Masked ({OBSERVED_RATIO:.0%} obs)\nPSNR={py:.2f} dB"),
            (d["inpaint"], f"ADMM-PnP ({label})\nPSNR={px:.2f} dB"),
        ]):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{name} - {title}", fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    fname = OUT_DIR / f"q3_inpainting_{label}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visual saved to {fname.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADMM-PnP inpainting: tune on a diverse subset, report on all 8.")
    parser.add_argument("--bm3d-mode", choices=["hard", "full"], default="hard",
                        help="BM3D stages inside ADMM. 'hard' (default) = "
                             "hard-thresholding only (~2x faster, standard PnP). "
                             "'full' = HT + Wiener.")
    args = parser.parse_args()
    q3(bm3d_mode=args.bm3d_mode)

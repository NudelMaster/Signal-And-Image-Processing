"""
runme_deblurring.py  -  Q2: ADMM-PnP deblurring with BM3D / BF denoisers
=======================================================================
Noise model:  y = conv(x_gt, k) + e,   e ~ N(0, sigma_e^2 * I),  sigma_e = 0.01,
random seed 0 (set once in build_observations before iterating images in
canonical order),  k(i,j) = 1/(1 + i^2 + j^2), i,j = -7..7, scaled so sum k = 1.
The 2D convolution is cyclic (FFT in utils); boundary pixels are unreliable, so
all PSNR uses crop=7 (kernel half-width).

Each invocation runs tune then report for one denoiser variant:

  1. TUNE   - grid-search a small hyper-parameter grid on TUNE_NAMES (5 diverse
              images: 3x256^2 + 2x512^2), scored by average PSNR with crop=7,
              print the full ranking, and pick the best config. Tuned-subset
              results are cached so the winner is not recomputed in step 2.
  2. REPORT - apply that single best config to all 8 test images; print/save a
              PSNR table (PSNR_y input vs PSNR_xhat restored, per image +
              average) and a visual grid (x_gt, y, x_hat) for VISUAL_NAMES.

Hyper-parameter grids (see BM3D_GRID / BF_GRID / BM3D_Q2C_GRID below):
  - Q2b BM3D: 8 combos  (n_iter, rho, sigma_d)
  - Q2b BF:   16 combos (n_iter, rho, sigma_s, sigma_r)
  - Q2c BM3D: 8 combos  (n_iter, rho_end, rho start factor, fixed sigma_d);
              rho INCREASES geometrically each ADMM iteration (PnP continuation)
              while sigma_d stays at its Q2b optimum. BM3D-only; forces
              --denoiser bm3d.

Outputs (under outputs/deblurring/):
  q2_deblurring_{BM3D,BF,BM3D_Q2C}.txt  - PSNR table + tuning ranking footer
  q2_deblurring_{BM3D,BF,BM3D_Q2C}.png  - VISUAL_NAMES rows x (gt, y, x_hat)

Usage:
  .venv/bin/python runme_deblurring.py --denoiser bm3d          # Q2b
  .venv/bin/python runme_deblurring.py --denoiser bf            # Q2b
  .venv/bin/python runme_deblurring.py --denoiser bm3d --bm3d-mode full
  .venv/bin/python runme_deblurring.py --q2c                    # Q2c improved BM3D
"""
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils import (load_all_images, psnr, add_gaussian_noise, make_blur_kernel,
                   fft_convolve_blur_kernel)
from bilateral_filter import bilateral_filter
from bm3d import bm3d, BM3DStages
from admm_pnp import admm_pnp_deblur

# BM3D stage map. 'hard' (hard-thresholding only) is ~2x faster on 512^2 images
# and is the standard choice for PnP-ADMM inner iterations.
BM3D_STAGE = {
    "hard": BM3DStages.HARD_THRESHOLDING,
    "full": BM3DStages.ALL_STAGES,
}

# Configuration
SIGMA_E   = 0.01
SEED      = 0
# Cyclic convolution makes the boundary ring (width = kernel half-width = 7)
# unreliable. so PSNR crops a 7-px border.
PSNR_CROP = 7

# Diverse subset used for tuning. We use 3 small (256x256) images + 2 512x512 images so the sweep
# stays fast (BM3D is called every ADMM iteration and costs ~4x more on 512x512).
# 
# the chosen config is then evaluated on all 8 images in the report below.
# Shared by both denoisers so they optimize a single config on identical
# content: 5 versatile images spanning sharp edges, smooth regions, portrait,
# and heavy texture. The chosen config is then evaluated on all 8 images below.
TUNE_NAMES   = ["cameraman", "house", "peppers", "lena", "barbara"]
# 4 examples shown in the report visual (also shared, so BM3D vs BF can be
# compared on the same content): a 256x256, two 512x512, texture + structure.
VISUAL_NAMES = ["cameraman", "lena", "barbara", "boat"]

OUT_DIR = Path("outputs") / "deblurring"

# Hyper-parameter grids. Sweep tuple order is (n_iter, rho, sigma_s, sigma_d);
# sigma_s is only used by BF (sigma_d plays the role of BF's sigma_r there).
BM3D_GRID = dict(n_iter=[8, 12], rho=[0.05, 0.1], sigma_s=[None],
                 sigma_d=[0.02, 0.04])               # 2 * 2 * 1 * 2 = 8 combos
BF_GRID   = dict(n_iter=[10, 15], rho=[0.05, 0.1], sigma_s=[1.5, 3.0],
                 sigma_d=[0.05, 0.08])               # 2 * 2 * 2 * 2 = 16 combos

# Q2c (q2c): the only change vs the Q2b baseline is INCREASING-rho continuation
# (textbook PnP): sigma_d is held fixed at its Q2b optimum while rho rises
# geometrically from start_factor*rho_end up to rho_end over the iterations.
# Here the tuple's 'rho' slot is the schedule's END (final) rho and 'start_factor'
# is where it begins (rho_start = start_factor*rho_end); start_factor < 1 gives
# increasing rho, start_factor=1 gives constant rho (== baseline, a control).
BM3D_Q2C_GRID = dict(n_iter=[12, 16], rho=[0.05, 0.1],
                     start_factor=[0.3, 1.0], sigma_d=[0.02])  # 2 * 2 * 2 * 1 = 8


def make_denoiser(denoiser_name, sigma_s, bm3d_mode="hard"):
    """Return a denoiser(img, sigma) callable for the requested family."""
    if denoiser_name == "bm3d":
        stage = BM3D_STAGE[bm3d_mode]
        def denoiser(img, sigma):
            return bm3d(z=img, sigma_psd=sigma, stage_arg=stage)
        return denoiser
    def denoiser(img, sigma_r):
        return bilateral_filter(img, sigma_s, sigma_r)
    return denoiser


def build_observations(images: dict, blur_kernel: np.ndarray) -> dict:
    """Generate y = conv(x_gt, k) + noise for each image."""
    base = {}
    # Set the random seed for the entire function 
    np.random.seed(SEED)
    for name, x_gt in images.items():
        blurred = fft_convolve_blur_kernel(x_gt, blur_kernel)
        y = np.clip(add_gaussian_noise(blurred, SIGMA_E), 0, 1)
        base[name] = {"gt": x_gt, "blurry": y}
    return base


def make_geom_schedule(end_value, start_factor, n_iter):
    """Geometric per-iteration schedule for Q2c continuation.

    Returns (init_value, schedule) where the value runs geometrically from
    start_factor*end_value to end_value over n_iter iterations. Used here for the
    rho schedule (start_factor < 1 => increasing rho); start_factor=1 is constant.
    schedule is a list of (iter, value) ready for admm_pnp_deblur's *_schedule args.
    """
    vals = np.geomspace(start_factor * end_value, end_value, n_iter)
    return float(vals[0]), list(enumerate(vals.tolist()))


def format_combo(denoiser_name, ni, r, ss, sd, q2c=False) -> str:
    if q2c:                                   # ss = rho start factor, r = rho_end
        return f"n_iter={ni}, rho={ss * r:.3f}->{r} (x{ss:g}), sigma_d={sd}"
    if denoiser_name == "bm3d":
        return f"n_iter={ni}, rho={r}, sigma_d={sd}"
    return f"n_iter={ni}, rho={r}, sigma_s={ss}, sigma_r={sd}"


def build_combos(denoiser_name, q2c=False):
    if q2c:
        g, third = BM3D_Q2C_GRID, BM3D_Q2C_GRID["start_factor"]  # sigma_d schedule start factor
    else:
        g = BM3D_GRID if denoiser_name == "bm3d" else BF_GRID
        third = g["sigma_s"]                                     # BF spatial sigma (None for BM3D)
    return list(itertools.product(g["n_iter"], g["rho"], third, g["sigma_d"]))


def q2b(denoiser_name: str = "bm3d", bm3d_mode: str = "hard", q2c: bool = False):
    """Tune on the diverse subset, then report the best config on all 8 images.

    q2c=True runs the improved BM3D variant: BM3D is forced as the denoiser and
    rho follows an increasing per-iteration schedule while sigma_d stays fixed at
    its Q2b optimum (see BM3D_Q2C_GRID).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if q2c:
        denoiser_name = "bm3d"                       # Q2c is BM3D-only per the spec
    label       = denoiser_name.upper() + ("_Q2C" if q2c else "")
    blur_kernel = make_blur_kernel(half=7)
    images      = load_all_images()
    base        = build_observations(images, blur_kernel)
    combos      = build_combos(denoiser_name, q2c)
    tune_names  = TUNE_NAMES
    visual_names = VISUAL_NAMES
    # ---- Step 1: tune on the per-denoiser diverse subset -------------------
    print(f"\nQ2 deblurring  -  denoiser={label}")
    print(f"Tuning {len(combos)} combos on {len(tune_names)} images: {tune_names}\n")

    cache   = {}   # (combo_idx, name) -> x_hat, so the winning combo isn't recomputed
    results = []   # (avg, idx, ni, r, ss, sd)
    for idx, (ni, r, ss, sd) in enumerate(tqdm(combos, desc="tune combos")):
        denoiser  = make_denoiser(denoiser_name, ss, bm3d_mode)
        # Q2c: rho follows an increasing schedule (sigma_d fixed); else fixed rho.
        rho_init, rho_sched = make_geom_schedule(r, ss, ni) if q2c else (r, None)
        per_image = []
        for name in tune_names:
            x_hat = admm_pnp_deblur(base[name]["blurry"], blur_kernel, denoiser,
                                    sd, rho_init, n_iter=ni, rho_schedule=rho_sched)
            cache[(idx, name)] = x_hat
            per_image.append(psnr(x_hat, base[name]["gt"], crop=PSNR_CROP))
        avg = float(np.mean(per_image))
        results.append((avg, idx, ni, r, ss, sd))
        tqdm.write(f"  {format_combo(denoiser_name, ni, r, ss, sd, q2c)}  avg PSNR = {avg:.2f}")

    results.sort(key=lambda t: -t[0])
    best_avg, best_idx, b_ni, b_r, b_ss, b_sd = results[0]
    best_cfg = format_combo(denoiser_name, b_ni, b_r, b_ss, b_sd, q2c)

    rank_lines = [f"Tuning ranking (avg PSNR over {tune_names}):", "",
                  f"{'avg':>8}   config", "-" * 60]
    for avg, idx, ni, r, ss, sd in results:
        rank_lines.append(f"{avg:>8.2f}   {format_combo(denoiser_name, ni, r, ss, sd, q2c)}")
    ranking = "\n".join(rank_lines)
    print("\n" + ranking)
    print(f"\nBest config: avg={best_avg:.2f} dB  -  {best_cfg}\n")

    # ---- Step 2: report best config on ALL 8 images ------------------------
    denoiser = make_denoiser(denoiser_name, b_ss, bm3d_mode)
    b_rho_init, b_rho_sched = make_geom_schedule(b_r, b_ss, b_ni) if q2c else (b_r, None)
    for name in tqdm(list(base.keys()), desc="report (best config)"):
        d = base[name]
        # reuse the tuned result for subset images; compute held-out images fresh
        d["deblur"] = cache.get((best_idx, name))
        if d["deblur"] is None:
            d["deblur"] = admm_pnp_deblur(d["blurry"], blur_kernel, denoiser,
                                          b_sd, b_rho_init, n_iter=b_ni,
                                          rho_schedule=b_rho_sched)

    psnr_y, psnr_x, body = [], [], []
    for name, d in base.items():
        pb = psnr(d["blurry"], d["gt"], crop=PSNR_CROP)
        px = psnr(d["deblur"], d["gt"], crop=PSNR_CROP)
        psnr_y.append(pb)
        psnr_x.append(px)
        body.append(f"{name:<12}{pb:>14.2f}{px:>14.2f}")

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

    txt_path = OUT_DIR / f"q2_deblurring_{label}.txt"
    txt_path.write_text(table + "\n\n" + ranking + "\n", encoding="utf-8")
    print(f"Table saved to {txt_path.name}")

    # ---- visual examples (x_gt, y, x_hat) ----------------------------------
    fig, axes = plt.subplots(len(visual_names), 3, figsize=(12, 4 * len(visual_names)))
    fig.suptitle(f"Q2 Deblurring - {label}  (best: {best_cfg})", fontsize=11, y=1.01)
    for row, name in enumerate(visual_names):
        d  = base[name]
        pb = psnr(d["blurry"], d["gt"], crop=PSNR_CROP)
        px = psnr(d["deblur"], d["gt"], crop=PSNR_CROP)
        for col, (arr, title) in enumerate([
            (d["gt"],     "Ground truth"),
            (d["blurry"], f"Blurry+Noisy\nPSNR={pb:.2f} dB"),
            (d["deblur"], f"ADMM-PnP ({label})\nPSNR={px:.2f} dB"),
        ]):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{name} - {title}", fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    fname = OUT_DIR / f"q2_deblurring_{label}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visual saved to {fname.name}")


def q2c(bm3d_mode: str = "hard"):
    """Improve the baseline by allowing more flexibility in the hyper-parameters.

    The modification is an increasing per-iteration rho schedule (textbook PnP
    continuation): rho ramps geometrically up from start_factor*rho_end to
    rho_end while sigma_d stays fixed at its Q2b optimum. Low early rho avoids
    over-weighting the blurry observation; high final rho gives strong late data
    fidelity. Reuses the q2b tune->report machinery via its q2c flag; tunes
    BM3D_Q2C_GRID and writes q2_deblurring_BM3D_Q2C.{txt,png}.
    """
    return q2b(denoiser_name="bm3d", bm3d_mode=bm3d_mode, q2c=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADMM-PnP deblurring: tune on a diverse subset, report on all 8.")
    parser.add_argument("--denoiser", choices=["bm3d", "bf"], default="bm3d",
                        help="Which denoiser to plug into ADMM-PnP.")
    parser.add_argument("--bm3d-mode", choices=["hard", "full"], default="hard",
                        help="BM3D stages inside ADMM. 'hard' (default) = "
                             "hard-thresholding only (~2x faster, standard PnP). "
                             "'full' = HT + Wiener.")
    parser.add_argument("--q2c", action="store_true",
                        help="Q2c improved BM3D: add an increasing per-iteration "
                             "rho schedule (PnP continuation) with sigma_d fixed "
                             "at its Q2b optimum. Forces BM3D, tunes BM3D_Q2C_GRID, "
                             "and writes q2_deblurring_BM3D_Q2C.{txt,png}.")
    args = parser.parse_args()
    if args.q2c:
        q2c(bm3d_mode=args.bm3d_mode)
    else:
        q2b(denoiser_name=args.denoiser, bm3d_mode=args.bm3d_mode)

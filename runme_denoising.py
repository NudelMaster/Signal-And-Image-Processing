"""
runme_denoising.py  –  Q1: Bilateral Filter and BM3D denoising
================================================
Noise model:  y = x_gt + e,   e ~ N(0, sigma_e^2 * I),  sigma_e = 0.1
Random seed:  0
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils import load_all_images, psnr, add_gaussian_noise
from bilateral_filter import bilateral_filter
from bm3d import bm3d


# ── Configuration ─────────────────────────────────────────────────────────────
SIGMA_E = 0.1
SEED    = 0
OUT_DIR = Path("outputs") / "denoising"
np.random.seed(SEED)

VISUAL_NAMES = ["cameraman", "lena", "barbara"]




def main(
    bf_sigma_s: list[float] | None = None,
    bf_sigma_r: list[float] | None = None,
):
    """Run BF denoising for every (sigma_s, sigma_r) pair and BM3D on all test images.


    All combinations of bf_sigma_s × bf_sigma_r are evaluated. Results are
    printed and saved as a single PSNR table (.txt). The visual plot shows only
    the pair that achieved the highest average PSNR.


    :param bf_sigma_s: Spatial std values for the bilateral filter.
    :param bf_sigma_r: Range std values for the bilateral filter.
                    All combinations of the two lists are evaluated.
    """

    OUT_DIR.mkdir(parents=True, exist_ok=True)


    pairs   = list(itertools.product(bf_sigma_s, bf_sigma_r))
    n_pairs = len(pairs)
    images  = load_all_images()
    print("Loaded images")

    # ── BM3D once, BF for every pair ──────────────────────────────────────────
    base = {}
    for name, x_gt in tqdm(images.items(), desc = "Processing BM3D of images"):
        y = add_gaussian_noise(x_gt, SIGMA_E)
        x_bm3d = bm3d(z=y, sigma_psd=SIGMA_E)
        base[name] = {"gt": x_gt, "noisy": y, "bm3d": x_bm3d, "bf": []}

    
    for sigma_s, sigma_r in tqdm(pairs, desc="Processing all pairs for BF"):
        print(f"Calculate BF (sigma_s={sigma_s}, sigma_r={sigma_r})")
        for d in base.values():
            d["bf"].append(bilateral_filter(d["noisy"], sigma_s, sigma_r))


    # ── Build table ───────────────────────────────────────────────────────────
    bf_labels = [f"BF(s={s},r={r})" for s, r in pairs]
    col_w     = max(16, *(len(lbl) + 2 for lbl in bf_labels))
    sep       = "-" * (12 + 12 + n_pairs * col_w + 12)


    header = (f"{'Image':<12}{'PSNR_noisy':>12}" +
              "".join(f"{lbl:>{col_w}}" for lbl in bf_labels) +
              f"{'PSNR_BM3D':>12}")


    rows_noisy = []
    rows_bf    = [[] for _ in range(n_pairs)]
    rows_bm3d  = []
    data_lines = []


    for name, d in base.items():
        p_noisy = psnr(d["noisy"], d["gt"])
        p_bm3d  = psnr(d["bm3d"],  d["gt"])
        rows_noisy.append(p_noisy)
        rows_bm3d.append(p_bm3d)


        line = f"{name:<12}{p_noisy:>12.2f}"
        for i, x_bf in enumerate(d["bf"]):
            p_bf = psnr(x_bf, d["gt"])
            rows_bf[i].append(p_bf)
            line += f"{p_bf:>{col_w}.2f}"
        line += f"{p_bm3d:>12.2f}"
        data_lines.append(line)


    avg_bf   = [np.mean(rows_bf[i]) for i in range(n_pairs)]
    avg_line = (f"{'Average':<12}{np.mean(rows_noisy):>12.2f}" +
                "".join(f"{avg_bf[i]:>{col_w}.2f}" for i in range(n_pairs)) +
                f"{np.mean(rows_bm3d):>12.2f}")


    table = "\n".join([header, sep] + data_lines + [sep, avg_line])
    table += f"\n\nBM3D:  sigma={SIGMA_E}"


    print("\n" + table)


    txt_path = OUT_DIR / "q1_denoising_results.txt"
    txt_path.write_text(table + "\n", encoding="utf-8")
    print(f"Table saved to {txt_path.name}")


    # ── Plot: worst and best BF pair ─────────────────────────────────────────
    best_i          = int(np.argmax(avg_bf))
    worst_i         = int(np.argmin(avg_bf))
    best_s,  best_r  = pairs[best_i]
    worst_s, worst_r = pairs[worst_i]


    fig, axes = plt.subplots(len(VISUAL_NAMES), 5,
                             figsize=(20, 4 * len(VISUAL_NAMES)))
    fig.suptitle(
        f"Q1 Denoising  (σ_e={SIGMA_E})  "
        f"worst BF: σ_s={worst_s}, σ_r={worst_r}  |  best BF: σ_s={best_s}, σ_r={best_r}",
        fontsize=12, y=1.01,
    )


    for row, name in enumerate(VISUAL_NAMES):
        d       = base[name]
        p_noisy  = psnr(d["noisy"],        d["gt"])
        p_worst  = psnr(d["bf"][worst_i],  d["gt"])
        p_best   = psnr(d["bf"][best_i],   d["gt"])
        p_bm3d   = psnr(d["bm3d"],         d["gt"])


        for col, (arr, title) in enumerate([
            (d["gt"],           "Ground truth"),
            (d["noisy"],        f"Noisy\nPSNR={p_noisy:.2f} dB"),
            (d["bf"][worst_i],  f"BF worst (σ_s={worst_s}, σ_r={worst_r})\nPSNR={p_worst:.2f} dB"),
            (d["bf"][best_i],   f"BF best (σ_s={best_s}, σ_r={best_r})\nPSNR={p_best:.2f} dB"),
            (d["bm3d"],         f"BM3D\nPSNR={p_bm3d:.2f} dB"),
        ]):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{name} – {title}", fontsize=9)
            ax.axis("off")


    plt.tight_layout()
    fname = OUT_DIR / "q1_denoising_results.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visual saved to {fname.name}")




if __name__ == "__main__":
    bf_sigma_s = [0.5, 1.0, 3.0]
    bf_sigma_r = [0.15, 0.3, 0.5, 0.75]
    main(bf_sigma_s=bf_sigma_s, bf_sigma_r=bf_sigma_r)

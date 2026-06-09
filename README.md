# Signal & Image Processing — Project 1

Bilateral Filter, BM3D, and ADMM Plug-and-Play (PnP) for three classical image
reconstruction tasks: **denoising**, **deblurring**, and **inpainting**.

All experiments run on a shared set of 8 grayscale test images (`test_set/`,
3×256² + 5×512²), loaded and normalised to `[0, 1]`. Each question has a
`runme_*.py` driver that produces a PSNR table (`.txt`) and a visual comparison
(`.png`) under `outputs/<question>/`.

## Questions

| Q | Task | Driver | Denoiser(s) | Noise / observation |
|---|------|--------|-------------|---------------------|
| **Q1** | Denoising | `runme_denoising.py` | Bilateral Filter (σ_s×σ_r sweep) vs BM3D | `y = x + e`, σ_e = 0.1 |
| **Q2** | Deblurring (ADMM-PnP) | `runme_deblurring.py` | BM3D **or** BF (flag) | `y = x∗k + e`, σ_e = 0.01, `k(i,j)=1/(1+i²+j²)`, i,j=−7..7 |
| **Q3** | Inpainting (ADMM-PnP) | `runme_inpainting.py` | BM3D | `y = x_gt(M)`, M = random 20% of pixels, **no noise** |

- **Q2a / Q3a** — ADMM-PnP derivations (closed-form data-fidelity update) are
  written up in `main.tex` and documented in the `admm_pnp.py` module docstring.
- **Q2c** — improvement over the Q2b baseline: an increasing geometric ρ schedule
  (PnP continuation), σ_d fixed at the Q2b optimum.

## Setup

Install the dependencies (Python 3.11+) into a virtualenv:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies: `numpy`, `Pillow`, `matplotlib`, `tqdm`, `bm3d`, and `torch`
(Q3 only, for the median-inpainting initialisation). Pinned versions are in
`requirements.txt`.

## Running

Each driver runs one **tune → report** pass: grid-search a small hyper-parameter
grid on 5 diverse images, pick the single best config (per-image tuning is
forbidden by the spec), apply it to all 8 images, and save the table + figure.

```bash
.venv/bin/python runme_denoising.py                       # Q1            (~minutes)
.venv/bin/python runme_deblurring.py --denoiser bm3d      # Q2b BM3D      (~9 min)
.venv/bin/python runme_deblurring.py --denoiser bf        # Q2b BF        (~9 min)
.venv/bin/python runme_deblurring.py --q2c                # Q2c ρ-schedule (~20 min)
.venv/bin/python runme_inpainting.py                      # Q3 BM3D       (~25–30 min)
```

Optional flags: `--bm3d-mode {hard,full}` (default `hard` = hard-thresholding
only; `full` adds Wiener, ~2× slower and ~1 dB worse at σ_e=0.01). Runs are
multi-minute because BM3D is invoked once per ADMM iteration.

## Results (average PSNR over the 8 images)

| Experiment | Input PSNR_y | Output PSNR_x̂ | Best config |
|---|---|---|---|
| Q1 BF (best)   | 20.01 | 27.29 | σ_s=1.0, σ_r=0.5 |
| Q1 BM3D        | 20.01 | 30.50 | σ=0.1 |
| Q2b deblur BM3D | 24.94 | 30.76 | n_iter=12, ρ=0.05, σ_d=0.02 |
| Q2b deblur BF   | 24.94 | 28.53 | n_iter=10, ρ=0.05, σ_s=1.5, σ_r=0.05 |
| Q2c deblur BM3D | 24.94 | 30.90 | n_iter=16, ρ=0.030→0.1 (×0.3), σ_d=0.02 (+0.14 dB) |
| Q3 inpaint BM3D | 6.62  | 28.15 | n_iter=20, ρ=0.02, σ_d=0.04 |

PSNR uses a **7-pixel crop in Q2** (cyclic convolution corrupts the boundary
ring; the spec says to ignore boundary pixels) and **no crop in Q1/Q3**.
Observations are reproducible: seed 0 is set once before generating noise (Q1/Q2)
or drawing the 20% indices (Q3), iterating images in canonical order.

## Architecture

Layered: pure operators → ADMM engine → per-question drivers.

```
utils.py              Image I/O, crop-aware psnr(), blur kernel, cyclic-conv FFT helpers
bilateral_filter.py   Bilateral filter denoiser
bm3d (pip)            BM3D denoiser
admm_pnp.py           PnP-ADMM solver:
                        • admm_pnp_deblur     — z-update is a closed-form FFT solve
                        • admm_pnp_inpainting — z-update is an elementwise mask solve
                      Both x-updates call the plugged-in denoiser. Optional
                      rho_schedule / sigma_d_schedule with scaled-dual rescaling.
median_inpainting.py  Prior-free Q3 initialisation (vendored from tirer-lab/CM4IR).
runme_denoising.py    Q1 driver
runme_deblurring.py   Q2 driver (Q2b/Q2c)
runme_inpainting.py   Q3 driver
test_set/             8 grayscale test images
outputs/              Generated PSNR tables (.txt) and figures (.png)
```

Both denoisers are wrapped into a uniform `denoiser(img, sigma)` callable so they
are interchangeable inside ADMM. For deblurring, the frequency-domain constants
(Aᵀy, |K̂|²) are precomputed once, so each ADMM iteration is one FFT + one iFFT.

## Notes

- `median_inpainting.py` is vendored from
  [tirer-lab/CM4IR](https://github.com/tirer-lab/CM4IR); its attribution header
  is retained, and the report includes it in the code appendix.
- Convolution is cyclic throughout (FFT); there is no spatial-domain path.

"""
admm_pnp.py  –  ADMM Plug-and-Play framework


Supports two forward operators:
  • 'deblur'   : A x = x * k  (cyclic convolution)


ADMM-PnP update equations
==========================
Minimise:   (1/2)||Ax - y|| ** 2 + beta * phi(x)


Variable Splitting split:
  L(x,z) = min_{x,z}(1/2)||Az-y|| ** 2  + beta * phi(x) s.t x = z

Lagrange Multiplier with scaled dual variable u
L_rho(x,z,u) = (1/2)||Az-y|| ** 2  + beta * phi(x) + (rho/2)||x-z+u|| ** 2 - rho/2 * ||u|| ** 2

Step 1 – z-update (data-fidelity, closed form):
  z_k ← argmin_z  (1/2)||Az-y|| ** 2 + (rho/2)||x_{k-1}-z+u_{k-1}|| ** 2


Step 2 – x-update (denoising / proximal):
  x_k ← argmin_x rho/2||x - z_k + u_{k-1}|| ** 2  ≡  Denoiser(z_k - u_{k-1}, sigma=sqrt(beta/rho))


Step 3 – u-update (dual ascent):
  u_k ← u_{k-1} + x_k - z_k


Deblurring z-update (FFT closed form)
--------------------------------------
With cyclic convolution  A = F⁻¹ diag(K̂) F:
  data fidelity equation (AtA + ρI) z = Aty + ρ(x+u)
  z = F⁻¹[ (K̂* Ŷ + ρ(X + Û)) / (|K̂|² + ρ) ]

"""

import numpy as np
from utils import blur_kernel_to_fd
# ── z-update: deblurring ─────────────────────────────────────────────────────


def z_update_deblur(x, u, rho, aty_freq, blur_kernel_fd_abs2):
    """Closed-form z-update for deblurring via FFT.

    Receives the precomputed frequency-domain quantities that do not change
    across ADMM iterations (Aᵀy in freq domain and |K̂|²), so each call costs
    one fft2 + one ifft2 instead of four.
    """
    rhs_freq = aty_freq + rho * np.fft.fft2(x + u)        # K̂* Ŷ + ρ F(x+u)
    z = np.real(np.fft.ifft2(rhs_freq / (blur_kernel_fd_abs2 + rho)))
    return z









# ── ADMM-PnP: deblurring ─────────────────────────────────────────────────────


def admm_pnp_deblur(y, blur_kernel, denoiser, sigma_d, rho,
                    n_iter=30, x_init=None, verbose=False,
                    rho_schedule=None, sigma_d_schedule=None):
    """
    ADMM-PnP for deblurring.


    :param y                : blurry + noisy image
    :param blur_kernel      : blur kernel (spatial domain, small array)
    :param denoiser         : callable denoiser(img, sigma) -> denoised img
    :param sigma_d          : denoiser noise level = sqrt(beta/rho)
    :param rho              : ADMM penalty parameter
    :param n_iter           : number of ADMM iterations
    :param x_init           : initialisation (default: y)
    :param rho_schedule     : optional list of (iter, rho) tuples for adaptive rho
    :param sigma_d_schedule : optional list of (iter, sigma_d) tuples for
                              continuation in the denoiser noise level
                              (standard PnP trick: start large, decrease)
    """


    blur_kernel_fd      = blur_kernel_to_fd(blur_kernel, y.shape)
    blur_kernel_fd_abs2 = np.abs(blur_kernel_fd) ** 2
    aty_freq            = np.conj(blur_kernel_fd) * np.fft.fft2(y)   # K̂* Ŷ — constant across iters


    x = y.copy() if x_init is None else x_init.copy()
    z = x.copy()
    u = np.zeros_like(x)


    rho_sched   = dict(rho_schedule or [])
    sigma_sched = dict(sigma_d_schedule or [])


    for it in range(n_iter):
        if it in rho_sched:
            new_rho = rho_sched[it]
            # scaled dual u = (unscaled multiplier)/rho, so when rho changes the
            # scaled dual must be rescaled to keep the multiplier consistent.
            u = u * (rho / new_rho)
            rho = new_rho
        if it in sigma_sched:
            sigma_d = sigma_sched[it]


        z = z_update_deblur(x, u, rho, aty_freq, blur_kernel_fd_abs2)
        x = denoiser(np.clip(z - u, 0, 1), sigma_d)
        u = u + x - z


        if verbose:
            print(f"  iter {it+1:3d}/{n_iter}")


    return np.clip(x, 0, 1)


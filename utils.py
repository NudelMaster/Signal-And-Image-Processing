"""
utils.py - shared helpers for Project 1
"""
import numpy as np
from PIL import Image
from pathlib import Path


# ── Image I/O ────────────────────────────────────────────────────────────────


_TEST_SET = Path(__file__).parent / "test_set"


IMAGE_PATHS = {
    "cameraman": _TEST_SET / "1_Cameraman256.png",
    "house":     _TEST_SET / "2_house.png",
    "peppers":   _TEST_SET / "3_peppers256.png",
    "lena":      _TEST_SET / "4_Lena512.png",
    "barbara":   _TEST_SET / "5_barbara.png",
    "boat":      _TEST_SET / "6_boat.png",
    "hill":      _TEST_SET / "7_hill.png",
    "couple":    _TEST_SET / "8_couple.png",
}


def load_image(path: str) -> np.ndarray:
    """Load a grayscale image and normalise to [0, 1]."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64) / 255.0


def load_all_images() -> dict:
    return {name: load_image(path) for name, path in IMAGE_PATHS.items()}


# ── PSNR ────────────────────────────────────────────────────────────


def psnr(x_hat: np.ndarray, x_gt: np.ndarray, crop: int = 0) -> float:
    """PSNR with r=1 (images normalised to [0,1]).

    crop: pixels to remove from each border before computing MSE. Used in
    deblurring where cyclic-conv assumption makes boundary pixels unreliable
    (project PDF: "ignore the boundary pixels of the reconstruction").
    """
    if crop > 0:
        x_hat = x_hat[crop:-crop, crop:-crop]
        x_gt = x_gt[crop:-crop, crop:-crop]
    mse = np.mean((x_hat - x_gt) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10(1.0 / mse)


# ── Noisy data generation ──────────────────────────────────────────────────────────


def add_gaussian_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    return x + np.random.normal(0, sigma, x.shape)


# ── Kernel builder ────────────────────────────────────────────────────────────


def make_blur_kernel(half: int = 7) -> np.ndarray:
    """k(i,j) = 1/(1+i²+j²), i,j in [-half, half], normalised to sum=1."""
    r = np.arange(-half, half + 1)
    I, J = np.meshgrid(r, r, indexing="ij")
    k = 1.0 / (1.0 + I**2 + J**2)
    # normalize
    return k / k.sum()


# ── Cyclic convolution via FFT ────────────────────────────────────────────────


def blur_kernel_to_fd(blur_kernel: np.ndarray, shape: tuple) -> np.ndarray:
    """Pad and FFT a small blur kernel to its frequency-domain representation."""
    # Image shape
    h, w = shape
    # Kernel Shape
    kh, kw = blur_kernel.shape

    padded = np.zeros(shape)
    padded[:kh, :kw] = blur_kernel
    # centre so that k[0,0] is the DC component
    padded = np.roll(padded, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(padded)


def fft_convolve(x: np.ndarray, blur_kernel_fd: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.fft2(x) * blur_kernel_fd))


def fft_convolve_blur_kernel(x: np.ndarray, blur_kernel: np.ndarray) -> np.ndarray:
    blur_kernel_fd = blur_kernel_to_fd(blur_kernel, x.shape)
    return fft_convolve(x, blur_kernel_fd)

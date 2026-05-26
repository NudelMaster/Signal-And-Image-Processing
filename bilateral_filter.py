import numpy as np
def bilateral_filter(img: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """
    Bilateral filter using the efficient Gaussian-range approximation.


    Parameters
    ----------
    img     : 2-D float array in [0, 1]
    sigma_s : spatial standard deviation (pixels)
    sigma_r : range (intensity) standard deviation


    Returns
    -------
    Filtered image clipped to [0, 1].


    Algorithm
    ---------
    For each pixel p:
        w(p,q) = G_s(||p-q||) * G_r(|I(p)-I(q)|)
        I_out(p) = Σ_q w(p,q)*I(q) / Σ_q w(p,q)


    We implement this with a truncated window of radius = ceil(3*sigma_s)
    and vectorised NumPy operations, which is fast enough for 512×512 images.
    """
    radius = int(np.ceil(3 * sigma_s))
    h, w = img.shape


    # Pad image (reflect) so we don't need boundary checks
    img_pad = np.pad(img, radius, mode="reflect")


    # Pre-compute spatial Gaussian weights for the window
    r = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(r, r, indexing="ij")
    spatial_kernel = np.exp(-(X**2 + Y**2) / (2 * sigma_s**2))  # (2r+1, 2r+1)
    

    out = np.zeros_like(img)
    norm = np.zeros_like(img)
    
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            # Shifted image patch
            shifted = img_pad[radius + di: radius + di + h,
                               radius + dj: radius + dj + w]
            # Range weight
            range_w = np.exp(-((shifted - img) ** 2) / (2 * sigma_r**2))
            # norm factor
            norm += spatial_kernel[di + radius, dj + radius] * range_w
            # Combined weight
            w_ij = spatial_kernel[di + radius, dj + radius] * range_w
            out += w_ij * shifted


    out = out / (norm + 1e-12)
    return np.clip(out, 0, 1)

import numpy as np
from scipy.ndimage import gaussian_filter


def generate(N, S, W=300, sigma1=4, sigma2=2, t=0.01, bins=64):
    z = np.zeros((N, S, 2))
    for n in range(N):
        z[n, 0] = np.random.uniform(0, W, size=(2))
        for s in range(S - 1):
            d_1 = np.random.normal(0, sigma1)
            d_2 = np.random.normal(0, sigma1)
            z[n, s + 1, 0] = (z[n, s, 0] + d_1) % W
            z[n, s + 1, 1] = (z[n, s, 1] + d_2) % W

    z_r = z.reshape(N * S, 2)
    H, _, _ = np.histogram2d(z_r[:, 0], z_r[:, 1], bins=bins)

    G = gaussian_filter(H, sigma2)
    G[G < t] = 0

    return G / G.max()


def generate_ob_hir(n_images, N1=100, S1=30, N2=250, S2=10, W=300, sigma1=4, sigma2=2, t=0.01, bins=32):
    images = np.zeros((n_images, bins, bins))
    for n in range(n_images // 2):
        images[n] = generate(N1, S1, W, sigma1, sigma2, t, bins)

    for n in range(n_images // 2):
        images[n + n_images // 2] = generate(N2, S2, W, sigma1, sigma2, t, bins)

    return images

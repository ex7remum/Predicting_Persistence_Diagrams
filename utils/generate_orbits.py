import numpy as np
import torch


def generate_orbit(point_0, r, n):
    X = np.zeros((n, 2))

    xcur, ycur = point_0[0], point_0[1]

    for idx in range(n):
        xcur = (xcur + r * ycur * (1. - ycur)) % 1
        ycur = (ycur + r * xcur * (1. - xcur)) % 1
        X[idx, :] = [xcur, ycur]

    return X


def generate_orbits(m, rs=[2.5, 3.5, 4.0, 4.1, 4.3], n=1000, random_seed=0):
    # initial points
    rng = np.random.default_rng(random_seed)
    points_0 = rng.uniform(size=(len(rs), m, 2))

    orbits = np.zeros((len(rs), m, n, 2))

    for i, r in enumerate(rs):
        for j, point_0 in enumerate(points_0[i]):
            orbits[i, j] = generate_orbit(point_0, r, n)

    return orbits


def get_orbit_dataset(m_over, m, n, rr, random_seed, device):
    cdist_batched = torch.vmap(torch.cdist)
    quantile_batched = torch.vmap(torch.vmap(torch.quantile))
    X_raw = torch.Tensor(generate_orbits(m_over, n=n, random_seed=random_seed)).to(device)
    D = torch.zeros((rr, m_over, int(n * (n - 1) / 2))).to(device)

    distances = cdist_batched(X_raw, X_raw).to(device)
    triu_idx = torch.triu_indices(n, n, 1, device=device)

    for i in range(rr):
        for j in range(m_over):
            D[i, j, :] = distances[i, j][triu_idx[0], triu_idx[1]]

    Q = quantile_batched(D, q=0.05)
    Q_sorted, sort_idx = torch.sort(Q, dim=1)

    rng = np.random.default_rng(random_seed)
    random_seeds = rng.integers(0, 1e6, size=rr)

    tau = 0.025
    outlier_idx = Q_sorted.cpu().numpy() < tau
    start_idx = outlier_idx.sum(axis=1)

    m_select = m  # 10
    X_sorted = torch.zeros_like(X_raw)
    X = np.zeros((rr, m_select, n, 2))

    for i in range(rr):
        X_sorted[i] = X_raw[i][sort_idx[i]]

    X_sorted_np = X_sorted.cpu().numpy()

    select_idx = np.zeros((rr, m_select)).astype(int)

    for i in range(rr):
        select_idx[i] = rng.choice(np.arange(start_idx[i], m_over), m_select, replace=False)
        X[i] = X_sorted_np[i][select_idx[i]]

    return X

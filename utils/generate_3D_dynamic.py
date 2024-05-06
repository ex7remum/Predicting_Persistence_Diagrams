import numpy as np


def K(x):
    if x == 0:
        return 1
    else:
        return (1 - np.exp(-x)) / x


def generate_point(point_0, M0, n):
    M1, M2, M3 = 1.0, 4.0, 4.0
    X = np.zeros((n, 3))

    xcur, ycur, zcur = point_0[0], point_0[1], point_0[2]

    for idx in range(n):
        X[idx, :] = [xcur, ycur, zcur]
        xnext = (M0 * xcur * np.exp(-ycur)) / (1 + xcur * max(np.exp(-ycur), K(zcur) * K(ycur)))
        ynext = M1 * xcur * ycur * np.exp(-zcur) * K(ycur) * K(M3 * ycur * zcur)
        znext = M2 * ycur * zcur
        xcur, ycur, zcur = xnext, ynext, znext

    return X


def generate_3D_dynamic(m_over=1500, m=1000, rs=[3.0, 3.3, 3.48, 3.54, 3.57, 3.532, 3.571, 3.3701, 3.4001],
                        n=500, random_seed=0):
    # initial points
    rng = np.random.default_rng(random_seed)
    points_0 = rng.uniform(size=(len(rs), m_over, 3))

    points = np.zeros((len(rs), m, n, 3))

    for i, M0 in enumerate(rs):
        j = 0
        for point_0 in points_0[i]:
            pc = generate_point(point_0, M0, n)
            # filter point clouds that converge to one point
            if not (pc[-1, 0] == pc[-2, 0] and pc[-2, 0] == pc[-3, 0]):
                points[i, j] = pc
                j += 1
            if j >= m:
                break
        assert j >= m

    return points

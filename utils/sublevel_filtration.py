import gudhi as gd
import numpy as np
import torch


def sublevel_filtration(image, num_channels=1, dimensions=[32, 32]):
    pd = np.zeros((0, 3))
    for channel in range(num_channels):
        layer = image[channel]
        cc_density_crater = gd.CubicalComplex(
            dimensions=dimensions,
            top_dimensional_cells=layer.flatten()
        )
        cc_density_crater.compute_persistence()
        diagram = cc_density_crater.persistence()

        for k, pair in diagram:
            if not np.isinf(pair[1]):
                cur = np.zeros((1, 3))
                cur[0, 0], cur[0, 1], cur[0, 2] = pair[0], pair[1], k
                pd = np.concatenate((pd, cur))
    return pd

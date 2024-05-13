import gudhi as gd
import numpy as np


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

        critical_pairs = cc_density_crater.cofaces_of_persistence_pairs()

        # get critical pixels corresponding to critical simplices
        try:
            bpx0 = [critical_pairs[0][0][i][0] for i in range(len(critical_pairs[0][0]))]
            dpx0 = [critical_pairs[0][0][i][1] for i in range(len(critical_pairs[0][0]))]
        except IndexError:
            bpx0 = []
            dpx0 = []

        mask_pd_0 = np.zeros_like(layer)[None, :].repeat(len(bpx0), axis=0)

        for i, (b, d) in enumerate(zip(bpx0, dpx0)):
            b_i, b_j = b // dimensions[0], b % dimensions[0]
            d_i, d_j = d // dimensions[0], d % dimensions[0]
            mask_pd_0[i, b_i, b_j] = 1
            mask_pd_0[i, d_i, d_j] = 1

        try:
            bpx1 = [critical_pairs[0][1][i][0] for i in range(len(critical_pairs[0][1]))]
            dpx1 = [critical_pairs[0][1][i][1] for i in range(len(critical_pairs[0][1]))]
        except IndexError:
            bpx1 = []
            dpx1 = []

        mask_pd_1 = np.zeros_like(layer)[None, :].repeat(len(bpx1), axis=0)
        for i, (b, d) in enumerate(zip(bpx1, dpx1)):
            b_i, b_j = b // dimensions[0], b % dimensions[0]
            d_i, d_j = d // dimensions[0], d % dimensions[0]
            mask_pd_1[i, b_i, b_j] = 1
            mask_pd_1[i, d_i, d_j] = 1

        for k, pair in diagram:
            if not np.isinf(pair[1]):
                cur = np.zeros((1, 3))
                cur[0, 0], cur[0, 1], cur[0, 2] = pair[0], pair[1], k
                pd = np.concatenate((pd, cur))
    return pd, mask_pd_0, mask_pd_1

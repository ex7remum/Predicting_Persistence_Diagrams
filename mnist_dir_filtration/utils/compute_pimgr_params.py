import numpy as np
from scipy.spatial import distance_matrix


def compute_pimgr_parameters(diagrams):
    min_b, max_b, min_p, max_p = 0., 0., 0., 0.
    sigma = 0.
    n_total = 0
    for pd in diagrams:
        pd = pd.numpy()
        min_b = min(min_b, np.min(pd[..., 0]))
        max_b = max(max_b, np.max(pd[..., 0]))
        min_p = min(min_p, np.min(pd[..., 1] - pd[..., 0]))
        max_p = max(max_p, np.max(pd[..., 1] - pd[..., 0]))
        
        pairwise_distances = np.triu(distance_matrix(pd, pd)).flatten()
        pairwise_distances = pairwise_distances[pairwise_distances > 0]
        if len(pairwise_distances) != 0:
            sigma += np.quantile(pairwise_distances, q=0.2)
            n_total += 1
            
    im_range = [min_b, max_b, min_p, max_p]
    sigma /= n_total
        
    return sigma, im_range
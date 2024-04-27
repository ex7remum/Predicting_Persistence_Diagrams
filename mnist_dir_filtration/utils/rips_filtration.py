from gph import ripser_parallel
import numpy as np


def rips_filtration(X, maxdim=1):
    result = ripser_parallel(X, maxdim=maxdim, n_threads=-1)
    pd = np.zeros((0, 3))
    for dim in range(maxdim + 1):
        diagram_k = result["dgms"][dim]
        diagram_k = np.concatenate((diagram_k, dim * np.ones((diagram_k.shape[0], 1))), axis=1)
        pd = np.concatenate((pd, diagram_k))

    return pd

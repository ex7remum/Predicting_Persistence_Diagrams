import torch
import torch.nn as nn
from torch.nn import Module
from ot.sliced import sliced_wasserstein_distance, get_random_projections
from ot.lp import wasserstein_1d

sliced_wasserstein_distance_batched = torch.vmap(sliced_wasserstein_distance, randomness="same")

def weighted_sliced_wasserstein_distance_batched(X, Y, a=None, b=None, P=None, n_projections=64, p=2, device=None, random_seed=None):
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must have the same number of batch elements, {} and {} respectively given".format(X.shape[0], Y.shape[0]))
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "X and Y must have the same number of dimensions, {} and {} respectively given".format(X.shape[-1], Y.shape[-1]))
    
    (n_batch, n, d), m = X.shape, Y.shape[1]
    
    if a==None:
        a = torch.ones((n_batch, n), device=device) / n
    if b==None:
        b = torch.ones((n_batch, m), device=device) / m
    
    if P==None:
        P = torch.tensor(get_random_projections(d, n_projections, random_seed), device=device, dtype=torch.float)
    
    X_P = X @ P
    Y_P = Y @ P
    
    w1d = wasserstein_1d(X_P.swapaxes(0,1), Y_P.swapaxes(0,1), a.swapaxes(0,1), b.swapaxes(0,1), p)
    
    return (torch.mean(w1d, dim=1) + 1e-6) ** (1/p)

def get_persistence_weights(X, q=None, max_persistence=None):
    
    if q==None:
        n_batch, n = X.shape[0], X.shape[1]
        weights = torch.ones((n_batch, n)) / n
    
    elif q in [1, 2]:
        persistence = (X[:,:,1] - X[:,:,0]) ** q
        if max_persistence==None:
            max_persistence, _ = torch.max(persistence, axis=1, keepdims=True)
        persistence_normed_max = persistence / max_persistence
        weights = persistence_normed_max / torch.sum(persistence_normed_max, axis=1, keepdims=True)
        
    else:
        raise ValueError("Q should be None (for uniform weighting), 1 or 2 (for q-power persistence weighting).")
    
    return weights


class LossFunction(nn.Module):
    
    def __init__(self, reduce):
        super().__init__()
        
        identity = lambda x: x
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if reduce==None:
            self.reduce_fn = identity
        elif reduce=="mean":
            self.reduce_fn = torch.mean
        elif reduce=="sum":
            self.reduce_fn = torch.sum
        else:
            raise ValueError("Reduce should be None, 'mean' or 'sum'.")
        
    def reduce(self, arr):
        return self.reduce_fn(arr)

    
class SlicedWassersteinLoss(LossFunction):
    
    def __init__(self, n_projections=1024, p=2, reduce="mean", random_seed=None):
        super().__init__(reduce)
        self.n_projections = n_projections
        self.p = p
        self.random_seed = random_seed
        
    def forward(self, X, Y):
        return self.reduce(sliced_wasserstein_distance_batched(X, Y, n_projections=self.n_projections, p=self.p, seed=self.random_seed))
    

class WeightedSlicedWassersteinLoss(LossFunction):
    
    def __init__(self, n_projections=1024, p=2, q=1, reduce="mean", random_seed=None):
        super().__init__(reduce)
        self.n_projections = n_projections
        self.p = p
        self.q = q
        self.random_seed = random_seed
        
    def forward(self, X, Y, a=None, b=None):
        return self.reduce(weighted_sliced_wasserstein_distance_batched(X, Y, a, b, n_projections=self.n_projections, p=self.p, random_seed=self.random_seed))

    
class PersistenceWeightedSlicedWassersteinLoss(LossFunction):
    
    def __init__(self, n_projections=1024, p=2, q=1, max_persistence=None, reduce="mean", random_seed=None):
        super().__init__(reduce)
        self.n_projections = n_projections
        self.p = p
        self.q = q
        self.max_persistence = max_persistence
        self.random_seed = random_seed
        
    def forward(self, X, Y):
        a = get_persistence_weights(X, self.q, self.max_persistence).to(self.device)
        b = get_persistence_weights(Y, self.q, self.max_persistence).to(self.device)
        return self.reduce(weighted_sliced_wasserstein_distance_batched(X, Y, a, b, n_projections=self.n_projections, p=self.p, device=self.device, random_seed=self.random_seed))


class ChamferLoss(LossFunction):
    
    def __init__(self, reduce="mean"):
        super().__init__(reduce)
        
    def forward(self, X, Y):
        A = torch.cdist(X, Y, p=2)
        X_inf, _ = torch.min(A, dim=2)
        Y_inf, _ = torch.min(A, dim=1)
        batch_dist = (torch.mean(X_inf, dim=1) + torch.mean(Y_inf, dim=1)) / 2
        return self.reduce(batch_dist)


class HausdorffLoss(LossFunction):
    
    def __init__(self, reduce="mean"):
        super().__init__(reduce)
        
    def forward(self, X, Y):
        A = torch.cdist(X, Y, p=2)
        X_inf, _ = torch.min(A, dim=2)
        Y_inf, _ = torch.min(A, dim=1)
        X_supinf, _ = torch.max(X_inf, dim=1)
        Y_supinf, _ = torch.max(Y_inf, dim=1)
        batch_dist = torch.max(X_supinf, Y_supinf)
        return self.reduce(batch_dist)
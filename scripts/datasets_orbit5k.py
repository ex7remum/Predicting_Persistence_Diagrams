import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import gudhi as gd
from gph import ripser_parallel

from itertools import product
from tqdm import tqdm

#from gtda.diagrams import PersistenceImage as PersistenceImageGiotto, PersistenceLandscape as PersistenceLandscapeGiotto
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImage

if __name__ == "__main__":
    
    # constants
    random_seed = 0
    device = torch.device("cuda:0")
    
    
    
    # - generate dataset
    # - compute persistent diagrams of a given filtration [shared, within domain]
    # - compute given vectorizations of persistence diagrams [shared]
    
    
    # ORBIT5K parameters
    # - filtration [VR, alpha]
    # - vectorization
    #   - persistent image (gamma, dims, range)
    #   - tropical coordinates
    
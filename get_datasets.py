from ripser import Rips
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImageGudhi
from gtda.homology import VietorisRipsPersistence
from scipy.spatial import distance_matrix

from tqdm import tqdm
import h5py

class MyDataset(Dataset):

    def __init__(self, pd, y, data, PI = None, transform=None):
        self.pd = pd
        self.y = y
        self.data = data
        self.PI = PI

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.PI is None:
            return self.pd[idx], self.y[idx], self.data[idx], None
        else:
            return self.pd[idx], self.y[idx], self.data[idx], self.PI[idx]

        
def generate_image(N, S, W=300, sigma1=4, sigma2=2, t=0.01, bins=64):
    from scipy.ndimage import gaussian_filter
    z = np.zeros((N, S, 2))
    for n in range(N):
        z[n, 0] = np.random.uniform(0, W, size=(2))
        for s in range(S-1):
            d_1 = np.random.normal(0, sigma1)
            d_2 = np.random.normal(0, sigma1)
            z[n, s+1, 0] = (z[n, s, 0] + d_1) % W
            z[n, s+1, 1] = (z[n, s, 1] + d_2) % W

    z_r = z.reshape(N*S, 2)
    H, _, _ = np.histogram2d(z_r[:,0], z_r[:,1], bins=bins)
    
    G = gaussian_filter(H, sigma2)
    G[G < t] = 0
    
    return G        
            
def generate_orbit(point_0, r, n=300):

    X = np.zeros([n, 2])

    xcur, ycur = point_0[0], point_0[1]

    for idx in range(n):
        xcur = (xcur + r * ycur * (1. - ycur)) % 1
        ycur = (ycur + r * xcur * (1. - xcur)) % 1
        X[idx, :] = [xcur, ycur]

    return X

def generate_orbits(m, rs=[2.5, 3.5, 4.0, 4.1, 4.3], n=300, random_state=None):

    # m orbits, each of n points of dimension 2
    orbits = np.zeros((m * len(rs), n, 2))

    # for each r
    for j, r in enumerate(rs):

        # initial points
        points_0 = random_state.uniform(size=(m,2))

        for i, point_0 in enumerate(points_0):
            orbits[j*m + i] = generate_orbit(points_0[i], rs[j], n)

    return orbits

def conv_pd(diagrams):
    pd = np.zeros((0, 2))
    # we predict only point which correspond to one-dimensional topological features (loops)
    for k, diagram_k in enumerate(diagrams):
        if k != 0:
            diagram_k= diagram_k[~np.isinf(diagram_k).any(axis=1)] # filter infs
            if len(diagram_k) > 0:
                pd = np.concatenate((pd, diagram_k))

    return pd

def collate_fn_pc(data):

    tmp_pd, _, orbit, pimg = data[0]
    pimg = pimg.reshape(50, 50)

    n_orbit = orbit.shape[0]
    n_channels = orbit.shape[1]

    n_batch = len(data)
    n_features_pd = tmp_pd.shape[1]
    n_points_pd = max(len(pd) for pd, _, _, _ in data)
    inputs_pd = np.zeros((n_batch, n_points_pd, n_features_pd), dtype=float)

    orbits = np.zeros((n_batch, n_orbit, n_channels), dtype=float)
    
    PI = np.zeros((n_batch, pimg.shape[0], pimg.shape[1]))

    mask = np.zeros((n_batch, n_points_pd))
    labels = np.zeros(len(data))

    for i, (pd, label, orbit, pimg) in enumerate(data):
        inputs_pd[i][:len(pd)] = pd
        labels[i] = label
        mask[i][:len(pd)] = 1
        orbits[i][:n_orbit] = orbit
        PI[i] = pimg.reshape(50, 50)
    return torch.Tensor(inputs_pd), torch.Tensor(mask).long(), torch.Tensor(labels).long(), torch.Tensor(orbits), torch.Tensor(PI)

def collate_fn_img(data):

    tmp_pd, _, image, pimg = data[0]
    pimg = pimg.reshape(50, 50)
    n_image = image.shape[0]

    n_batch = len(data)
    n_features_pd = tmp_pd.shape[1]
    n_points_pd = max(len(pd) for pd, _, _, _ in data)
    inputs_pd = np.zeros((n_batch, n_points_pd, n_features_pd), dtype=float)

    images = np.zeros((n_batch, n_image, n_image), dtype=float)

    mask = np.zeros((n_batch, n_points_pd))
    labels = np.zeros(len(data))
    PI = np.zeros((n_batch, pimg.shape[0], pimg.shape[1]))

    for i, (pd, label, image, pimg) in enumerate(data):
        inputs_pd[i][:len(pd)] = pd
        labels[i] = label
        mask[i][:len(pd)] = 1
        images[i] = image
        PI[i] = pimg.reshape(50, 50)
    return torch.Tensor(inputs_pd), torch.Tensor(mask).long(), torch.Tensor(labels).long(), torch.Tensor(images), torch.Tensor(PI)

def compute_pimgr_parameters(diagrams):
    min_b, max_b, min_p, max_p = 0., 0., 0., 0.
    sigma = 0.
    n_total = 0
    for pd in diagrams:   
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
    pimgr = PersistenceImageGudhi(bandwidth=sigma, resolution=[50, 50], 
                                  weight=lambda x: (x[1])**2, im_range=im_range)
    pimgs = pimgr.fit_transform(diagrams)
    # normalize PI to lie in [0,1]
    for i, img in enumerate(pimgs):
        img /= img.max()
        pimgs[i] = img
        
    # change to (b, d-b) coords
    for i, pd in enumerate(diagrams):
        pd[..., 1] -= pd[..., 0]
        diagrams[i] = pd
        
    return sigma, im_range, pimgs, diagrams

def generate_UIUC_dataset(sub_images=30, filtration = 'rips'):
    filename = "./data/uiuc.hdf5"
    with h5py.File(filename, "r") as f:
        data = f['data'][()]
        target = f['target'][()]
    images, labels = [], []
    all_diagrams = []
    
    for i, image in enumerate(data):
        # images are too big so we only choose and save sub_images random 32x32 subimages 
        assert (1 <= sub_images <= 300)
        indices = np.random.choice(300, sub_images, replace=False)
        for val in indices:
            i_shift = val // 20
            j_shift = val % 20
            sub_img = image[i_shift * 32 : (i_shift + 1) * 32, j_shift * 32 : (j_shift + 1) * 32]
            images.append(torch.tensor(sub_img))
            labels.append(target[i])
                
    if filtration == 'rips':
        # pi-net version of calculating pds
        vr = VietorisRipsPersistence(homology_dimensions=[1])
        for image in tqdm(images):
            nx, ny = image.shape
            x, y = torch.linspace(0, 1, nx), torch.linspace(0, 1, ny)
            xv, yv = torch.meshgrid(x, y)
            xv, yv, image = xv.unsqueeze(2), yv.unsqueeze(2), image.unsqueeze(2)
            res = torch.cat((xv, yv, image), 2).reshape(-1, 3)
            diagram = vr.fit_transform([res])
            all_diagrams.append(diagram[0, :, :2])
            
    elif filtration == 'sublevel':
        import gudhi as gd
        
        for image in tqdm(images):
            image /= torch.max(image)
            cc_density_crater = gd.CubicalComplex(
                dimensions = [32 , 32], 
                top_dimensional_cells = image.flatten()
            )
            cc_density_crater.compute_persistence()
            diagram = cc_density_crater.persistence()
            
            pd = np.zeros((0, 2))
            for k, pair in diagram:
                if k == 1 and not np.isinf(pair[1]):
                    pd = np.concatenate((pd, pair))
            all_diagrams.append(pd)     
    else:
        raise NotImplementedError            
    
    sigma, im_range, pimgs, all_diagrams = compute_pimgr_parameters(all_diagrams)
        
    n_train = int(len(labels) * 0.7)
    n_test = int(len(labels) * 0.3)
    dataset = MyDataset(all_diagrams, labels, images, pimgs)
    dataset_train, dataset_test = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(54))
    
    torch.save({
                'dataset_train' : dataset_train,
                'dataset_test' : dataset_test,
                'sigma' : sigma,
                'im_range' : im_range,
                }, './data/dataset_UIUC.pt')
    return 

def generate_obayashi_hiraoka_dataset(n_images=5000, W = 300, sigma1 = 4, sigma2 = 2, 
                        t = 0.01, N1 = 200, S1 = 30, N2 = 250, S2 = 20, 
                        bins=64, filtration = 'rips'):
    images = np.zeros((n_images, bins, bins))
    for n in tqdm(range(n_images // 2)):
        images[n] = generate_image(N1, S1, W, sigma1, sigma2, t, bins)

    for n in tqdm(range(n_images // 2)):
        images[n + n_images // 2] = generate_image(N2, S2, W, sigma1, sigma2, t, bins)
        
    all_diagrams = []
    images = torch.tensor(images)
    labels = torch.ones(n_images)
    labels[: n_images // 2] = 0
    
    if filtration == 'rips':
        # pi-net version of calculating pds
        vr = VietorisRipsPersistence(homology_dimensions=[1])
        for image in tqdm(images):
            nx, ny = image.shape
            x, y = torch.linspace(0, 1, nx), torch.linspace(0, 1, ny)
            xv, yv = torch.meshgrid(x, y)
            xv, yv, image = xv.unsqueeze(2), yv.unsqueeze(2), image.unsqueeze(2)
            res = torch.cat((xv, yv, image), 2).reshape(-1, 3)
            diagram = vr.fit_transform([res])
            all_diagrams.append(diagram[0, :, :2])
            
    elif filtration == 'sublevel':
        import gudhi as gd
        
        for image in tqdm(images):
            image /= torch.max(image)
            cc_density_crater = gd.CubicalComplex(
                dimensions = [32 , 32], 
                top_dimensional_cells = image.flatten()
            )
            cc_density_crater.compute_persistence()
            diagram = cc_density_crater.persistence()
            
            pd = np.zeros((0, 2))
            for k, pair in diagram:
                if k == 1 and not np.isinf(pair[1]):
                    pd = np.concatenate((pd, pair))
            all_diagrams.append(pd)     
    else:
        raise NotImplementedError
        
    sigma, im_range, pimgs, all_diagrams = compute_pimgr_parameters(all_diagrams)
        
    n_train = int(0.7 * n_images)
    n_test = n_images - n_train
    dataset = MyDataset(all_diagrams, labels, images, pimgs)
    dataset_train, dataset_test = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(54))
    torch.save({
                'dataset_train' : dataset_train,
                'dataset_test' : dataset_test,
                'sigma' : sigma,
                'im_range' : im_range,
                }, './data/dataset_Obayashi-Hiraoka.pt') 
    return 


def generate_orbit5k_dataset(n_points = 1000, dataset_size = 1000, train_frac=0.7):
    n_train = int(5 * dataset_size * train_frac)
    n_test = dataset_size * 5 - n_train
    vr = Rips()
    random_state = np.random.RandomState(54)
    X_orbit5k = generate_orbits(dataset_size, n=n_points, random_state=random_state)

    X_big = []

    y = np.zeros(dataset_size)

    for i in range(1, 5):
        y = np.concatenate((y, np.ones(dataset_size) * i))

    for x in tqdm(X_orbit5k):
        diagram = conv_pd(vr.fit_transform(x))
        X_big.append(diagram)
    
    sigma, im_range, pimgs, X_big = compute_pimgr_parameters(X_big)
    
    dataset = MyDataset(X_big, y, X_orbit5k, pimgs)
    dataset_train, dataset_test = random_split(dataset, [n_train, n_test], 
                                               generator=torch.Generator().manual_seed(54))
    torch.save({
                'dataset_train' : dataset_train,
                'dataset_test' : dataset_test,
                'sigma' : sigma,
                'im_range' : im_range,
                }, './data/dataset_Orbit5k.pt') 
    return

# https://huggingface.co/datasets/Msun/modelnet40/tree/main
def generate_modelnet40_dataset(sample_rate=1.):
    n_points = int(2048 * sample_rate)
    # getting train dataset
    X_all = []
    labels = []
    for i in range(5):
        filename = "./data/ModelNet40Raw/ply_data_train{0}.h5".format(i)
        with h5py.File(filename, "r") as f:
            data = f['data'][()]  # returns as a numpy array (bs, 2048, 3)
            cur_labels = f['label'][()]
            for (pc, label) in zip(data, cur_labels):
                np.random.shuffle(pc)
                X_all.append(pc[:n_points])
                labels.append(label)
                
    vr = Rips()            
    X_big = []       
    for x in tqdm(X_all):
        diagram = conv_pd(vr.fit_transform(x))
        X_big.append(diagram)
    
    sigma, im_range, pimgs, X_big = compute_pimgr_parameters(X_big)
    
    dataset_train = MyDataset(X_big, labels, X_all, pimgs)
    
    # getting test dataset
    X_all = []
    labels = []
    for i in range(2):
        filename = "./data/ModelNet40Raw/ply_data_test{0}.h5".format(i)
        with h5py.File(filename, "r") as f:
            data = f['data'][()]  # returns as a numpy array
            cur_labels = f['label'][()]
            for (pc, label) in zip(data, cur_labels):
                np.random.shuffle(pc)
                X_all.append(pc[:n_points])
                labels.append(label)
                
    X_big = []       
    for x in tqdm(X_all):
        diagram = conv_pd(vr.fit_transform(x))
        X_big.append(diagram)
    
    pimgr = PersistenceImageGudhi(bandwidth=sigma, resolution=[50, 50], 
                                  weight=lambda x: (x[1])**2, im_range=im_range)
    pimgs = pimgr.fit_transform(X_big)
    # normalizing
    for i, img in enumerate(pimgs):
        img /= img.max()
        pimgs[i] = img
    
    for i, diagram in enumerate(X_big):
        diagram[:, 1] -= diagram[:, 0] # to predict delta instead of second coord > first coord
        X_big[i] = diagram
    
    dataset_test = MyDataset(X_big, labels, X_all, pimgs)
    
    
    torch.save({
                'dataset_train' : dataset_train,
                'dataset_test' : dataset_test,
                'sigma' : sigma,
                'im_range' : im_range,
                }, './data/dataset_ModelNet40_{}_points.pt'.format(n_points)) 
    return


def get_loaders_by_name(dataset_name, batch_size, data_type):
    path = './data/dataset_' + dataset_name + '.pt'
    dataset = torch.load(path)
    dataset_train = dataset['dataset_train']
    dataset_test = dataset['dataset_test']
    im_range = dataset['im_range']
    sigma = dataset['sigma']
    
    if data_type == 'pc':
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, 
                                      shuffle=True, collate_fn=collate_fn_pc)
        dataloader_test =  DataLoader(dataset_test, batch_size=batch_size, 
                                      shuffle=False, collate_fn=collate_fn_pc)
    elif data_type == 'image':
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, 
                                      shuffle=True, collate_fn=collate_fn_img)
        dataloader_test =  DataLoader(dataset_test, batch_size=batch_size, 
                                      shuffle=False, collate_fn=collate_fn_img)
    
    if 'Orbit5k' in dataset_name:
        n_classes = 5  
    elif 'ModelNet40' in dataset_name:
        n_classes = 40
    elif 'Obayashi-Hiraoka' in dataset_name:
        n_classes = 2
    elif 'UIUC' in dataset_name:
        n_classes = 25
    else:
        # TBD
        raise NotImplementedError
    
    n_max = 0
    for src_pd, _, _, _, _ in dataloader_train:
        n_max = max(n_max, src_pd.shape[1])
        
    return dataloader_train, dataloader_test, n_classes, n_max, sigma, im_range
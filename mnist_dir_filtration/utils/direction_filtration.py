# https://github.com/ChervovNikita/topological_DL_course_work/blob/main/classification_transformer/utils.py
import gudhi as gd
import numpy as np
import math
import torch

def diagram(image, device, sublevel=True):
    # get height and square image
    h = int(np.sqrt(image.shape[0]))
    image_sq = image.reshape((h,h))

    # create complex
    cmplx = gd.CubicalComplex(dimensions=(h, h), top_dimensional_cells=(sublevel * image))

    # get pairs of critical simplices
    cmplx.compute_persistence()
    critical_pairs = cmplx.cofaces_of_persistence_pairs()
    
    # get essential critical pixel
    bpx0_essential = critical_pairs[1][0][0] // h, critical_pairs[1][0][0] % h

    # get critical pixels corresponding to critical simplices
    try:
        bpx0 = [critical_pairs[0][0][i][0] for i in range(len(critical_pairs[0][0]))]
        dpx0 = [critical_pairs[0][0][i][1] for i in range(len(critical_pairs[0][0]))]
    except IndexError:
        bpx0 = []
        dpx0 = []
        
    try:
        bpx1 = [critical_pairs[0][1][i][0] for i in range(len(critical_pairs[0][1]))]
        dpx1 = [critical_pairs[0][1][i][1] for i in range(len(critical_pairs[0][1]))]
    except IndexError:
        bpx1 = []
        dpx1 = []
    

    flat_image = image_sq.flatten()
    pd0_essential = torch.tensor([[image_sq[bpx0_essential], torch.max(image)]])

    if (len(bpx0)!=0):
        pdb0 = flat_image[bpx0][:, None]
        pdd0 = flat_image[dpx0][:, None]
        pd0 = torch.hstack([pdb0, pdd0])
        pd0 = torch.vstack([pd0, pd0_essential.to(device)])
    else:
        pd0 = pd0_essential

    if (len(bpx1)!=0):
        pdb1 = flat_image[bpx1][:, None]
        pdd1 = flat_image[dpx1][:, None]
        pd1 = torch.hstack([pdb1, pdd1])
    else:
        pd1 = torch.zeros((1, 2))
    
    return pd0, pd1


def process_by_direction(img, alpha):
    X = (math.cos(alpha) - (np.arange(0, img.shape[0]) - (img.shape[0] / 2 - 0.5)) / (img.shape[0] * math.sqrt(2))) * math.cos(alpha) / 2
    Y = (math.sin(alpha) - (np.arange(0, img.shape[1]) - (img.shape[1] / 2 - 0.5)) / (img.shape[1] * math.sqrt(2))) * math.sin(alpha) / 2
    direction_filter = X.reshape(-1, 1) + Y.reshape(1, -1)
    return np.maximum(direction_filter, img)


def process_image(img, num_filtrations, filter_type='uniform'):
    if filter_type == 'uniform':
        filter_params = np.arange(num_filtrations) / num_filtrations * 2 * math.pi
    else:
        raise NotImplementedError
    w = img.shape[-1]
    imgs = [process_by_direction(img.reshape(w, w), alpha) for alpha in filter_params]
    diagrams = []
    for i, img in enumerate(imgs):
        pd0, pd1 = diagram(img.flatten(), img.device)
        add_features0 = torch.tensor([0., filter_params[i]])
        add_features1 = torch.tensor([1., filter_params[i]])
        pd0 = torch.cat([pd0, add_features0.view(1, -1).repeat(pd0.shape[0], 1)], axis=1)
        pd1 = torch.cat([pd1, add_features1.view(1 ,-1).repeat(pd1.shape[0], 1)], axis=1)
        diagrams.append(torch.cat([pd0, pd1], axis=0))
    diagrams = torch.cat(diagrams)
    return diagrams, imgs
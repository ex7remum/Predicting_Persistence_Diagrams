import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def calc_pie(model : nn.Module, dataloader, pimgr=None):
    model.eval()
    total_pie = 0.
    mse = MSELoss(reduction = 'sum')
    for _, _, _, src_data, PI in dataloader:
        pred = model(src_data.to(device))
        PI = PI.reshape((-1, 50*50))
        if pimgr is not None:
            pred[..., 1] += pred[..., 0]
            pimgs = pimgr.fit_transform(pred.cpu().numpy())
            pimgs = np.array(pimgs)
            for i, img in enumerate(pimgs):
                pimgs[i] = img / img.max()
            pred_pi = torch.tensor(pimgs).reshape((-1, 50*50))
        else:
            pred_pi = pred
        
        total_pie += mse(pred_pi.to(device), PI.to(device))
    return total_pie.item() / len(dataloader.dataset)

@torch.no_grad()
def calc_inference_time(model : nn.Module, dataloader):
    model.eval()
    total_time = 0.
    for src_pd, mask, labels, src_data, _ in dataloader:
        t1 = time.time()
        _ = model(src_data.to(device))
        t2 = time.time()
        total_time += t2 - t1
    total_time /= len(dataloader.dataset)
    return total_time


@torch.no_grad()
def calc_gudhi_W2_dist(model : nn.Module, dataloader):
    from gudhi.wasserstein import wasserstein_distance
    
    model.eval()
    W2 = 0.
    for src_pd, mask, labels, src_data, _ in dataloader:
        tgt_pd = model(src_data.to(device))
        for src, tgt in zip(src_pd, tgt_pd):
            # our data is stored in (b, d - b) format
            src[:, 1] += src[:, 0]
            tgt[:, 1] += tgt[:, 0]
            W2 += wasserstein_distance(src.cpu(), tgt.cpu(), order=1., internal_p=2.)
    W2 /= len(dataloader.dataset)
    return W2


@torch.no_grad()
def calc_gudhi_bottleneck_dist(model : nn.Module, dataloader):
    from gudhi import bottleneck_distance
    
    model.eval()
    bottleneck = 0.
    for src_pd, mask, labels, src_data, _ in dataloader:
        tgt_pd = model(src_data.to(device))
        for src, tgt in zip(src_pd, tgt_pd):
            # our data is stored in (b, d - b) format
            src[:, 1] += src[:, 0]
            tgt[:, 1] += tgt[:, 0]
            bottleneck += bottleneck_distance(src.cpu(), tgt.cpu())
    bottleneck /= len(dataloader.dataset)
    return bottleneck


# calculate accuracy of classificaiton of some model trained on pds
@torch.no_grad()
def calculate_accuracy(model_pd : nn.Module, model_class : nn.Module, dataloader, on_real = True):
    model_pd.eval()
    model_class.eval()
    correct = 0.
    for src_pd, mask, labels, src_data, _ in dataloader:
        if on_real:
            logits = model_class(src_pd.to(device), mask.to(device))
        else:
            tgt_pd = model_pd(src_data.to(device))
            mask_pd = (tgt_pd != 0)[:, :, 1]
            logits = model_class(tgt_pd, mask_pd.to(device))
            
        correct += (labels.to(device) == torch.argmax(logits, axis=1)).sum()
    correct /= len(dataloader.dataset)
    return correct.item()
    
    
@torch.no_grad()
def logreg_and_rfc_acc(dataloader_train, dataloader_test, name, pi_type = 'default', model = None, pimgr = None):
    X_train, X_test = [], []
    y_train, y_test = [], []

    for src_pd, mask, labels, src_data, PI in dataloader_train:
        if pi_type == 'default':
            approx_pi = PI.clone()
        elif pi_type == 'pi_model':
            out = model(src_data.to(device))
            approx_pi = out.squeeze(1).cpu()
        elif pi_type == 'from_pd':
            pred_pds = model(src_data.to(device))
            pred_pds = pred_pds.cpu()
            # return back to (b, d) coords for pimgr to work correctly
            pred_pds[..., 1] += pred_pds[..., 0]
            approx_pi = np.array(pimgr.fit_transform(pred_pds)).reshape((-1, 50*50))
            for i, img in enumerate(approx_pi):
                approx_pi[i] = img / img.max()
        else:
            raise NotImplementedError
        
        for img in approx_pi:
            if pi_type != 'from_pd':
                X_train.append(img.numpy())
            else:
                X_train.append(img)
        for label in labels:
            if pi_type != 'from_pd':
                y_train.append(label.numpy())
            else:
                y_train.append(label)

    for src_pd, mask, labels, src_data, PI in dataloader_test:
        if pi_type == 'default':
            approx_pi = PI.clone()
        elif pi_type == 'pi_model':
            out = model(src_data.to(device))
            approx_pi = out.squeeze(1).cpu()
        elif pi_type == 'from_pd':
            pred_pds = model(src_data.to(device))
            pred_pds = pred_pds.cpu().numpy()    
            # return back to (b, d) coords for pimgr to work correctly
            pred_pds[..., 1] += pred_pds[..., 0]
            approx_pi = np.array(pimgr.fit_transform(pred_pds)).reshape((-1, 50*50))
            for i, img in enumerate(approx_pi):
                approx_pi[i] = img / img.max()
        else:
            raise NotImplementedError
        
        
        for img in approx_pi:
            if pi_type != 'from_pd':
                X_test.append(img.numpy())
            else:
                X_test.append(img)
        for label in labels:
            if pi_type != 'from_pd':
                y_test.append(label.numpy())
            else:
                y_test.append(label)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))
    
    scaler = StandardScaler()
    
    accuracies = []
    n_estimators = [5, 20, 50, 100, 500, 1000]
    for n_estimator in n_estimators:
        rfc = RandomForestClassifier(n_estimators=n_estimator)
        rfc.fit(scaler.fit_transform(X_train), y_train)
        accuracies.append(rfc.score(scaler.transform(X_test), y_test))

    acc_rfc = np.max(accuracies)
    
    accuracies = []

    Cs = [1, 5, 10, 100, 500]
    for C in Cs:
        log_reg = LogisticRegression(C=C, max_iter=1000)
        log_reg.fit(scaler.transform(X_train), y_train)
        accuracies.append(log_reg.score(scaler.transform(X_test), y_test))

    acc_logreg = np.max(accuracies)
    return acc_logreg, acc_rfc
    
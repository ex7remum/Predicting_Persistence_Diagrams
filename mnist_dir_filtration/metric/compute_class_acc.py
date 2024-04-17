import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def get_items_from_dataloader(dataloader, model=None, pimgr=None):
    data, labels = [], []
    
    for item in dataloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        Z = Z[..., :2].to(torch.float32)
        if model is None:
            # compute PIs on real PDs
            PI = torch.from_numpy(pimgr.fit_transform(Z)).to(torch.float32)
            PI = PI / PI.max(dim=1, keepdim=True)[0]
        else:
            out = model(X.to(device)).cpu()
            
            if pimgr is None:
                # PI model
                PI = out
            else:
                # PD model
                PI = torch.from_numpy(pimgr.fit_transform(out)).to(torch.float32)
        
        for img in PI:
            data.append(img.numpy())
        for label in v:
            labels.append(label.item())
            
    return data, labels

@torch.no_grad()
def logreg_and_rfc_acc(dataloader_train, dataloader_test, model=None, pimgr=None):
    X_train, y_train = get_items_from_dataloader(dataloader_train, model, pimgr)
    X_test, y_test = get_items_from_dataloader(dataloader_test, model, pimgr)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
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
    
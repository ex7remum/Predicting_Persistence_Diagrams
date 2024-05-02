import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from trainer import move_batch_to_device


@torch.no_grad()
def get_items_from_dataloader(dataloader, model=None, pimgr=None):
    data, labels = [], []
    device = next(model.parameters()).device
    
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        v = batch['labels']
        if model is None:
            # compute PIs on real PDs
            PI = batch['pis']
        else:
            out = model(batch)
            
            if pimgr is None:
                # PI model
                PI = out['pred_pis'].cpu()
            else:
                # PD model
                out = out['pred_pds'].cpu()
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


# calculate accuracy of classificaiton of some model trained on pds
@torch.no_grad()
def calculate_accuracy_on_pd(model_pd: nn.Module, model_class: nn.Module, dataloader, on_real=True):
    model_class.eval()
    if model_pd is not None:
        model_pd.eval()
    correct = 0.
    device = next(model_class.parameters()).device

    val_len = len(dataloader.dataset)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        labels = batch['labels']
        with torch.no_grad():
            if on_real:
                logits = model_class(batch)['logits']
            else:
                PD_pred = model_pd(batch)['pred_pds']
                dumb_batch = {'pds': PD_pred}
                logits = model_class(dumb_batch)['logits']
            correct += (labels == torch.argmax(logits, axis=1)).sum()

    val_pd_acc = correct / val_len
    return val_pd_acc.item()

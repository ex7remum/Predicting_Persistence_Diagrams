import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from trainer import move_batch_to_device
import trainer


@torch.no_grad()
def get_items_from_dataloader(dataloader, device, model=None, pimgr=None):
    data, labels = [], []
    
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        v = batch['labels'].cpu()
        if model is None:
            # compute PIs on real PDs
            PI = batch['pis'].cpu()
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
def logreg_and_rfc_acc(dataloader_train, dataloader_test, device, model=None, pimgr=None):
    X_train, y_train = get_items_from_dataloader(dataloader_train, device, model, pimgr)
    X_test, y_test = get_items_from_dataloader(dataloader_test, device, model, pimgr)

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
def calculate_accuracy_on_pd(model, valloader, device):
    model.eval()
    correct = 0.
    val_len = len(valloader.dataset)

    for batch in valloader:
        batch = trainer.move_batch_to_device(batch, device)
        labels = batch['labels']
        with torch.no_grad():
            logits = model(batch)['logits']
            correct += (labels == torch.argmax(logits, axis=1)).sum()

    val_pd_acc = correct / val_len
    return val_pd_acc.item()

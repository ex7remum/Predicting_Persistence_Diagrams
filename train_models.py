import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from models.point_clouds_encoders import CustomPersformer

from utils.loss_functions import HungarianLoss, SlicedWasserstein, ChamferLoss
from utils.metrics import calc_gudhi_W2_dist, calc_gudhi_bottleneck_dist, calculate_accuracy

import wandb
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_size_predictor(model_size, dataloader_train, dataloader_test, name, n_epochs = 200):
    lr = 0.001
    mse_crit = MSELoss()
    optimizer = AdamW(model_size.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=20, min_lr=1e-5, factor=0.5)
    
    for epoch_idx in range(n_epochs):
        model_size.train()

        loss = 0
        for _, mask, _, src_data, _ in dataloader_train:
            sizes = (torch.sum(mask, dim=1)) * 1.
            pred = model_size(src_data.to(device)).squeeze()
            loss_batch = mse_crit(pred, sizes.to(device))
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss += loss_batch.detach().cpu()

        loss_train = loss / len(dataloader_train.dataset)

        model_size.eval()
        for _, mask, _, src_data, _ in dataloader_test:
            with torch.no_grad():
                sizes = torch.sum(mask, dim=1) * 1.
                pred = model_size(src_data.to(device)).squeeze()
                loss_batch = mse_crit(pred, sizes.to(device))
                loss += loss_batch.detach().cpu()

        loss_test = loss / len(dataloader_test.dataset)
        scheduler.step(loss_test)
        print(epoch_idx, "train loss size predictor " + name, loss_train.item(), 
              "test loss size_predictor " + name, loss_test.item())
        #wandb.log({"train loss size predictor " + name: loss_train.item(), "test loss size_predictor " + name: loss_test.item()})

    return model_size


def train_classifier_on_pds(dataloader_train, dataloader_test, n_classes, real_pd, model, 
                            name, n_epochs = 300, warmup_iters=10):
    if real_pd:
        add = 'real'
    else:
        add = 'pred'
    model_pd = CustomPersformer(n_in = 2, embed_dim = 128, fc_dim = 256, num_heads = 8, 
                                num_layers = 5, n_out_enc = n_classes, dropout = 0.0, 
                                reduction = 'mean').to(device)
    crit = CrossEntropyLoss()
    lr = 1e-3
    optimizer = AdamW(model_pd.parameters(), lr=lr, weight_decay=1e-4)
    
    n_epochs += warmup_iters
    scheduler1 = LinearLR(optimizer, start_factor=0.000000001, total_iters=warmup_iters)
    
    best_test_acc = 0
    best_model = CustomPersformer(n_in = 2, embed_dim = 128, fc_dim = 256, num_heads = 8, 
                                  num_layers = 5, n_out_enc = n_classes, dropout = 0.0, 
                                  reduction = 'mean').to(device)
    name += '_classificator'
    for epoch_idx in range(n_epochs):
        model_pd.train()
        loss_train, correct_train = 0.0, 0.0
        for src_pd, mask, labels, src_data, PI in dataloader_train:
            with torch.no_grad():
                if real_pd:
                    pred_pds = src_pd.detach().clone().to(device)
                    mask_pd = mask.detach().clone().to(device)
                else:
                    pred_pds = model(src_data.to(device))
                    mask_pd = (pred_pds != 0)[:, :, 1]


            logits = model_pd(pred_pds, mask_pd)
            loss_batch = crit(logits, labels.to(device))
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss_batch.detach().cpu()
            correct_train += (labels.to(device) == torch.argmax(logits, axis=1)).sum()
        loss_train /= len(dataloader_train.dataset)
        correct_train /= len(dataloader_train.dataset)
        
        model_pd.eval()
        loss_test, correct_test = 0.0, 0.0
        for src_pd, mask, labels, src_data, PI in dataloader_test:
            with torch.no_grad():
                if real_pd:
                    pred_pds = src_pd.detach().clone().to(device)
                    mask_pd = mask.to(device)
                else:
                    pred_pds = model(src_data.to(device))
                    mask_pd = (pred_pds != 0)[:, :, 1]
                    
                logits = model_pd(pred_pds, mask_pd)
                loss_batch = crit(logits, labels.to(device))
                loss_test += loss_batch
                correct_test += (labels.to(device) == torch.argmax(logits, axis=1)).sum()
                
        loss_test /= len(dataloader_test.dataset)
        correct_test /= len(dataloader_test.dataset)
        
        if correct_test > best_test_acc:
            best_test_acc = correct_test
            best_model = copy.deepcopy(model_pd)
        
        if epoch_idx < warmup_iters:
            scheduler1.step()
        else:
            if epoch_idx == warmup_iters:
                scheduler2 = ReduceLROnPlateau(optimizer, patience=25, min_lr=1e-5, factor=0.5)
            scheduler2.step(loss_test)
            
        print(epoch_idx, "train loss classifier " + add + ' ' + name, loss_train.item(), 
                   "test loss classifier " + add + ' ' + name, loss_test.item())
        print(epoch_idx, "acc train classifier " + add + ' ' + name, correct_train.item(), 
                   "acc test classifier " + add + ' ' + name, correct_test.item())
        #wandb.log({"train loss classifier " + add + ' ' + name: loss_train.item(), 
        #           "test loss classifier " + add + ' ' + name: loss_test.item()})
        #wandb.log({"acc train classifier " + add + ' ' + name: correct_train.item(), 
        #           "acc test classifier " + add + ' ' + name: correct_test.item()})
        
    best_model.eval()
    acc = f'{best_test_acc:.4f}'
    torch.save(best_model.state_dict(), './pretrained_models/model_classifier_' + add + '_' + acc + '_' + name + ".pt")
    
    return best_model

# Train our model for predcting pds
def train_full_model(model, dataloader_train, dataloader_test, name, 
                     run_number, n_epochs = 200, warmup_iters = 20):
    # Define hyperparameters for training
    lr = 0.001
    n_epochs += warmup_iters
    criterion = SlicedWasserstein(n_projections=100)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler1 = LinearLR(optimizer, start_factor=0.00000001, total_iters=warmup_iters)
    

    for epoch_idx in range(n_epochs):
        model.train()
        loss = 0
        for src_pd, mask, _, src_data, _ in dataloader_train:
            tgt_pd = model(src_data.to(device), mask.to(device))
            loss_batch = criterion(src_pd.to(device), tgt_pd)
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss += loss_batch.detach().cpu()
        loss_train = loss / len(dataloader_train.dataset)

        loss = 0
        model.eval()
        for src_pd, mask, _, src_data, _ in dataloader_test:
            with torch.no_grad():
                tgt_pd = model(src_data.to(device))
                loss_batch = criterion(src_pd.to(device), tgt_pd)
                loss += loss_batch
        loss_test = loss / len(dataloader_test.dataset)
        print(epoch_idx, "train loss full model " + name, loss_train.item(), 
              "test loss full model " +  name, loss_test.item())
        #wandb.log({"train loss full model " + name: loss_train.item(), "test loss full model " +  name: loss_test.item()})
        
        if epoch_idx < warmup_iters:
            scheduler1.step()
        else:
            if epoch_idx == warmup_iters:
                scheduler2 = ReduceLROnPlateau(optimizer, patience=25, min_lr=1e-5, factor=0.5)
            scheduler2.step(loss_test)

        if epoch_idx % 10 == 0:
            train_W2 = calc_gudhi_W2_dist(model, dataloader_train) 
            test_W2 = calc_gudhi_W2_dist(model, dataloader_test)
            
            torch.save(model.state_dict(), './pretrained_models/full_model_' + str(epoch_idx) + '_epoch_' + name + '_run_' + str(run_number) + ".pt")
            
            train_btlnck = calc_gudhi_bottleneck_dist(model, dataloader_train)
            test_btlnck = calc_gudhi_bottleneck_dist(model, dataloader_test)
            print(epoch_idx, "W2 full model train " + name, train_W2, "W2 full model test " +  name, test_W2)
            print(epoch_idx, "bottleneck full model train " + name, train_btlnck, "bottleneck full model test " +  name, test_btlnck)
            #wandb.log({"W2 full model train " + name: train_W2, "W2 full model test " +  name: test_W2})
            #wandb.log({"bottleneck full model train " + name: train_btlnck, "bottleneck full model test " +  name: test_btlnck})
                   
    return model

def train_pi_model(model, dataloader_train, dataloader_test, crit, name, lr=1e-3, n_epochs=300):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=25, min_lr=1e-6, factor=0.5)

    for _ in range(n_epochs):
        model.train()
        loss_train = 0.0
        for src_pd, mask, labels, src_data, PI in dataloader_train:
            out = model(src_data.to(device))
            loss_batch = crit(out, PI.reshape((-1, 2500)).to(device))
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss_batch.detach().cpu()
        loss_train /= len(dataloader_train.dataset)

        model.eval()
        loss_test = 0.0
        for  src_pd, mask, labels, src_data, PI in dataloader_test:
            with torch.no_grad():
                out = model(src_data.to(device))
                loss_batch = crit(out, PI.reshape((-1, 2500)).to(device))
                loss_test += loss_batch
        loss_test /= len(dataloader_test.dataset)
        scheduler.step(loss_test)
        print(_, "train loss " + name, loss_train.item(), "test loss " +  name, loss_test.item())
        #wandb.log({"train loss " + name: loss_train.item(), "test loss " +  name: loss_test.item()})  
    return model
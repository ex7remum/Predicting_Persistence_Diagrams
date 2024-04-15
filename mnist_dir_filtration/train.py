import torch
import wandb
from datasets import PDMnist
from torch.utils.data import Dataset, DataLoader
from models import ImageSet2Set
from losses import PersistenceWeightedSlicedWassersteinLoss, ChamferLoss, HausdorffLoss
from config import hyperparams
import argparse

def val_step(model, valloader, device):
    model.eval()
    metric_chamfer = 0.
    metric_hausdorff = 0.
    val_len = len(valloader.dataset)
    for X, Z, v in valloader:
        with torch.no_grad():
            Z = Z[..., :2].to(torch.float32).to(device)
            Z_hat = model(X.to(device))
        
        metric_chamfer += ChamferLoss(reduce='sum')(Z_hat, Z)
        metric_hausdorff += HausdorffLoss(reduce='sum')(Z_hat, Z)
    wandb.log({'val_chamfer': metric_chamfer / val_len, 'val_hausdorff': metric_hausdorff / val_len})

def train_loop(model, trainloader, valloader, optimizer, loss_fn, device, scheduler=None, n_epochs=25):
    torch.manual_seed(0)
    
    for _ in range(n_epochs):
        model.train()
        for X, Z, v in trainloader:
            optimizer.zero_grad()
            
            Z = Z[..., :2].to(torch.float32).to(device)
            Z_hat = model(X.to(device))
            loss = loss_fn(Z_hat, Z)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            lr = optimizer.param_groups[0]['lr']
            wandb.log({'loss': loss, 'grad_norm': total_norm, 'learning rate': lr})
        
        val_step(model, valloader, device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-w",
        "--wandb_key",
        default=None,
        type=str,
        help="wandb key for logging",
    )
    args = parser.parse_args()
    
    
    train_dataset = PDMnist(data_dir='data', train=True, num_filtrations=hyperparams["num_filtrations"], leave=32)
    test_dataset = PDMnist(data_dir='data', train=False, num_filtrations=hyperparams["num_filtrations"], leave=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], num_workers=2, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], num_workers=2, shuffle=False)

    model = ImageSet2Set(**hyperparams["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=hyperparams["n_steps_warmup"])

    loss_fn = PersistenceWeightedSlicedWassersteinLoss(q=1, reduce="sum", random_seed=0)

    run = hyperparams["run_name"]
    wandb.login(key=args.wandb_key)
    wandb.init(project=hyperparams["project_name"], 
               name=f"experiment_{run}",
               config=hyperparams
    )

    final_model = train_loop(model, trainloader, testloader, optimizer, loss_fn, device, scheduler, n_epochs=hyperparams["n_epochs"])
    
    torch.save(final_model.state_dict(), f'pretrained_models/{run}_model.pth')
    wandb.finish()
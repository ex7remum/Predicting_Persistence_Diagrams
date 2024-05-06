import torch
import wandb
import argparse
import utils
import os
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImageGudhi
import trainer
import json
from torch.utils.data import DataLoader
import collate_fn
import metrics


val_functions = {
    "pi": trainer.val_step_pi_model,
    "pd": trainer.val_step_pd_model,
    "size_pred": trainer.val_step_size_predictor,
    "class": trainer.val_step_classificator
}


def run_exp_full(args):
    f = open(args.config)
    config = json.load(f)
    n_runs = config['trainer']['n_runs']
    for n_run in range(n_runs):

        if 'device' in config['trainer']:
            device = config['trainer']['device']
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train1_dataset, train2_dataset, test_dataset = trainer.get_dataloaders(config)

        loss_fn = trainer.get_loss_fn(config)
        pimgr = trainer.get_pimgr(config)
        if pimgr is None and (args.type == 'pd' or args.type == 'pi'):
            sigma, im_range = utils.compute_pimgr_parameters(train1_dataset.pds)
            pimgr = PersistenceImageGudhi(bandwidth=sigma,
                                          resolution=[50, 50],
                                          weight=lambda x: x[1],
                                          im_range=im_range)

        collator = collate_fn.Collator(pimgr=pimgr)

        trainloader1 = DataLoader(train1_dataset, batch_size=config['data']['train']['batch_size'],
                                  num_workers=config['data']['train']['num_workers'], shuffle=True, drop_last=True,
                                  collate_fn=collator)

        trainloader2 = DataLoader(train2_dataset, batch_size=config['data']['train']['batch_size'],
                                  num_workers=config['data']['train']['num_workers'], shuffle=True, drop_last=True,
                                  collate_fn=collator)

        testloader = DataLoader(test_dataset, batch_size=config['data']['test']['batch_size'],
                                num_workers=config['data']['test']['num_workers'], shuffle=False, collate_fn=collator)

        model, optimizer, scheduler = trainer.get_train_model_params(config)
        model = model.to(device)

        run = trainer.init_wandb(config=config, wandb_key=args.wandb_key, run_num=n_run)
        wandb.watch(model)

        if args.type == 'class':
            trainloader = trainloader2
        else:
            trainloader = trainloader1

        final_model = trainer.train_loop(model=model,
                                         trainloader=trainloader,
                                         valloader=testloader,
                                         optimizer=optimizer,
                                         loss_fn=loss_fn,
                                         device=device,
                                         val_function=val_functions[args.type],
                                         exp_type=args.type,
                                         scheduler=scheduler,
                                         n_epochs=config["trainer"]["n_epochs"],
                                         clip_norm=config["trainer"]["grad_norm_clip"],
                                         seed=n_run)

        if 'save_path' in config['trainer']:
            torch.save(final_model.state_dict(), config['trainer']['save_path'])
        else:
            os.makedirs('pretrained_models', exist_ok=True)
            torch.save(final_model.state_dict(), f'pretrained_models/{run}_model.pth')

        if args.type == 'pi' or args.type == 'pd' or args.type == 'class':
            res_metrics = {}

            if args.type == 'pi' or args.type == 'pd':
                time = metrics.calc_inference_time(model, testloader)
                res_metrics.update({f'time_{args.type}': time})

            if args.type == 'pd':
                pie = metrics.calc_pie_from_pd(model, testloader, pimgr)
                w2 = metrics.calc_gudhi_W2_dist(model, testloader)
                res_metrics.update({f'PIE_{args.type}': pie.item()})
                res_metrics.update({'W2': w2})

            elif args.type == 'pi':
                pie = metrics.calc_pie_from_pi(model, testloader, pimgr)
                res_metrics.update({f'PIE_{args.type}': pie.item()})

            else:
                class_acc = metrics.calculate_accuracy_on_pd(model, testloader, device)
                if model.is_real:
                    res_metrics.update({f'acc_real_pd_class': class_acc})
                else:
                    res_metrics.update({f'acc_pred_pd_class': class_acc})

            f = open('result_metrics.json')
            all_metrics = json.load(f)
            dataset_name = config['data']['train']['dataset']['type']
            model_name = config['arch']['type']
            exp_name = config['trainer']['run_name']

            if dataset_name not in all_metrics:
                all_metrics[dataset_name] = {}

            if model_name not in all_metrics[dataset_name]:
                all_metrics[dataset_name][model_name] = {}

            if exp_name not in all_metrics[dataset_name][model_name]:
                all_metrics[dataset_name][model_name][exp_name] = {}

            for metric_name in res_metrics:
                if metric_name not in all_metrics[dataset_name][model_name][exp_name]:
                    all_metrics[dataset_name][model_name][exp_name][metric_name] = []

                all_metrics[dataset_name][model_name][exp_name][metric_name].append(res_metrics[metric_name])

            with open('result_metrics.json', 'w') as f:
                json.dump(all_metrics, f)
            wandb.log(res_metrics)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-w",
        "--wandb_key",
        default=None,
        type=str,
        help="wandb key for logging",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="path to trainer config",
    )
    parser.add_argument(
        "-t",
        "--type",
        default=None,
        type=str,
        help="type of experiment",
    )

    args = parser.parse_args()
    run_exp_full(args)

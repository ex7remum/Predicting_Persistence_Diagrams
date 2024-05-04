import torch
import wandb
import argparse
import utils
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImageGudhi
import trainer
import json
from torch.utils.data import DataLoader
import collate_fn
import metrics


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

        pimgr = trainer.get_pimgr(config)
        if pimgr is None:
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

        _ = trainer.init_wandb(config=config, wandb_key=args.wandb_key, run_num=n_run)

        res_metrics = {}
        if args.type == 'real':
            acc_logreg_real, acc_rfc_real = metrics.logreg_and_rfc_acc(trainloader2, testloader, device, None, pimgr)
            res_metrics.update({'logreg_acc_real_pi': acc_logreg_real})
            res_metrics.update({'rfc_acc_real_pi': acc_rfc_real})

        else:
            model, optimizer, scheduler = trainer.get_train_model_params(config)
            model.load_state_dict(torch.load(args.model_path))
            model = model.to(device)
            if args.type == 'pd':
                acc_logreg, acc_rfc = metrics.logreg_and_rfc_acc(trainloader2, testloader, device, model, pimgr)
                res_metrics.update({f'logreg_acc_pred_{args.type}': acc_logreg})
                res_metrics.update({f'rfc_acc_pred_{args.type}': acc_rfc})

            elif args.type == 'pi':
                acc_logreg, acc_rfc = metrics.logreg_and_rfc_acc(trainloader2, testloader, device, model, None)
                res_metrics.update({f'logreg_acc_pred_{args.type}': acc_logreg})
                res_metrics.update({f'rfc_acc_pred_{args.type}': acc_rfc})

            elif args.type == 'class':
                # classification model
                class_acc = metrics.calculate_accuracy_on_pd(model, testloader, device)
                if model.is_real:
                    res_metrics.update({f'acc_real_pd_class': class_acc})
                else:
                    res_metrics.update({f'acc_pred_pd_class': class_acc})
            else:
                raise NotImplementedError

        f = open('result_metrics.json')
        all_metrics = json.load(f)
        dataset_name = config['data']['train']['dataset']['type']
        if args.type != 'real':
            model_name = config['arch']['type']
            exp_name = config['trainer']['run_name']
        else:
            model_name = 'real'
            exp_name = 'real'

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
    parser.add_argument(
        "-p",
        "--model_path",
        default=None,
        type=str,
        help="path to pretrained model",
    )

    args = parser.parse_args()
    run_exp_full(args)

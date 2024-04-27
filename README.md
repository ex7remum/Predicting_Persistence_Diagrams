## Dataset preparation (MNIST)
```buildoutcfg
python prepare_dataset.py -c configs/data_configs/mnist_dir10_config.json
```

## Run experiment on model that predicts PD (MNIST)
```buildoutcfg
python run_exp_pd_model.py -c configs/train_configs/img_transformer_config.json
                           -l configs/train_configs/persformer_mnist_dir10.json
                           -w {your_wandb_key}
```

## Run experiment on model that predicts PI (PI-Net for MNIST)
```buildoutcfg
python run_exp_pi_model.py -c configs/train_configs/pinet_mnist_dir_10.json
                           -w {your_wandb_key}
```
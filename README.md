## Dataset preparation
```buildoutcfg
python prepare_dataset.py -c path_to_data_config.json
```

## Run any experiment
```buildoutcfg
python run_exp_unified.py -c path_to_config.json
                           -t {model_type: pd, pi, class, size_pred}
                           -w {your_wandb_key}
```

You can see how configs look like in ```configs``` folder. 
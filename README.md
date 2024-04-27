## Dataset preparation
```buildoutcfg
python prepare_dataset.py -c {path_to_config}.json
```

## Run experiment 
```buildoutcfg
python run_exp_{pi/pd}_model.py -c {path_to_config}.json
                                -w {your_wandb_key'}
                                [-l {path_to_conig_of_classification_model}.json]
```
-l key is used only in PD model experiment.
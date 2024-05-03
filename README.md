## Dataset preparation
```buildoutcfg
python prepare_dataset.py -c path_to_data_config.json
```

To create dataset for experiments you should specify dataset from `datasets` folder and filtration function from `utils` (or add custom one) in configs.

Examlpe configs for dataset creation can be found in `configs/data_configs` 

## Train approximation model 
```buildoutcfg
python run_exp_unified.py  -c path_to_config.json
                           -t {model_type: pd, pi, class, size_pred}
                           -w {your_wandb_key}
```

```-t``` - flags meanings: 

```pd``` - train model that predicts PDs

```pi``` - train model that predicts PIs

```class``` - train model (Persformer) that runs classification task on PDs (see ```configs/train_configs/classificators/``` for more info)

```size_pred``` - size predictor for PDs (in case of exps with PDs of various lengths)

You can see how configs look like in ```configs``` folder. 

## Get metrics
```buildoutcfg
python calculate_metrics.py  -c path_to_config.json
                             -t {model_type: pd, pi, class, real}
                             -w {your_wandb_key}
                             -p {path to pretrained model}
``` 
```-t``` - flags meanings:

```real``` - calculate RFC and LogReg acc on real PIs (you may not provide ```-p``` flag in this case) 

```pd``` - calculate RFC and LogReg acc on PIs obtained from approximated PDs

```pi``` - calculate RFC and LogReg acc on PIs obtained from approximation model

```class``` - calculate classification accuracy either on real PDs or approximated ones (see ```configs/train_configs/classificators/``` for more info)

 All the resulted metrics will be stored in ```result_metrics.json``` file.
 
 To get final table with metrics see ```compute_metrics_table.ipynb```.
#!/bin/sh
pd_model_config=''
pi_model_config=''
class_real_config=''
class_pred_config=''
wandb_key=''
saved_pd_model_path=''
saved_pi_model_path=''


python -u run_exp_unified -c $pd_model_config -t pd -w $wandb_key > pd_model.log

python -u run_exp_unified -c $pi_model_config -t pi -w $wandb_key > pi_model.log

python -u run_exp_unified -c $class_real_config -t class -w $wandb_key > class_real_model.log

python -u run_exp_unified -c $class_pred_config -t class -w $wandb_key > class_pred_model.log

python -u calculate_metrics -c $pd_model_config -t real -w $wandb_key > real_metrics.log

python -u calculate_metrics -c $pd_model_config -t pd -w $wandb_key -p $saved_pd_model_path > pd_metrics.log

python -u calculate_metrics -c $pi_model_config -t pi -w $wandb_key -p $saved_pi_model_path > pi_metrics.log
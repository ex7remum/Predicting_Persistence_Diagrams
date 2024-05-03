You can see some examples of how configs for classificators look like in configs in this folder.

To train classificators on real PDs, please set ```on_real``` flag in config to ```false```.

In this case you can either set ```model_pd_config``` and ```model_pd_path``` to ```null``` (or even remove them)

If you want to train classificator on approximated PDs you need to set  ```on_real``` flag in config as ```true``` and specify
```model_pd_config``` and ```model_pd_path```. 

```model_pd_config``` should contain `arch` field as in config that are used
for training (you can just use the same configs as for training, it should be fine).
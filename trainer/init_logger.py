import wandb


def init_wandb(config, wandb_key, run_num=None):
    run = config["trainer"]["run_name"]
    if run_num is not None:
        run = f'{run}_{run_num}'

    wandb.login(key=wandb_key, relogin=True)
    wandb.init(project=config["trainer"]["wandb_project"],
               name=f"{run}",
               config=config)
    return run

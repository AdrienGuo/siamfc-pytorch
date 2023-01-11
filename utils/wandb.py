import wandb


class WandB():
    def __init__(self, name: str, config: dict, init: bool) -> None:
        self.info = None
        if init:
            wandb.init(
                name=name,
                project="SiamFC",
                config=config
            )

    def update(self, info, epoch):
        self.info = info
        self.info['Epoch'] = epoch

    def upload(self, commit=True) -> None:
        wandb.log(
            self.info,
            commit=commit
        )
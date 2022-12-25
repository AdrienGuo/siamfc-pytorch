import wandb


class WandB():
    def __init__(self, name: str, config: dict) -> None:
        wandb.init(
            name=name,
            project="SiamFC",
            config=config
        )

    def upload(self, train_info: float, val_info: float, epoch, commit=True) -> None:
        wandb.log({
            'Train': {
                'Loss': train_info,
            },
            'Test': {
                'Loss': val_info,
            },
            'Epoch': epoch,
        }, commit=commit)
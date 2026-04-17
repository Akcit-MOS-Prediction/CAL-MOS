import os
import argparse

import wandb
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from models.calmos_wrapper import CALMOSWrapper

# Disable warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def run_train(config, args) -> dict:
    if config.data.get("use_seqaug", False):
        print(f"Using sequence augmentation!")

    if "tags" not in config or config.tags is None:
        raise Exception(
            f"You must add a list of tags in attribute ``tags`` in your experiment setup.\n \
                E.g.\n\
                tags:\n\
                - <your tag>"
        )

    exp_title = config.title

    tags = ["MOS-Prediction"]
    tags += [dataset["name"] for dataset in config.datasets.train]  # add training datasets as tags
    tags += config.tags  # add tags defined for experiments

    OmegaConf.resolve(config)

    checkpoint_dir = args.checkpoint_dir if getattr(args, 'checkpoint_dir', None) else config.model_checkpoint.get("dirpath", "../checkpoints/mos-prediction")
    checkpoint_dir = os.path.join(checkpoint_dir, exp_title)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_save_path = os.path.join(checkpoint_dir, "resolved_config.yaml")
    OmegaConf.save(config=config, f=config_save_path)
    print(f"Resolved config saved at: {config_save_path}")

    wandb.init(
        project="MOS-Prediction",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True),
        reinit=True
    )
    logger = WandbLogger(
        project="MOS-Prediction",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    model_checkpoint_config = dict(config["model_checkpoint"])
    model_checkpoint_config["dirpath"] = checkpoint_dir

    callbacks = [
        ModelCheckpoint(**model_checkpoint_config),
        LearningRateMonitor("step"),
        EarlyStopping(**config["early_stopping"]),
    ]

    model = CALMOSWrapper(config)

    print(model)

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
        default_root_dir=checkpoint_dir
    )

    trainer.fit(model)

    # Get the best checkpoint path from the ModelCheckpoint callback
    best_model_path = "N/A"
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
            break

    # Capture wandb run ID before closing
    wandb_run_id = wandb.run.id if wandb.run is not None else "N/A"

    # Close wandb run
    wandb.finish()

    # Return best metrics, best checkpoint path, and wandb run id
    return trainer.callback_metrics, best_model_path, wandb_run_id

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        action="append",
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        default=0,
        help="GPU device",
        type=int
    )
    parser.add_argument(
        "-ck",
        "--checkpoint-dir",
        required=False,
        type=str,
        default=None,
    )

    args = parser.parse_args()

    configs = [OmegaConf.load(path) for path in args.config_path]
    config = OmegaConf.merge(*configs)
    
    run_train(config, args)

if __name__ == "__main__":
    main()

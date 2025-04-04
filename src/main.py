import os
import argparse


import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models.calmos_wrapper import CALMOSWrapper

# Disable warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
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
        default="../checkpoints/mos-prediction",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    if config.data.get("use_seqaug", False):
        print(f"Using sequence augmentation!")

    if "tags" not in config or config.tags is None:
        raise Exception(
            f"You must add a list of tags in attribute ``tags`` in your experiment \
                setup file {args.config_path}.\n \
                E.g.\n\
                tags:\n\
                - <your tag>"
        )

    exp_title = config.title

    tags = ["MOS-Prediction"]
    tags += [dataset["name"] for dataset in config.datasets.train]  # add training datasets as tags
    tags += config.tags  # add tags defined for experiments
    wandb.init(
        project="MOS-Prediction",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )
    logger = WandbLogger(
        project="MOS-Prediction",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    config["model_checkpoint"].pop("dirpath")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]

    model = CALMOSWrapper(config)

    print(model)

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
        default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()

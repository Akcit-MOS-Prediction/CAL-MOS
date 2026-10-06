import os
import argparse

import wandb
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from models.calmos_wrapper import CALMOSWrapper
from utils.forgetting_metrics import ForgettingMetrics

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

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        help="Load model weights from the previous stage; optimizer/scheduler start fresh",
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

    checkpoint_settings = OmegaConf.to_container(config.model_checkpoint, resolve=True)
    checkpoint_settings["dirpath"] = os.path.abspath(args.checkpoint_dir)
    checkpoint_settings["save_top_k"] = 1
    checkpoint_settings["save_last"] = False

    monitor = checkpoint_settings.get("monitor")
    if not monitor:
        raise ValueError("model_checkpoint.monitor must name a validation metric")

    early_stopping_settings = (
        OmegaConf.to_container(config.early_stopping, resolve=True)
        if config.get("early_stopping") else {}
    )
    early_stopping_settings.update(
        monitor=monitor,
        mode=checkpoint_settings.get("mode", "max"),
        patience=25,
        min_delta=0.0,
        check_on_train_epoch_end=False,
    )

    callbacks = [
        ModelCheckpoint(**checkpoint_settings),
        EarlyStopping(**early_stopping_settings),
        LearningRateMonitor("step"),
    ]
    if config.get("only_mlp", False):
        protocol = config.get("forgetting_metrics") or {}
        callbacks.append(ForgettingMetrics(
            output_dir=os.path.dirname(os.path.abspath(args.checkpoint_dir)),
            val_loss_target=protocol.get("val_loss_target", 0.05),
            budget_seconds=protocol.get("budget_seconds", 1800.0),
        ))

    if args.init_checkpoint:
        if not config.get("only_mlp", False) or not config.model.get("freeze_backbone", False):
            raise ValueError("--init-checkpoint requires only_mlp=true and freeze_backbone=true")
        model = CALMOSWrapper.load_from_checkpoint(
            args.init_checkpoint, config=config, map_location="cpu", strict=True
        )
        print(f"Initialized model weights from: {args.init_checkpoint}")
    else:
        model = CALMOSWrapper(config)

    print(model)

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
        default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()

import os
import argparse

import torch
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


def tensor_fingerprint(tensor: torch.Tensor, n_values: int = 1024) -> float:
    flat = tensor.detach().float().cpu().reshape(-1)
    return float(flat[: min(n_values, flat.numel())].sum().item())


def load_finetune_weights(model: CALMOSWrapper, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Fine-tuning checkpoint not found: {checkpoint_path}")

    print(f"Loading fine-tuning weights from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    matched_keys = [
        key for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    ]
    if not matched_keys:
        raise RuntimeError(
            "O checkpoint não possui tensores compatíveis com o modelo de destino."
        )

    trainable_keys = {
        key for key, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing_trainable_keys = sorted(trainable_keys.difference(matched_keys))
    if missing_trainable_keys:
        preview = ", ".join(missing_trainable_keys[:10])
        raise RuntimeError(
            "O checkpoint não cobre todos os parâmetros treináveis; o teste de "
            f"retenção seria inválido. Ausentes ({len(missing_trainable_keys)}): {preview}"
        )

    probe_keys = matched_keys[:3]
    before = {
        key: tensor_fingerprint(model_state[key])
        for key in probe_keys
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded_state = model.state_dict()
    after = {
        key: tensor_fingerprint(loaded_state[key])
        for key in probe_keys
    }
    ckpt = {
        key: tensor_fingerprint(state_dict[key])
        for key in probe_keys
    }

    print(
        "Fine-tuning weights loaded "
        f"(checkpoint_tensors={len(state_dict)}, matched_tensors={len(matched_keys)}, "
        f"trainable_tensors={len(trainable_keys)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})."
    )
    print("Fine-tuning checkpoint verification:")
    for key in probe_keys:
        print(
            f"  {key}: before={before[key]:.6f} "
            f"checkpoint={ckpt[key]:.6f} after={after[key]:.6f}"
        )


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

    model = CALMOSWrapper(config)
    load_finetune_weights(model, config.get("finetune", {}).get("init_checkpoint"))

    print(model)

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

import os
import glob
import argparse


import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from models.calmos_wrapper import CALMOSWrapper
from eval.inference_func import inference

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
        default=None,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-cd",
        "--config_path_data",
        default=None,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-cm",
        "--config_path_model",
        default=None,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-ct",
        "--config_path_trainer",
        default=None,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-l",
        "--layer_id",
        default=None,
        type=str,
        help="Which layer to extract features from during training"
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

    if args.config_path is not None:
        config = OmegaConf.load(args.config_path)
    else:
        config_data = OmegaConf.load(args.config_path_data)
        config_model = OmegaConf.load(args.config_path_model)
        config_trainer = OmegaConf.load(args.config_path_trainer)

        if args.layer_id is not None:
            print(f"Using layer id {args.layer_id} for feature extraction.")
            config_model.model.specific_layer_idx = int(args.layer_id)

        config = OmegaConf.merge(config_data, config_model, config_trainer)

    if "tags" not in config or config.tags is None:
        raise Exception(
            f"You must add a list of tags in attribute ``tags`` in your experiment \
                setup file {args.config_path}.\n \
                E.g.\n\
                tags:\n\
                - <your tag>"
        )

    exp_title = config.title
    title_dir = config.title.replace("/", "-").replace(" ", "_")
    wandb_runs_dir = config.logger.get("runs_dir", "./experiments")

    tags = [config.logger.wandb.project]  # add project name as a tag
    tags += [dataset["name"] for dataset in config.datasets.train]  # add training datasets as tags
    tags += config.tags  # add tags defined for experiments
    wandb_run = wandb.init(
        project=config.logger.wandb.project,
        name=exp_title,
        tags=tags,
        entity=config.logger.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    wandb_run_id = wandb_run.id

    logger = WandbLogger(
        project=config.logger.wandb.project,
        name=exp_title,
        tags=tags,
        entity=config.logger.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    # save the run id to the config
    wandb_run_id_path = os.path.join(wandb_runs_dir, title_dir, "wandb_run_id.txt")
    os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
    with open(wandb_run_id_path, "a+") as f:
        f.write(wandb_run_id + "\n")

    config["model_checkpoint"].pop("dirpath")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]

    if "early_stopping" in config:
        print("Using EarlyStopping callback with config:", config["early_stopping"])
        callbacks.append(EarlyStopping(**config["early_stopping"]))

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

    # unload memory
    del model
    del trainer
    torch.cuda.empty_cache()

    checkpoint_paths = glob.glob(os.path.join("./", config.logger.wandb.project, str(wandb_run_id), "**", "*.ckpt"), recursive=True)
    # remove "last.ckpt" from the list
    checkpoint_paths = [path for path in checkpoint_paths if "last.ckpt" not in path]
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]

    results, predict_mean_scores, true_mean_scores, sys_pred_means, sys_true_means = inference(
        config=config,
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    # write results to a csv file and formated in a txt file
    results_df = pd.DataFrame(results, columns=["sys_mse", "sys_lcc", "sys_srcc", "sys_ktau", "utt_mse", "utt_lcc", "utt_srcc", "utt_ktau"])
    results_df["sys_mse"] = [results["system_level"]["MSE"]]
    results_df["sys_lcc"] = [results["system_level"]["LCC"]]
    results_df["sys_srcc"] = [results["system_level"]["SRCC"]]
    results_df["sys_ktau"] = [results["system_level"]["KTAU"]]
    results_df["utt_mse"] = [results["utterance_level"]["MSE"]]
    results_df["utt_lcc"] = [results["utterance_level"]["LCC"]]
    results_df["utt_srcc"] = [results["utterance_level"]["SRCC"]]
    results_df["utt_ktau"] = [results["utterance_level"]["KTAU"]]

    sys_level_str = f"System-level Results:\t{results['system_level']['MSE']:.3f}\t{results['system_level']['LCC']:.3f}\t{results['system_level']['SRCC']:.3f}\t{results['system_level']['KTAU']:.3f}\n"
    utt_level_str = f"Utterance-level Results:\t{results['utterance_level']['MSE']:.3f}\t{results['utterance_level']['LCC']:.3f}\t{results['utterance_level']['SRCC']:.3f}\t{results['utterance_level']['KTAU']:.3f}\n"

    predictions_utt = pd.DataFrame({
        "predicted_mean_scores": predict_mean_scores,
        "true_mean_scores": true_mean_scores
    })

    predictions_sys = pd.DataFrame({
        "predicted_sys_mean_scores": sys_pred_means,
        "true_sys_mean_scores": sys_true_means
    })

    # save files
    results_df.to_csv(os.path.join(wandb_runs_dir, title_dir, "results.csv"), index=False)
    predictions_utt.to_csv(os.path.join(wandb_runs_dir, title_dir, "predictions_utt.csv"), index=False)
    predictions_sys.to_csv(os.path.join(wandb_runs_dir, title_dir, "predictions_sys.csv"), index=False)
    with open(os.path.join(wandb_runs_dir, title_dir, "results.txt"), "w") as f:
        f.write("Metric\tMSE\tLCC\tSRCC\tKTAU\n")
        f.write(sys_level_str)
        f.write(utt_level_str)


if __name__ == "__main__":
    main()

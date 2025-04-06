import os
import argparse

import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models.calmos_wrapper import CALMOSWrapper
from eval.inference import run_inference

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


def main() -> None:
    print("[MAIN] Iniciando o script...")
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
        "-ck", "--checkpoint-dir", required=False, type=str,
        default="../checkpoints/mos-prediction",
        help="Diretório base para salvar os checkpoints"
    )
    parser.add_argument(
        "--datasets", nargs='+', type=str, default=["test"],
        choices=["test", "val"],
        help="Conjuntos de dados a serem avaliados. Exemplo: --datasets val test"
    )

    args = parser.parse_args()
    print("[MAIN] Argumentos parseados:", args)

    print("[MAIN] Carregando config de:", args.config_path)
    config = OmegaConf.load(args.config_path)
    print("[MAIN] Config carregada:")
    print(OmegaConf.to_yaml(config))

    exp_title = config.title
    print("[MAIN] Título da experiência:", exp_title)

    if config.data.get("use_seqaug", False):
        print("[MAIN] Sequence augmentation está ativada!")

    if "tags" not in config or config.tags is None:
        raise Exception(
            f"You must add a list of tags in attribute ``tags`` in your experiment setup file {args.config_path}."
        )

    tags = ["MOS-Prediction"]
    # Adiciona os nomes dos datasets de treino como tags
    train_datasets = config.datasets.train
    print("[MAIN] Datasets de treino encontrados:", train_datasets)
    tags += [dataset["name"] for dataset in train_datasets]
    tags += config.tags  # Adiciona as tags definidas
    print("[MAIN] Tags finais:", tags)

    print("[MAIN] Inicializando wandb...")
    wandb.init(
        project="MOS-Prediction",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )
    print("[MAIN] Wandb init finalizado.")

    logger = WandbLogger(
        project="MOS-Prediction",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )
    print("[MAIN] Logger Wandb criado.")

    # Remove dirpath de model_checkpoint
    if "dirpath" in config["model_checkpoint"]:
        popped = config["model_checkpoint"].pop("dirpath")
        print("[MAIN] Removido dirpath do model_checkpoint:", popped)
    else:
        print("[MAIN] dirpath não encontrado em model_checkpoint.")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]
    print("[MAIN] Callbacks configurados.")

    print("[MAIN] Criando o modelo com CALMOSWrapper...")
    model = CALMOSWrapper(config)
    print("[MAIN] Modelo criado:", model.__class__.__name__)

    print("[MAIN] Criando o trainer...")
    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
        default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
    )
    print("[MAIN] Trainer criado. Iniciando treinamento com trainer.fit(model)...")
    trainer.fit(model)
    print("[MAIN] Treinamento finalizado.")

    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    print("[MAIN] Melhor checkpoint salvo em:", best_checkpoint_path)

    print("[MAIN] Carregando o modelo a partir do checkpoint...")
    model = CALMOSWrapper.load_from_checkpoint(best_checkpoint_path, config=config)
    print("[MAIN] Modelo carregado do checkpoint.")

    for dataset in args.datasets:
        print(f"\n=== Rodando inferência em: {dataset} ===")
        mse, lcc, srcc, tau = run_inference(
            config_path=args.config_path,
            gpu=args.gpu,
            checkpoint_path=best_checkpoint_path,
            dataset=dataset,
            batch_size=config.train.batch_size
        )
        print(f"Resultados de {dataset}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  LCC:  {lcc:.4f}")
        print(f"  SRCC: {srcc:.4f}")
        print(f"  KTAU: {tau:.4f}")

        wandb.log({
            f"{dataset}_MSE": mse,
            f"{dataset}_LCC": lcc,
            f"{dataset}_SRCC": srcc,
            f"{dataset}_KTAU": tau,
        })

    print("[MAIN] Processo finalizado com sucesso.")


if __name__ == "__main__":
    main()

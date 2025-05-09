import torch
from torch.nn import MSELoss
from lion_pytorch import Lion
import pytorch_lightning as pl
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from transformers import AutoFeatureExtractor
from lightning.pytorch.utilities import grad_norm
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef

from models.factory import create_model
from utils.utils import build_dataloaders
from utils.schedulers import CosineWarmupLR, LinearLR
from utils.dataloader import (
    AllLayersEmbeddingCollate,
    OneLayerEmbeddingCollate,
    DynamicCollate,
    DynamicAudioCollate,
)


class CALMOSWrapper(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)

        self.config = config

        self.model = create_model(
            **config.model
        )
        self.loss = MSELoss()
        # Metrics
        # Training
        self.train_mse = MeanSquaredError()
        self.train_pearson = PearsonCorrCoef()
        self.train_spearman = SpearmanCorrCoef()
        # Validation
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset, self.val_dataset = build_dataloaders(self.config)

    def train_dataloader(self):
        """Return the training dataloader."""
        if self.config.model.model_type.lower() == "dynamic":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.model_name)
            collate_fn = DynamicCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
            )
        elif self.config.model.model_type.lower() == "dynamic_melspec":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.model_name)
            collate_fn = DynamicAudioCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
            )
        elif self.config.model.model_type.lower() == "all_layers_embedding":
            collate_fn = AllLayersEmbeddingCollate()
        elif self.config.model.model_type.lower() == "one_layer_embedding":
            collate_fn = OneLayerEmbeddingCollate()
        else:
            raise ValueError(f"Invalid model type: {self.config.model.model_type}")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        if self.config.model.model_type.lower() == "dynamic":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.model_name)
            collate_fn = DynamicCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
            )
        elif self.config.model.model_type.lower() == "dynamic_melspec":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.model_name)
            collate_fn = DynamicAudioCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
            )
        elif self.config.model.model_type.lower() == "all_layers_embedding":
            collate_fn = AllLayersEmbeddingCollate()
        elif self.config.model.model_type.lower() == "one_layer_embedding":
            collate_fn = OneLayerEmbeddingCollate()
        else:
            raise ValueError(f"Invalid model type: {self.config.model.model_type}")

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        dataset_size = len(dataset)

        gpu_count = self.trainer.num_devices if self.trainer.num_devices else 1
        accumulate_grad_batches = self.trainer.accumulate_grad_batches

        effective_batches = dataset_size // (gpu_count * accumulate_grad_batches)

        return effective_batches * self.trainer.max_epochs

    def configure_optimizers(self):
        """Configures the optimizer and the learning rate scheduler."""
        # Start dataloaders to be able to get the number of steps per epoch
        self.trainer.fit_loop.setup_data()

        max_num_steps = self.num_training_steps()

        print(f"Max number of steps: {max_num_steps}")

        opt_params = self.config.optimizer["params"]
        scheduler_params = self.config.scheduler["params"]

        if self.config.optimizer.name.lower() == "adam":
            optimizer = Adam(
                self.parameters(),
                eps=opt_params["eps"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"]
            )

        elif self.config.optimizer.name.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(),
                eps=opt_params["eps"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"]
            )

        elif self.config.optimizer.name.lower() == "lion":
            optimizer = Lion(
                self.parameters(),
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"],
                use_triton=opt_params.get("use_triton", False),
            )

        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer.name}")

        if not self.config["scheduler"]:
            return optimizer

        scheduler = None
        if self.config.scheduler.name.lower() == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=scheduler_params.get("patience", max_num_steps*0.25),
                factor=0.9,
                min_lr=opt_params.get("min_learning_rate", 1.0e-6)
            )

        if self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=opt_params.get("min_learning_rate", 1.0e-6),
                lr_max=opt_params["learning_rate"],
                warmup=scheduler_params.get("warmup_lr", max_num_steps*0.05),
                T_max=max_num_steps
            )

        if self.config.scheduler.name.lower() == "linearlr":
            scheduler = LinearLR(
                optimizer,
                start_factor=scheduler_params.get("start_factor", 1.0 / 3.0),
                end_factor=scheduler_params.get("end_factor", 1.0),
                total_iters=scheduler_params.get("total_iters", 5),
                last_epoch=scheduler_params.get("last_epoch", -1),
                verbose=scheduler_params.get("verbose", False)
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def forward(self, x):
        """Forward pass for the model."""
        return self.model(x)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def training_step(self, train_batch, batch_idx):
        """Training step."""
        input_features, target = train_batch

        logits = self.forward(input_features)
        loss = self.loss(logits, target)

        self.train_mse(logits, target)
        self.train_pearson(logits, target)
        self.train_spearman(logits, target)

        self.log("train/loss", loss)
        self.log("train/mse", self.train_mse, on_step=True, on_epoch=True)
        self.log("train/pearson", self.train_pearson, on_step=True, on_epoch=True)
        self.log("train/spearman", self.train_spearman, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        """Validation step."""
        input_features, target = val_batch

        logits = self.forward(input_features)
        loss = self.loss(logits, target)

        self.val_mse(logits, target)
        self.val_pearson(logits, target)
        self.val_spearman(logits, target)

        self.log("val/loss", loss)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True)
        self.log("val/pearson", self.val_pearson, on_step=False, on_epoch=True)
        self.log("val/spearman", self.val_spearman, on_step=False, on_epoch=True)

        return loss
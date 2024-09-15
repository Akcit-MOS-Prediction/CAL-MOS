import torch
from torch.nn import MSELoss
from lion_pytorch import Lion
import pytorch_lightning as pl
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from lightning.pytorch.utilities import grad_norm
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef

from models.base_models import CalmosModel
from utils.schedulers import CosineWarmupLR, LinearLR

class CALMOSWrapper(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)

        self.config = config

        self.model = CalmosModel(
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

    def configure_optimizers(self):
        """Configures the optimizer and the learning rate scheduler."""
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
                patience=scheduler_params.get("patience", self.trainer.max_steps*0.25),
                factor=0.9,
                min_lr=opt_params.get("min_learning_rate", 1.0e-6)
            )

        if self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=opt_params.get("min_learning_rate", 1.0e-6),
                lr_max=opt_params["learning_rate"],
                warmup=scheduler_params.get("warmup_lr", self.trainer.max_steps*0.05),
                T_max=self.trainer.max_steps
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

        self.log("train_loss", loss)
        self.log("train_mse", self.train_mse, on_step=True, on_epoch=True)
        self.log("train_pearson", self.train_pearson, on_step=True, on_epoch=True)
        self.log("train_spearman", self.train_spearman, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        """Validation step."""
        input_features, target = val_batch

        logits = self.forward(input_features)
        loss = self.loss(logits, target)

        self.val_mse(logits, target)
        self.val_pearson(logits, target)
        self.val_spearman(logits, target)

        self.log("val_loss", loss)
        self.log("val_mse", self.val_mse, on_step=False, on_epoch=True)
        self.log("val_pearson", self.val_pearson, on_step=False, on_epoch=True)
        self.log("val_spearman", self.val_spearman, on_step=False, on_epoch=True)

        return loss
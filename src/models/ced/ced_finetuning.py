import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from models.ced.audiotransformer import (
    ced_tiny, # 5.5M
    ced_mini, # 9.6M
    ced_small, # 22M
    ced_base # 86M
)


class FineTuneCED(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 768,
        proj_size: int = 512,
        proj_dropout: float = 0.2,
        freeze_backbone_flag: bool = False,
    ):
        super().__init__()

        if embedding_dim==128:
            self.backbone = ced_tiny(pretrained=pretrained)
        elif embedding_dim==256:
            self.backbone = ced_mini(pretrained=pretrained)
        elif embedding_dim==384:
            self.backbone = ced_small(pretrained=pretrained)
        elif embedding_dim==768:
            self.backbone = ced_base(pretrained=pretrained)
        else:
            raise ValueError("embedding_dim must be one of 128, 256, 384, 768")

        if freeze_backbone_flag:
            self.freeze_backbone()

        self.proj_layer = nn.Sequential(
            nn.Linear(embedding_dim, proj_size),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_size, proj_size),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.backbone(x)
        embeddings = self.proj_layer(x)
        return embeddings
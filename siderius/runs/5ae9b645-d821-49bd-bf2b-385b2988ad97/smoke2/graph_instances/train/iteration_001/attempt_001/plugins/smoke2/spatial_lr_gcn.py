import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from typing import Self

from typing import Self

PLUGIN_MODEL_TYPE = "spatial_lr_gcn"


class SpatialLrGcnConfig(BaseModel):
    model_type: str = Field(default="spatial_lr_gcn", description="Plugin model type key.")
    segmentation_size: int = Field(ge=1)
    batch_size: int = Field(default=1, ge=1)
    hidden_dim: int = Field(default=64, ge=16, le=256)
    num_layers: int = Field(default=3, ge=1, le=5)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)


PLUGIN_CONFIG_CLASS = SpatialLrGcnConfig


class SpatialLrGcn(nn.Module):
    def __init__(self, config: "SpatialLrGcnConfig"):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            in_dim = 5 if i == 0 else self.hidden_dim
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ))
        self.readout = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward contract: input [B, 5, 1024] float32 → output [B, 1] float32
        x_perm = x.permute(0, 2, 1)  # [B, 1024, 5]
        mask = x[:, 4, :].unsqueeze(-1)  # [B, 1024, 1]
        h = x_perm
        for layer in self.layers:
            h = layer(h)
        scores = self.readout(h)  # [B, 1024, 1]
        masked_scores = scores * mask
        sum_scores = masked_scores.sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return sum_scores / count


PLUGIN_MODEL_CLASS = SpatialLrGcn
PLUGIN_OUTPUT_TYPE = "regressor"  # [B, 1] → continuous [B, 1] output

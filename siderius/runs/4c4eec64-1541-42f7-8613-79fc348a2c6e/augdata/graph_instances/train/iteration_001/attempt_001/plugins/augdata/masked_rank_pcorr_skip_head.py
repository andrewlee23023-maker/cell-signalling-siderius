import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from typing import Self


PLUGIN_MODEL_TYPE = "masked_rank_pcorr_skip_head"


class MaskedRankPcorrSkipHeadConfig(BaseModel):
    model_type: str = Field(default="masked_rank_pcorr_skip_head", description="Plugin model type key.")
    segmentation_size: int = Field(ge=1)
    batch_size: int = Field(default=64, ge=1)
    hidden_dim: int = Field(default=16, ge=2, le=256)
    num_hidden_layers: int = Field(default=2, ge=1, le=4)
    use_head: bool = Field(default=True)
    use_skip: bool = Field(default=True)
    skip_scale: float = Field(default=0.2, ge=0.0, le=10.0)
    z_clip: float = Field(default=15.0, gt=0.1, le=100.0)
    min_contacts: int = Field(default=5, ge=1, le=1024)
    ridge_eps: float = Field(default=0.001, gt=1e-6, le=1.0)
    max_contacts: int = Field(default=1024, ge=1, le=1024)
    @model_validator(mode='after')
    def check_constraints(self):
        if self.min_contacts > self.max_contacts:
            raise ValueError(f'min_contacts ({self.min_contacts}) must be less than or equal to max_contacts ({self.max_contacts})')
        return self

PLUGIN_CONFIG_CLASS = MaskedRankPcorrSkipHeadConfig


class MaskedRankPcorrSkipHead(nn.Module):
    def __init__(self, config: "MaskedRankPcorrSkipHeadConfig"):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.use_head = config.use_head
        self.use_skip = config.use_skip
        self.skip_scale = config.skip_scale
        self.z_clip = config.z_clip
        self.min_contacts = config.min_contacts
        self.ridge_eps = config.ridge_eps
        self.max_contacts = config.max_contacts
        head_layers = [nn.Linear(5, self.hidden_dim), nn.ReLU()]
        head_layers += [module for _ in range(self.num_hidden_layers - 1) for module in (nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())]
        head_layers += [nn.Linear(self.hidden_dim, 1)]
        self.head = nn.Sequential(*head_layers)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward contract: input [B, 5, 1024] float32 → output [B, 1] float32
        x = x.float()
        m_bool = x[:, 4, :] > 0.5
        m = m_bool.to(x.dtype)
        n = m.sum(dim=1, keepdim=True)
        n_safe = n.clamp_min(1.0)
        variables = (x[:, 0, :], x[:, 1, :], x[:, 2, :], x[:, 3, :])
        rank_values = []
        for v in variables:
            less = v.unsqueeze(2) > v.unsqueeze(1)
            equal = v.unsqueeze(2) == v.unsqueeze(1)
            rank = (less.to(x.dtype) * m.unsqueeze(1)).sum(dim=2)
            rank = rank + 0.5 * (equal.to(x.dtype) * m.unsqueeze(1)).sum(dim=2)
            rank_values.append(rank * m)
        ranks = torch.stack(rank_values, dim=1)
        mask_channels = m.unsqueeze(1)
        means = (ranks * mask_channels).sum(dim=2, keepdim=True) / n_safe.unsqueeze(1)
        centered = (ranks - means) * mask_channels
        variance = (centered.square() * mask_channels).sum(dim=2, keepdim=True) / n_safe.unsqueeze(1)
        z_ranks = centered / torch.sqrt(variance + self.ridge_eps)
        z_ranks = z_ranks * mask_channels
        correlation = torch.bmm(z_ranks, (z_ranks * mask_channels).transpose(1, 2)) / n_safe.unsqueeze(2)
        identity = torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0)
        precision = torch.linalg.inv(correlation + self.ridge_eps * identity)
        denominator = torch.sqrt((precision[:, 0, 0] * precision[:, 1, 1]).clamp_min(self.ridge_eps))
        r_partial = -precision[:, 0, 1] / denominator
        r_raw = correlation[:, 0, 1]
        effective_n = n.squeeze(1)
        valid = effective_n >= self.min_contacts
        r_partial = torch.where(valid, r_partial, torch.zeros_like(r_partial))
        r_partial_clipped = r_partial.clamp(-0.999, 0.999)
        scale = torch.sqrt((effective_n - 3.0).clamp_min(1.0))
        z = torch.atanh(r_partial_clipped) * scale
        z = z.clamp(-self.z_clip, self.z_clip)
        z = torch.where(valid, z, torch.zeros_like(z))
        feature_contact = torch.log1p(effective_n) / torch.log1p(torch.tensor(float(self.max_contacts), device=x.device, dtype=x.dtype))
        features = torch.stack((r_partial, z / self.z_clip, r_raw, feature_contact, r_partial - r_raw), dim=1)
        head_output = self.head(features)
        z_column = z.unsqueeze(1)
        logit = self.use_head * head_output + self.use_skip * self.skip_scale * z_column
        output = torch.sigmoid(logit).to(torch.float32)
        return output


PLUGIN_MODEL_CLASS = MaskedRankPcorrSkipHead
PLUGIN_OUTPUT_TYPE = "regressor"  # [B, 1] → continuous [B, 1] output

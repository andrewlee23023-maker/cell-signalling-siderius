import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from typing import Self


PLUGIN_MODEL_TYPE = "masked_contact_coupling_net"


class MaskedContactCouplingNetConfig(BaseModel):
    model_type: str = Field(default="masked_contact_coupling_net", description="Plugin model type key.")
    segmentation_size: int = Field(ge=1)
    batch_size: int = Field(default=32, ge=1)
    hidden_dim: int = Field(default=32, ge=4, le=512)
    num_layers: int = Field(default=2, ge=1, le=4)
    dropout: float = Field(default=0.1, ge=0.0, lt=0.8)
    ridge_lambda: float = Field(default=0.001, gt=0.0)
    min_contacts: int = Field(default=3, ge=1, le=1024)
    eps: float = Field(default=1e-6, gt=0.0)
    residualize: bool = Field(default=True)
    use_rank_stats: bool = Field(default=True)
    use_weighted_corr: bool = Field(default=True)
    use_log_n: bool = Field(default=True)


PLUGIN_CONFIG_CLASS = MaskedContactCouplingNetConfig


class MaskedContactCouplingNet(nn.Module):
    def __init__(self, config: "MaskedContactCouplingNetConfig"):
        super().__init__()
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.ridge_lambda = config.ridge_lambda
        self.min_contacts = config.min_contacts
        self.eps = config.eps
        self.residualize = config.residualize
        self.use_rank_stats = config.use_rank_stats
        self.use_weighted_corr = config.use_weighted_corr
        self.use_log_n = config.use_log_n
        self.feature_norm = nn.LayerNorm(11)
        head_layers = [nn.Linear(11, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout)]
        head_layers.extend([module for _ in range(self.num_layers - 1) for module in (nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout))])
        head_layers.extend([nn.Linear(self.hidden_dim, 1)])
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward contract: input [B, 5, 1024] float32 → output [B, 1] float32
        m = x[:, 4, :].to(x.dtype)
        values = x[:, :4, :] * m[:, None, :]
        ligand = values[:, 0, :]
        receptor = values[:, 1, :]
        library_offset = values[:, 2, :]
        sender_fraction = values[:, 3, :]
        n = m.sum(dim=1)
        valid = n >= self.min_contacts
        if self.residualize:
            ones = torch.ones_like(library_offset)
            Z = torch.stack((ones, library_offset, sender_fraction), dim=-1)
            Y = torch.stack((ligand, receptor), dim=-1)
            masked_Z = Z * m[..., None]
            A = Z.transpose(1, 2) @ masked_Z
            ridge_eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
            A = A + self.ridge_lambda * ridge_eye
            rhs = Z.transpose(1, 2) @ (Y * m[..., None])
            beta = torch.linalg.solve(A, rhs)
            predicted = Z @ beta
            residuals = (Y - predicted) * m[..., None]
            e_L = residuals[:, :, 0]
            e_R = residuals[:, :, 1]
        else:
            e_L = ligand
            e_R = receptor
        denom_n = n.clamp_min(1).unsqueeze(-1)
        mu_L = (m * e_L).sum(dim=1, keepdim=True) / denom_n
        mu_R = (m * e_R).sum(dim=1, keepdim=True) / denom_n
        c_L = e_L - mu_L
        c_R = e_R - mu_R
        denom_var = (n - 1).clamp_min(1).unsqueeze(-1)
        var_L = (m * c_L.square()).sum(dim=1, keepdim=True) / denom_var
        var_R = (m * c_R.square()).sum(dim=1, keepdim=True) / denom_var
        cov = (m * c_L * c_R).sum(dim=1, keepdim=True) / denom_var
        sd_L = torch.sqrt(var_L.clamp_min(0) + self.eps)
        sd_R = torch.sqrt(var_R.clamp_min(0) + self.eps)
        rho = cov / torch.sqrt(var_L * var_R + self.eps)
        rho = rho.clamp(-1 + self.eps, 1 - self.eps)
        slope = cov / (var_L + self.eps)
        t = rho * torch.sqrt((n - 2).clamp_min(0).unsqueeze(-1) / (1 - rho.square() + self.eps))
        z = torch.atanh(rho.clamp(-1 + self.eps, 1 - self.eps))
        z = z * torch.sqrt((n - 3).clamp_min(0).unsqueeze(-1))
        if self.use_rank_stats:
            with torch.no_grad():
                rank_key_L = torch.where(m > 0, e_L, torch.full_like(e_L, float('inf')))
                rank_key_R = torch.where(m > 0, e_R, torch.full_like(e_R, float('inf')))
                order_L = torch.argsort(rank_key_L, dim=1)
                order_R = torch.argsort(rank_key_R, dim=1)
                rank_positions = torch.arange(x.shape[1] * 0 + x.shape[2], device=x.device, dtype=x.dtype).unsqueeze(0).expand_as(e_L)
                rank_L = torch.zeros_like(e_L)
                rank_R = torch.zeros_like(e_R)
                rank_L.scatter_(1, order_L, rank_positions)
                rank_R.scatter_(1, order_R, rank_positions)
            rank_mu_L = (m * rank_L).sum(dim=1, keepdim=True) / denom_n
            rank_mu_R = (m * rank_R).sum(dim=1, keepdim=True) / denom_n
            rank_c_L = rank_L - rank_mu_L
            rank_c_R = rank_R - rank_mu_R
            rank_var_L = (m * rank_c_L.square()).sum(dim=1, keepdim=True) / denom_var
            rank_var_R = (m * rank_c_R.square()).sum(dim=1, keepdim=True) / denom_var
            rank_cov = (m * rank_c_L * rank_c_R).sum(dim=1, keepdim=True) / denom_var
            rho_s = rank_cov / torch.sqrt(rank_var_L * rank_var_R + self.eps)
            rho_s = rho_s.clamp(-1 + self.eps, 1 - self.eps)
        else:
            rho_s = torch.zeros_like(rho)
        if self.use_weighted_corr:
            w = m * sender_fraction
            sum_w = w.sum(dim=1, keepdim=True).clamp_min(1)
            mu_wL = (w * e_L).sum(dim=1, keepdim=True) / sum_w
            mu_wR = (w * e_R).sum(dim=1, keepdim=True) / sum_w
            wc_L = e_L - mu_wL
            wc_R = e_R - mu_wR
            weighted_var_L = (w * wc_L.square()).sum(dim=1, keepdim=True) / sum_w
            weighted_var_R = (w * wc_R.square()).sum(dim=1, keepdim=True) / sum_w
            weighted_cov = (w * wc_L * wc_R).sum(dim=1, keepdim=True) / sum_w
            weighted_rho = weighted_cov / torch.sqrt(weighted_var_L * weighted_var_R + self.eps)
            weighted_rho = weighted_rho.clamp(-1 + self.eps, 1 - self.eps)
        else:
            weighted_rho = torch.zeros_like(rho)
        raw_mean_L = (m * ligand).sum(dim=1, keepdim=True) / denom_n
        raw_mean_R = (m * receptor).sum(dim=1, keepdim=True) / denom_n
        log_n = torch.log1p(n).unsqueeze(-1) if self.use_log_n else torch.zeros_like(rho)
        features = torch.cat((rho, slope, t, z, rho_s, weighted_rho, sd_L, sd_R, raw_mean_L, raw_mean_R, log_n), dim=1)
        features = features * valid.to(features.dtype).unsqueeze(1)
        squashed = torch.sign(features) * torch.log1p(torch.abs(features))
        normalized = self.feature_norm(squashed)
        score = self.head(normalized)
        score = torch.nan_to_num(score).float()
        return score


PLUGIN_MODEL_CLASS = MaskedContactCouplingNet
PLUGIN_OUTPUT_TYPE = "regressor"  # [B, 1] → continuous [B, 1] output

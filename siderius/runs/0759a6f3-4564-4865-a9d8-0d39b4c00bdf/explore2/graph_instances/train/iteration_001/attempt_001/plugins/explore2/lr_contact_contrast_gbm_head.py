import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from typing import Self


PLUGIN_MODEL_TYPE = "lr_contact_contrast_gbm_head"


class LrContactContrastGbmHeadConfig(BaseModel):
    model_type: str = Field(default="lr_contact_contrast_gbm_head", description="Plugin model type key.")
    segmentation_size: int = Field(ge=1)
    batch_size: int = Field(default=64, ge=1)
    n_hidden: int = Field(default=2, ge=1, le=4)
    hidden_dim: int = Field(default=64, ge=8, le=256)
    dropout: float = Field(default=0.1, ge=0.0, lt=0.8)
    use_deepset: bool = True
    d_phi: int = Field(default=32, ge=8, le=128)
    d_set: int = Field(default=32, ge=8, le=128)
    shrink_n0: float = Field(default=20.0, ge=0.0, le=200.0)
    lambda_ridge: float = Field(default=0.001, gt=0.0, le=1.0)
    use_skip_spearman: bool = True
    eps: float = Field(default=1e-6, gt=0.0, le=1e-2)
    logit_clip: float = Field(default=20.0, ge=1.0, le=100.0)


PLUGIN_CONFIG_CLASS = LrContactContrastGbmHeadConfig


class LrContactContrastGbmHead(nn.Module):
    def __init__(self, config: "LrContactContrastGbmHeadConfig"):
        super().__init__()
        self.n_hidden = config.n_hidden
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.use_deepset = config.use_deepset
        self.d_phi = config.d_phi
        self.d_set = config.d_set
        self.shrink_n0 = config.shrink_n0
        self.lambda_ridge = config.lambda_ridge
        self.use_skip_spearman = config.use_skip_spearman
        self.eps = config.eps
        self.logit_clip = config.logit_clip
        statistics_dim = 23 + (1 if self.use_skip_spearman else 0)
        self.feature_dim = statistics_dim + (self.d_set if self.use_deepset else 0)
        if self.use_deepset:
            self.set_phi = nn.Sequential(
                nn.Linear(6, self.d_phi),
                nn.GELU(),
                nn.Linear(self.d_phi, self.d_set),
            )
        head_layers = [nn.LayerNorm(self.feature_dim)]
        head_layers.extend(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim if i == 0 else self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                )
                for i in range(self.n_hidden)
            ]
        )
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.prediction_head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward contract: input [B, 5, 1024] float32 → output [B, 1] float32
        x = x.float()
        L = x[:, 0, :]
        R = x[:, 1, :]
        lib = x[:, 2, :]
        nfrac = x[:, 3, :]
        mask = x[:, 4, :].clamp(0.0, 1.0)
        eps = self.eps
        mask_sum = mask.sum(dim=1, keepdim=True)
        n = mask_sum.clamp_min(3.0)

        def _standardize(v):
            mean = (v * mask).sum(dim=1, keepdim=True) / n
            centered = v - mean
            var = (centered.square() * mask).sum(dim=1, keepdim=True) / n
            return centered / (var.sqrt() + eps) * mask

        zL = _standardize(L)
        zR = _standardize(R)
        z_lib = _standardize(lib)
        z_nfrac = _standardize(nfrac)

        with torch.no_grad():
            valid_count = mask.sum(dim=1).long().clamp_min(1)
            rankL_u = torch.zeros_like(L)
            rankR_u = torch.zeros_like(R)

            values = torch.where(mask > 0, L, torch.full_like(L, float('inf')))
            sorted_values, order = torch.sort(values, dim=1)
            positions = torch.arange(1, L.shape[1] + 1, device=L.device, dtype=L.dtype).unsqueeze(0).expand_as(L)
            changes = torch.ones_like(sorted_values, dtype=torch.long)
            changes[:, 1:] = (sorted_values[:, 1:] != sorted_values[:, :-1]).long()
            groups = changes.cumsum(dim=1) - 1
            group_sum = torch.zeros_like(L).scatter_add(1, groups, positions)
            group_count = torch.zeros_like(L).scatter_add(1, groups, torch.ones_like(positions))
            sorted_ranks = (group_sum.gather(1, groups) / group_count.gather(1, groups)).to(L.dtype)
            normalized = (sorted_ranks - 1.0) / (valid_count.to(L.dtype).unsqueeze(1) - 1.0).clamp_min(1.0)
            normalized = torch.where(mask.gather(1, order) > 0, normalized, torch.zeros_like(normalized))
            rankL_u.scatter_(1, order, normalized)
            rankL_u = rankL_u * mask

            values = torch.where(mask > 0, R, torch.full_like(R, float('inf')))
            sorted_values, order = torch.sort(values, dim=1)
            changes = torch.ones_like(sorted_values, dtype=torch.long)
            changes[:, 1:] = (sorted_values[:, 1:] != sorted_values[:, :-1]).long()
            groups = changes.cumsum(dim=1) - 1
            group_sum = torch.zeros_like(R).scatter_add(1, groups, positions)
            group_count = torch.zeros_like(R).scatter_add(1, groups, torch.ones_like(positions))
            sorted_ranks = (group_sum.gather(1, groups) / group_count.gather(1, groups)).to(R.dtype)
            normalized = (sorted_ranks - 1.0) / (valid_count.to(R.dtype).unsqueeze(1) - 1.0).clamp_min(1.0)
            normalized = torch.where(mask.gather(1, order) > 0, normalized, torch.zeros_like(normalized))
            rankR_u.scatter_(1, order, normalized)
            rankR_u = rankR_u * mask

        rankL = _standardize(rankL_u)
        rankR = _standardize(rankR_u)

        r_pearson = (zL * zR * mask).sum(dim=1, keepdim=True) / n
        r_spearman = (rankL * rankR * mask).sum(dim=1, keepdim=True) / n
        r_pearson = r_pearson.clamp(-1.0 + eps, 1.0 - eps)
        r_spearman = r_spearman.clamp(-1.0 + eps, 1.0 - eps)
        t_pearson = r_pearson * (n - 2.0).clamp_min(1.0).sqrt() / (1.0 - r_pearson.square() + eps).sqrt()
        t_spearman = r_spearman * (n - 2.0).clamp_min(1.0).sqrt() / (1.0 - r_spearman.square() + eps).sqrt()

        C = torch.stack([torch.ones_like(z_lib), z_lib, z_nfrac, z_lib * z_nfrac], dim=-1)
        weighted_C = C * mask.unsqueeze(-1)
        gram = torch.einsum('bni,bnj->bij', weighted_C, C)
        gram = gram + self.lambda_ridge * torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0)
        qL = torch.einsum('bni,bn->bi', weighted_C, zL).unsqueeze(-1)
        qR = torch.einsum('bni,bn->bi', weighted_C, zR).unsqueeze(-1)
        qL_rank = torch.einsum('bni,bn->bi', weighted_C, rankL).unsqueeze(-1)
        qR_rank = torch.einsum('bni,bn->bi', weighted_C, rankR).unsqueeze(-1)
        betaL = torch.linalg.solve(gram, qL)
        betaR = torch.linalg.solve(gram, qR)
        betaL_rank = torch.linalg.solve(gram, qL_rank)
        betaR_rank = torch.linalg.solve(gram, qR_rank)
        eL_raw = (zL - torch.einsum('bni,bij->bn', C, betaL)).mul(mask)
        eR_raw = (zR - torch.einsum('bni,bij->bn', C, betaR)).mul(mask)
        eL_rank = (rankL - torch.einsum('bni,bij->bn', C, betaL_rank)).mul(mask)
        eR_rank = (rankR - torch.einsum('bni,bij->bn', C, betaR_rank)).mul(mask)

        def _corr(a, b, weights):
            count = weights.sum(dim=1, keepdim=True).clamp_min(eps)
            mean_a = (a * weights).sum(dim=1, keepdim=True) / count
            mean_b = (b * weights).sum(dim=1, keepdim=True) / count
            ca = a - mean_a
            cb = b - mean_b
            covariance = (ca * cb * weights).sum(dim=1, keepdim=True) / count
            variance_a = (ca.square() * weights).sum(dim=1, keepdim=True) / count
            variance_b = (cb.square() * weights).sum(dim=1, keepdim=True) / count
            return covariance / (variance_a * variance_b + eps).sqrt()

        r_partial = _corr(eL_raw, eR_raw, mask).clamp(-1.0 + eps, 1.0 - eps)
        r_partial_spearman = _corr(eL_rank, eR_rank, mask).clamp(-1.0 + eps, 1.0 - eps)
        slope_raw = (eL_raw * eR_raw * mask).sum(dim=1, keepdim=True) / (eL_raw.square() * mask).sum(dim=1, keepdim=True).add(eps)
        slope_rank = (eL_rank * eR_rank * mask).sum(dim=1, keepdim=True) / (eL_rank.square() * mask).sum(dim=1, keepdim=True).add(eps)
        t_partial = r_partial * (n - 2.0).clamp_min(1.0).sqrt() / (1.0 - r_partial.square() + eps).sqrt()
        t_partial_spearman = r_partial_spearman * (n - 2.0).clamp_min(1.0).sqrt() / (1.0 - r_partial_spearman.square() + eps).sqrt()

        top_mask = mask * (rankL_u >= (2.0 / 3.0))
        bottom_mask = mask * (rankL_u <= (1.0 / 3.0))
        top_count = top_mask.sum(dim=1, keepdim=True).clamp_min(eps)
        bottom_count = bottom_mask.sum(dim=1, keepdim=True).clamp_min(eps)
        top_mean = (zR * top_mask).sum(dim=1, keepdim=True) / top_count
        bottom_mean = (zR * bottom_mask).sum(dim=1, keepdim=True) / bottom_count
        top_var = ((zR - top_mean).square() * top_mask).sum(dim=1, keepdim=True) / top_count
        bottom_var = ((zR - bottom_mean).square() * bottom_mask).sum(dim=1, keepdim=True) / bottom_count
        tercile_t = (top_mean - bottom_mean) / (top_var / top_count + bottom_var / bottom_count + eps).sqrt()

        with torch.no_grad():
            sorted_nfrac = torch.sort(torch.where(mask > 0, nfrac, torch.full_like(nfrac, float('inf'))), dim=1).values
            median_index = ((valid_count - 1) // 2).unsqueeze(1)
            median_nfrac = sorted_nfrac.gather(1, median_index)
        low_mask = mask * (nfrac <= median_nfrac)
        high_mask = mask * (nfrac > median_nfrac)
        r_partial_low = _corr(eL_raw, eR_raw, low_mask).clamp(-1.0 + eps, 1.0 - eps)
        r_partial_high = _corr(eL_raw, eR_raw, high_mask).clamp(-1.0 + eps, 1.0 - eps)
        split_difference = r_partial_high - r_partial_low

        mean_L = (L * mask).sum(dim=1, keepdim=True) / n
        mean_R = (R * mask).sum(dim=1, keepdim=True) / n
        std_L = (((L - mean_L).square() * mask).sum(dim=1, keepdim=True) / n).sqrt()
        std_R = (((R - mean_R).square() * mask).sum(dim=1, keepdim=True) / n).sqrt()
        min_L = torch.where(mask > 0, L, torch.full_like(L, float('inf'))).amin(dim=1, keepdim=True)
        min_R = torch.where(mask > 0, R, torch.full_like(R, float('inf'))).amin(dim=1, keepdim=True)
        fraction_L_at_min = (((L - min_L).abs() <= 1e-6) * mask).sum(dim=1, keepdim=True) / n
        fraction_R_at_min = (((R - min_R).abs() <= 1e-6) * mask).sum(dim=1, keepdim=True) / n
        mean_nfrac = (nfrac * mask).sum(dim=1, keepdim=True) / n
        log_n = torch.log(mask_sum.clamp_min(1.0))

        statistics = torch.cat([
            r_pearson, r_spearman, t_pearson,
            r_partial, r_partial_spearman, slope_raw, slope_rank, t_partial, t_partial_spearman,
            tercile_t, r_partial_low, r_partial_high, split_difference,
            log_n, mean_L, std_L, mean_R, std_R, fraction_L_at_min, fraction_R_at_min, mean_nfrac,
            r_spearman - r_partial_spearman, r_partial - r_pearson,
        ], dim=1)
        if self.use_skip_spearman:
            statistics = torch.cat([statistics, r_spearman], dim=1)
        statistics = torch.sign(statistics) * torch.log1p(statistics.abs())
        shrink = mask_sum / (mask_sum + self.shrink_n0)
        statistics = statistics * shrink

        if self.use_deepset:
            set_input = torch.stack([zL, zR, z_lib, nfrac, zL * zR, eL_raw * eR_raw], dim=-1)
            phi_output = self.set_phi(set_input)
            set_embedding = (phi_output * mask.unsqueeze(-1)).sum(dim=1) / n
            features = torch.cat([statistics, set_embedding], dim=1)
        else:
            features = statistics

        score = self.prediction_head(features)
        score = torch.clamp(score, -self.logit_clip, self.logit_clip)
        score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
        return score.float()


PLUGIN_MODEL_CLASS = LrContactContrastGbmHead
PLUGIN_OUTPUT_TYPE = "regressor"  # [B, 1] → continuous [B, 1] output

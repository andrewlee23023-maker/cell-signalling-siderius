import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from typing import Self


PLUGIN_MODEL_TYPE = "masked_contact_stats_deepsets_head"


class MaskedContactStatsDeepsetsHeadConfig(BaseModel):
    model_type: str = Field(default="masked_contact_stats_deepsets_head", description="Plugin model type key.")
    segmentation_size: int = Field(ge=1)
    batch_size: int = Field(default=32, ge=1)
    embed_dim: int = Field(default=16, ge=4, le=128)
    phi_hidden_dim: int = Field(default=16, ge=4, le=128)
    hidden_dim: int = Field(default=32, ge=4, le=256)
    n_stats: int = Field(default=6, ge=6, le=6)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    min_contacts: int = Field(default=8, ge=2, le=1024)
    ridge_lambda: float = Field(default=0.001, ge=1e-8, le=1.0)
    eps: float = Field(default=1e-6, ge=1e-12, le=1e-2)


PLUGIN_CONFIG_CLASS = MaskedContactStatsDeepsetsHeadConfig


class MaskedContactStatsDeepsetsHead(nn.Module):
    def __init__(self, config: "MaskedContactStatsDeepsetsHeadConfig"):
        super().__init__()
        super().__init__()
        self.embed_dim = config.embed_dim
        self.phi_hidden_dim = config.phi_hidden_dim
        self.hidden_dim = config.hidden_dim
        self.n_stats = config.n_stats
        self.dropout = config.dropout
        self.min_contacts = config.min_contacts
        self.ridge_lambda = config.ridge_lambda
        self.eps = config.eps
        self.rank_chunk_size = 128
        self.phi = nn.Sequential(
            nn.Linear(3, self.phi_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.phi_hidden_dim, self.embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(self.n_stats + 2 * self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward contract: input [B, 5, 1024] float32 → output [B, 1] float32
        x = x.float()
        batch_size = x.shape[0]
        n_cells = x.shape[2]
        ligand = x[:, 0, :]
        receptor = x[:, 1, :]
        nuisance_1 = x[:, 2, :]
        nuisance_2 = x[:, 3, :]
        count_key = x[:, 3, :]
        mask = (x[:, 4, :] > 0).to(x.dtype)
        counts = mask.sum(dim=1)
        valid = counts >= self.min_contacts

        def rank_values(values, local_mask):
            ranks = torch.zeros_like(values)
            for start in range(0, n_cells, self.rank_chunk_size):
                end = min(start + self.rank_chunk_size, n_cells)
                candidate = values[:, start:end]
                lower = (candidate.unsqueeze(2) > values.unsqueeze(1)) & (local_mask.unsqueeze(1) > 0)
                equal = (candidate.unsqueeze(2) == values.unsqueeze(1)) & (local_mask.unsqueeze(1) > 0)
                chunk_ranks = lower.to(values.dtype).sum(dim=2)
                chunk_ranks = chunk_ranks + 0.5 * (equal.to(values.dtype).sum(dim=2) + 1.0)
                chunk_ranks = chunk_ranks * local_mask[:, start:end]
                ranks[:, start:end] = chunk_ranks
            return ranks

        def masked_standardize(values, local_mask):
            local_count = local_mask.sum(dim=1).clamp_min(1.0)
            mean = (values * local_mask).sum(dim=1) / local_count
            centered = (values - mean.unsqueeze(1)) * local_mask
            variance = (centered.square() * local_mask).sum(dim=1) / local_count
            standardized = centered / torch.sqrt(variance.unsqueeze(1) + self.eps)
            return standardized * local_mask, mean, variance

        def masked_correlation(left, right, local_mask):
            left_std, _, left_var = masked_standardize(left, local_mask)
            right_std, _, right_var = masked_standardize(right, local_mask)
            count = local_mask.sum(dim=1).clamp_min(1.0)
            covariance = (left_std * right_std * local_mask).sum(dim=1) / count
            return covariance / torch.sqrt(left_var * right_var + self.eps)

        with torch.no_grad():
            rank_l = rank_values(ligand, mask)
            rank_r = rank_values(receptor, mask)
            r_l, _, _ = masked_standardize(rank_l, mask)
            r_r, _, _ = masked_standardize(rank_r, mask)

            s1 = masked_correlation(r_l, r_r, mask)
            _, ligand_mean, ligand_var = masked_standardize(ligand, mask)
            _, receptor_mean, receptor_var = masked_standardize(receptor, mask)
            count_safe = counts.clamp_min(1.0)
            ligand_centered = (ligand - ligand_mean.unsqueeze(1)) * mask
            receptor_centered = (receptor - receptor_mean.unsqueeze(1)) * mask
            raw_covariance = (ligand_centered * receptor_centered).sum(dim=1) / count_safe
            s2 = raw_covariance / torch.sqrt(ligand_var * receptor_var + self.eps)

            z1, _, _ = masked_standardize(nuisance_1, mask)
            z2, _, _ = masked_standardize(nuisance_2, mask)
            design = torch.stack([mask, z1, z2], dim=-1)
            weighted_design = design * mask.unsqueeze(-1)
            gram = torch.einsum('bni,bnj->bij', weighted_design, design)
            gram = gram + self.ridge_lambda * torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
            rhs_l = torch.einsum('bni,bn->bi', weighted_design, r_l)
            rhs_r = torch.einsum('bni,bn->bi', weighted_design, r_r)
            beta_l = torch.linalg.solve(gram, rhs_l.unsqueeze(-1)).squeeze(-1)
            beta_r = torch.linalg.solve(gram, rhs_r.unsqueeze(-1)).squeeze(-1)
            residual_l = r_l - torch.einsum('bni,bi->bn', design, beta_l)
            residual_r = r_r - torch.einsum('bni,bi->bn', design, beta_r)
            s3 = masked_correlation(residual_l, residual_r, mask)

            slope = raw_covariance / (ligand_var + self.eps)
            s4 = torch.asinh(slope)

            bottom_mask = mask * (rank_l <= counts.unsqueeze(1) / 3.0).to(x.dtype)
            top_mask = mask * (rank_l > 2.0 * counts.unsqueeze(1) / 3.0).to(x.dtype)
            bottom_count = bottom_mask.sum(dim=1)
            top_count = top_mask.sum(dim=1)
            bottom_mean = (receptor * bottom_mask).sum(dim=1) / bottom_count.clamp_min(1.0)
            top_mean = (receptor * top_mask).sum(dim=1) / top_count.clamp_min(1.0)
            receptor_std = torch.sqrt(receptor_var + self.eps)
            tercile_effect = (top_mean - bottom_mean) / (receptor_std + self.eps)
            s5 = torch.asinh(tercile_effect)
            s5 = torch.where((bottom_count > 0) & (top_count > 0), s5, torch.zeros_like(s5))

            sorted_key = count_key.masked_fill(mask <= 0, float('inf')).sort(dim=1).values
            median_index = (counts.long().clamp_min(1) - 1).clamp_min(0)
            median_key = sorted_key.gather(1, median_index.unsqueeze(1)).squeeze(1)
            high_mask = mask * (count_key > median_key.unsqueeze(1)).to(x.dtype)
            low_mask = mask * (count_key <= median_key.unsqueeze(1)).to(x.dtype)
            rank_high_l = rank_values(ligand, high_mask)
            rank_high_r = rank_values(receptor, high_mask)
            rank_low_l = rank_values(ligand, low_mask)
            rank_low_r = rank_values(receptor, low_mask)
            corr_high = masked_correlation(rank_high_l, rank_high_r, high_mask)
            corr_low = masked_correlation(rank_low_l, rank_low_r, low_mask)
            s6 = corr_high - corr_low
            half_min = float(self.min_contacts) / 2.0
            s6 = torch.where((high_mask.sum(dim=1) >= half_min) & (low_mask.sum(dim=1) >= half_min), s6, torch.zeros_like(s6))

            stats = torch.stack([s1, s2, s3, s4, s5, s6], dim=1)
            stats = stats * valid.to(x.dtype).unsqueeze(1)

        cell_features = torch.stack([r_l, r_r, r_l * r_r], dim=-1)
        embedded = self.phi(cell_features)
        embedded = embedded * mask.unsqueeze(-1)
        mean_embedding = embedded.sum(dim=1) / counts.clamp_min(1.0).unsqueeze(1)
        max_embedding = embedded.masked_fill(mask.unsqueeze(-1) <= 0, -1e4).max(dim=1).values
        max_embedding = torch.where((counts > 0).unsqueeze(1), max_embedding, torch.zeros_like(max_embedding))
        pooled = torch.cat([mean_embedding, max_embedding], dim=1)
        head_input = torch.cat([stats, pooled], dim=1)
        score = torch.sigmoid(self.head(head_input))
        return score.float()


PLUGIN_MODEL_CLASS = MaskedContactStatsDeepsetsHead
PLUGIN_OUTPUT_TYPE = "regressor"  # [B, 1] → continuous [B, 1] output

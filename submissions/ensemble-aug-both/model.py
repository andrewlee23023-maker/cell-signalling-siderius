import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from typing import Self


PLUGIN_MODEL_TYPE = "lr_coupling_mlp"


class LrCouplingMlpConfig(BaseModel):
    model_type: str = Field(default="lr_coupling_mlp", description="Plugin model type key.")
    segmentation_size: int = Field(ge=1)
    batch_size: int = Field(default=32, ge=1)
    hidden_dim: int = Field(default=32, ge=8, le=128)
    head_hidden_dim: int = Field(default=32, ge=8, le=128)
    n_bins: int = Field(default=4, ge=2, le=8)
    ridge_lambda: float = Field(default=0.001, ge=1e-6, le=0.1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    use_residualisation: bool = True
    use_skip_zscore: bool = True
    use_film: bool = True


PLUGIN_CONFIG_CLASS = LrCouplingMlpConfig


class LrCouplingMlp(nn.Module):
    def __init__(self, config: "LrCouplingMlpConfig"):
        super().__init__()
        H = config.hidden_dim
        Hh = config.head_hidden_dim
        self.hidden_dim = H
        self.n_bins = config.n_bins
        self.ridge_lambda = config.ridge_lambda
        self.use_residualisation = config.use_residualisation
        self.use_skip_zscore = config.use_skip_zscore
        self.use_film = config.use_film
        self.phi = nn.Sequential(
            nn.Linear(7, H),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(H, H),
            nn.GELU(),
        )
        self.film = nn.Sequential(
            nn.Linear(3, H),
            nn.GELU(),
            nn.Linear(H, 4 * H),
        )
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)
        self.psi = nn.Sequential(
            nn.Linear(2 * H + 12, Hh),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(Hh, 1),
        )
        if config.use_skip_zscore:
            nn.init.zeros_(self.psi[3].weight)
            nn.init.zeros_(self.psi[3].bias)
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward contract: input [B, 5, 1024] float32 → output [B, 1] float32
        x = torch.nan_to_num(x.float())
        eps = 1e-8
        K = self.n_bins

        def mmean(v, m, n):
            return (v * m).sum(1) / n

        def mstd(v, m, n):
            mu = mmean(v, m, n)
            var = (((v - mu[:, None]) ** 2) * m).sum(1) / n
            return torch.sqrt(var + eps)

        def pearson(a, b, m, n):
            ac = (a - mmean(a, m, n)[:, None]) * m
            bc = (b - mmean(b, m, n)[:, None]) * m
            cov = (ac * bc).sum(1)
            den = torch.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1) + 1e-12)
            return cov / den

        def rank_unit(v, m, n):
            vm = v.masked_fill(m == 0, float('inf')).contiguous()
            s, _ = torch.sort(vm, dim=1)
            lo = torch.searchsorted(s, vm, right=False)
            hi = torch.searchsorted(s, vm, right=True)
            rank = 0.5 * (lo + hi).float()
            return ((rank - 0.5) / n[:, None] - 0.5) * m

        def signlog(z):
            return torch.sign(z) * torch.log1p(torch.abs(z))

        def fisher(rho, n):
            r = torch.clamp(rho, -0.999, 0.999)
            return torch.atanh(r) * torch.sqrt(torch.clamp(n - 3.0, min=1.0))

        with torch.no_grad():
            L = x[:, 0]
            R = x[:, 1]
            c1 = x[:, 2]
            c2 = x[:, 3]
            m = (x[:, 4] > 0.5).float()
            n = m.sum(1).clamp(min=1.0)
            B, N = L.shape

            uL = rank_unit(L, m, n)
            uR = rank_unit(R, m, n)

            mc1 = mmean(c1, m, n)
            mc2 = mmean(c2, m, n)
            c1s = (c1 - mc1[:, None]) / mstd(c1, m, n)[:, None] * m
            c2s = (c2 - mc2[:, None]) / mstd(c2, m, n)[:, None] * m

            if self.use_residualisation:
                Zc = torch.stack([m, c1s, c2s], dim=-1) * m[..., None]
                Zt = Zc.transpose(1, 2)
                A = Zt @ Zc + self.ridge_lambda * torch.eye(3, device=x.device, dtype=x.dtype)[None]
                A = A + 1e-6 * torch.eye(3, device=x.device, dtype=x.dtype)[None]
                bL = Zt @ (uL * m)[..., None]
                bR = Zt @ (uR * m)[..., None]
                cL = torch.linalg.solve(A, bL)
                cR = torch.linalg.solve(A, bR)
                eL = (uL - (Zc @ cL).squeeze(-1)) * m
                eR = (uR - (Zc @ cR).squeeze(-1)) * m
            else:
                eL = (uL - mmean(uL, m, n)[:, None]) * m
                eR = (uR - mmean(uR, m, n)[:, None]) * m

            rho_raw = pearson(uL, uR, m, n)
            z_raw = fisher(rho_raw, n)
            rho_p = pearson(eL, eR, m, n)
            z_p = fisher(rho_p, n)
            slope = (m * eL * eR).sum(1) / ((m * eL * eL).sum(1) + eps)

            kb = torch.floor((uL + 0.5) * K).clamp(0, K - 1).long()
            onehot = F.one_hot(kb, K).float() * m[..., None]
            cnt = onehot.sum(1)
            sums = torch.einsum('bnk,bn->bk', onehot, eR)
            means = sums / cnt.clamp(min=1.0)
            ne = (cnt > 0).float()
            contrast = (means[:, -1] - means[:, 0]) * ne[:, -1] * ne[:, 0]
            idx = torch.arange(K, device=x.device, dtype=x.dtype)[None, :]
            sw = ne.sum(1).clamp(min=1.0)
            xm = (ne * idx).sum(1) / sw
            ym = (ne * means).sum(1) / sw
            tcov = (ne * (idx - xm[:, None]) * (means - ym[:, None])).sum(1)
            tvar = (ne * (idx - xm[:, None]) ** 2).sum(1)
            trend = tcov / (tvar + 1e-6)

            logn = torch.log(n)
            sdL_u = mstd(uL, m, n)
            sdR_u = mstd(uR, m, n)

            g = torch.stack([
                rho_raw,
                signlog(z_raw),
                rho_p,
                signlog(z_p),
                slope,
                contrast,
                trend,
                logn,
                mc1,
                mc2,
                sdL_u,
                sdR_u,
            ], dim=1)
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

            sdL = mstd(eL, m, n)[:, None]
            sdR = mstd(eR, m, n)[:, None]
            eLn = eL / sdL
            eRn = eR / sdR
            sdu = 0.2886751
            f = torch.stack([
                eLn,
                eRn,
                eLn * eRn,
                uL / sdu,
                uR / sdu,
                c1s,
                c2s,
            ], dim=-1) * m[..., None]
            f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            g3 = torch.stack([mc1, mc2, logn], dim=1)
            g3 = torch.nan_to_num(g3, nan=0.0, posinf=0.0, neginf=0.0)
            zp_skip = torch.nan_to_num(z_p, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.phi(f)
        mk = m[..., None]
        mean_pool = (h * mk).sum(1) / n[:, None]
        max_pool = h.masked_fill(mk == 0, float('-inf')).amax(1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        p = torch.cat([mean_pool, max_pool], dim=-1)

        if self.use_film:
            gb = self.film(g3)
            gamma, beta = gb.chunk(2, dim=-1)
            p = (1.0 + gamma) * p + beta

        s_head = self.psi(torch.cat([p, g], dim=-1))
        if self.use_skip_zscore:
            out = s_head + self.a * zp_skip[:, None]
        else:
            out = s_head
        out = torch.nan_to_num(out).float()
        return out


PLUGIN_OUTPUT_TYPE = "regressor"  # [B, 1] → continuous [B, 1] output


class _OriginalCellSignalModel(nn.Module):
    """Submission entry point wrapping the SIDERIUS-generated LrCouplingMlp (iter1, 2026-11-21).

    Built either as CellSignalModel(**cfg_json) or CellSignalModel(LrCouplingMlpConfig(...)) /
    CellSignalModel(config=...), so it loads under both plugin conventions. Input [B, 5, 1024] float32,
    output [B, 1] float32 (higher = more likely injected).
    """

    def __init__(self, config: "LrCouplingMlpConfig | None" = None, **kwargs):
        super().__init__()
        if config is None:
            kwargs.setdefault("segmentation_size", 1024)
            config = LrCouplingMlpConfig(**kwargs)
        elif isinstance(config, dict):
            config = LrCouplingMlpConfig(**config)
        self.net = LrCouplingMlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





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



  # [B, 1] → continuous [B, 1] output



# Component configs, embedded so the plugin needs no side files.
M1_CONFIG = {"model_type": "lr_coupling_mlp", "segmentation_size": 1024, "batch_size": 32, "hidden_dim": 48,
             "head_hidden_dim": 32, "n_bins": 6, "ridge_lambda": 0.001, "dropout": 0.2,
             "use_residualisation": True, "use_skip_zscore": False, "use_film": True}
M2_CONFIG = {"segmentation_size": 1024}  # lr_contact_contrast_gbm_head defaults (explore2 run)


class CellSignalModel(nn.Module):
    """Fixed-weight score ensemble: w1 * iter1-noskip-gen (m1) + w2 * explore2 contact-contrast head (m2).

    Each component is trained separately by train/train.py with its own recipe; the weights are fixed (cfg.json),
    not learned. Input [B, 5, 1024] float32, output [B, 1] float32.
    """

    def __init__(self, w1: float = 0.70, w2: float = 0.30, **kwargs):
        super().__init__()
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.m1 = _OriginalCellSignalModel(config=dict(M1_CONFIG))
        self.m2 = LrContactContrastGbmHead(LrContactContrastGbmHeadConfig(**M2_CONFIG))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1 * self.m1(x) + self.w2 * self.m2(x)


PLUGIN_MODEL_CLASS = CellSignalModel
PLUGIN_OUTPUT_TYPE = "regressor"

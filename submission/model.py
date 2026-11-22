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


class CellSignalModel(nn.Module):
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


PLUGIN_MODEL_CLASS = CellSignalModel

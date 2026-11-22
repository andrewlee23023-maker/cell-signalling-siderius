import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from typing import Self

import math

PLUGIN_MODEL_TYPE = "lr_pair_scorer"


class LrPairScorerConfig(BaseModel):
    model_type: str = Field(default="lr_pair_scorer", description="Plugin model type key.")
    segmentation_size: int = Field(ge=1)
    batch_size: int = Field(default=32, ge=1)
    hidden_dim: int = Field(default=32, ge=8, le=128)
    num_layers: int = Field(default=2, ge=1, le=4)
    ridge: float = Field(default=0.001, ge=1e-6, le=0.1)
    use_covariates: bool = True
    rank_chunk: int = Field(default=128, ge=16, le=1024)
    feature_clip: float = Field(default=8.0, ge=2.0, le=20.0)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    learn_skip: bool = True
    min_contacts: int = Field(default=4, ge=4, le=64)


PLUGIN_CONFIG_CLASS = LrPairScorerConfig


class LrPairScorer(nn.Module):
    def __init__(self, config: "LrPairScorerConfig"):
        super().__init__()
        self.ridge = float(config.ridge)
        self.use_covariates = bool(config.use_covariates)
        self.rank_chunk = int(config.rank_chunk)
        self.feature_clip = float(config.feature_clip)
        self.min_contacts = int(config.min_contacts)
        self.log_ref = math.log(256.0)
        self.num_features = 13
        layers = []
        in_dim = self.num_features
        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        final_layer = nn.Linear(in_dim, 1)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.mlp = nn.Sequential(*layers)
        if config.learn_skip:
            self.w_skip = nn.Parameter(torch.ones(()))
        else:
            self.register_buffer('w_skip', torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward contract: input [B, 5, 1024] float32 → output [B, 1] float32
        x = x.float()
        N = x.shape[-1]
        with torch.no_grad():
            m = (x[:, 4] > 0.5).float()
            zero = torch.zeros_like(x[:, 0])
            L = torch.where(m > 0, x[:, 0], zero)
            R = torch.where(m > 0, x[:, 1], zero)
            c2 = torch.where(m > 0, x[:, 2], zero)
            c3 = torch.where(m > 0, x[:, 3], zero)
            n = m.sum(1)
            n_c = n.clamp(min=4.0)
            n_col = n_c[:, None]

            def masked_rank(v):
                outs = []
                for s in range(0, N, self.rank_chunk):
                    e = min(s + self.rank_chunk, N)
                    vi = v[:, s:e, None]
                    vj = v[:, None, :]
                    cmp = (vj < vi).float() + 0.5 * (vj == vi).float()
                    outs.append((cmp * m[:, None, :]).sum(2) - 0.5)
                return torch.cat(outs, 1)

            def corr(a, b, w):
                ws = w.sum(1, keepdim=True).clamp(min=1e-6)
                ma = (a * w).sum(1, keepdim=True) / ws
                mb = (b * w).sum(1, keepdim=True) / ws
                da = (a - ma) * w
                db = (b - mb)
                cov = (da * db).sum(1)
                va = (da * (a - ma)).sum(1)
                vb = (w * db * db).sum(1)
                return cov / torch.sqrt(va * vb + 1e-12)

            def zstat(r):
                return torch.sqrt((n - 3.0).clamp(min=1.0)) * torch.atanh(r.clamp(-0.999, 0.999))

            rank_L = masked_rank(L)
            rank_R = masked_rank(R)
            centre = (n_col - 1.0) / 2.0
            rL = (rank_L - centre) / n_col * m
            rR = (rank_R - centre) / n_col * m

            r_plain = corr(rL, rR, m)
            z_plain = zstat(r_plain)

            c2m = (c2 * m).sum(1, keepdim=True) / n_col
            c3m = (c3 * m).sum(1, keepdim=True) / n_col
            c2d = (c2 - c2m) * m
            c3d = (c3 - c3m) * m
            c2s = torch.sqrt((c2d * c2d).sum(1, keepdim=True) / n_col).clamp(min=1e-4)
            c3s = torch.sqrt((c3d * c3d).sum(1, keepdim=True) / n_col).clamp(min=1e-4)
            c2c = c2d / c2s
            c3c = c3d / c3s

            Y = torch.stack([rL, rR], 2)
            if self.use_covariates:
                Zm = torch.stack([m, c2c * m, c3c * m], 1)
                eye = torch.eye(3, device=x.device, dtype=x.dtype)
                A = torch.bmm(Zm, Zm.transpose(1, 2)) + self.ridge * eye[None]
                bvec = torch.bmm(Zm, Y)
                beta = torch.linalg.solve(A, bvec)
                E = (Y - torch.bmm(Zm.transpose(1, 2), beta)) * m[:, :, None]
            else:
                E = Y
            r_part = corr(E[:, :, 0], E[:, :, 1], m)
            z_part = zstat(r_part)

            w3 = c3 * m
            r_w = corr(rL, rR, w3)

            q = (rank_R + 0.5) / n_col
            hi = (q > 2.0 / 3.0).float() * m
            lo = (q < 1.0 / 3.0).float() * m
            hi_n = hi.sum(1)
            lo_n = lo.sum(1)
            hi_mean = (rR * hi).sum(1) / hi_n.clamp(min=1.0)
            lo_mean = (rR * lo).sum(1) / lo_n.clamp(min=1.0)
            valid = ((hi_n > 0) & (lo_n > 0)).float()
            contrast = (hi_mean - lo_mean) * valid

            n1 = corr(rR, c2, m)
            n2 = corr(rR, c3, m)
            n3 = corr(rL, c3, m)
            n4 = corr(rL, c2, m)

            log_n = (torch.log(n_c) - self.log_ref) / 1.5
            c3_mean = (c3 * m).sum(1) / n_c
            c3_var = (((c3 - c3_mean[:, None]) * m) ** 2).sum(1) / n_c
            c3_std = torch.sqrt(c3_var + 1e-12)

            f = torch.stack([
                z_part / 3.0,
                3.0 * r_part,
                z_plain / 3.0,
                3.0 * r_plain,
                3.0 * r_w,
                4.0 * contrast,
                3.0 * n1,
                3.0 * n2,
                3.0 * n3,
                3.0 * n4,
                log_n,
                c3_mean,
                c3_std,
            ], dim=1)
            f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            f = f.clamp(-self.feature_clip, self.feature_clip)
            ok = (n >= float(self.min_contacts))[:, None]
            f = torch.where(ok, f, torch.zeros_like(f))
        s = self.w_skip * f[:, 0:1] + self.mlp(f)
        return s.float().reshape(-1, 1)


PLUGIN_MODEL_CLASS = LrPairScorer
PLUGIN_OUTPUT_TYPE = "regressor"  # [B, 1] → continuous [B, 1] output

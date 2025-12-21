import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class LowRankProjector(nn.Module):
    """Low-rank projector for efficient vision-language fusion.

    Decomposes the projection into two smaller matrices (W ≈ A * B),
    reducing parameters from O(d^2) to O(r * d), where r is the rank.
    """

    def __init__(self, config, vision_cfg=None):
        super().__init__()
        self._config = config
        self.rank = getattr(config, "mm_low_rank_rank", 64)
        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.hidden_size

        # Low-rank factorization: A (mm_hidden_size -> rank), B (rank -> hidden_size)
        self.down_project = nn.Linear(self.mm_hidden_size, self.rank, bias=False)
        self.norm = nn.LayerNorm(self.rank)
        self.act = nn.GELU()
        self.up_project = nn.Linear(self.rank, self.hidden_size, bias=False)

        # Initialize weights
        nn.init.kaiming_uniform_(self.down_project.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up_project.weight)

    def forward(self, x, *args, **kwargs):
        # x: [batch, tokens, mm_hidden_size]
        x = self.down_project(x)   # -> [batch, tokens, rank]
        x = self.norm(x)
        x = self.act(x)
        x = self.up_project(x)     # -> [batch, tokens, hidden_size]
        return x

    @property
    def config(self):
        return {"mm_projector_type": "low_rank", "mm_low_rank_rank": self.rank}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    if projector_type == "low_rank":
        return LowRankProjector(config, kwargs.get("vision_cfg"))

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from dacite import from_dict, Config as DaciteConfig


@dataclass(frozen=True)
class EnvConfig:
    obs_dim: int = 416
    n_tgt: int = 8
    tc_rate: float = 0.0005
    patch_len: int = 16


@dataclass(frozen=True)
class BufferConfig:
    buffer_dir: str = "data/trajectory_buffer/"
    n_episodes: int = 255
    n_futures: int = 16
    refresh_ratio: float = 0.2
    refresh_every_epochs: int = 10
    val_ratio: float = 0.2


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    log_dir: str = "tb_logs/strate_iv/"


@dataclass(frozen=True)
class StrateIVConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def load_config(path: str) -> StrateIVConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return from_dict(
        data_class=StrateIVConfig,
        data=config_dict,
        config=DaciteConfig(strict=True),
    )

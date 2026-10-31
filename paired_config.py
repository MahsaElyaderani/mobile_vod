from dataclasses import asdict, dataclass
import math


def probability(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1].")


@dataclass(frozen=True)
class PairedConfig:
    fast_width_mult: float = 0.35
    fast_image_size: int = 160
    min_depth: int = 8
    small_only_state_policy: str = "update"
    freeze_bn_stats: bool = True
    fast_bn_momentum: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.fast_width_mult) or not 0 < self.fast_width_mult <= 1:
            raise ValueError("fast_width_mult must be in (0,1].")
        if self.fast_image_size < 16 or self.fast_image_size % 16 or self.min_depth < 1:
            raise ValueError("Fast input must be a positive multiple of 16; min_depth must be positive.")
        if self.small_only_state_policy not in ("update", "hold"):
            raise ValueError("small_only_state_policy must be update or hold.")
        if self.fast_bn_momentum is not None:
            if not math.isfinite(self.fast_bn_momentum) or not 0 < self.fast_bn_momentum <= 1:
                raise ValueError("fast_bn_momentum must be in (0,1].")


@dataclass(frozen=True)
class MemoryCorruption:
    probability: float = 0.0
    dropout: float = 0.1
    noise_std: float = 0.1

    def __post_init__(self):
        probability(self.probability, "Corruption probability")
        probability(self.dropout, "Memory dropout")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("Memory noise_std must be finite and non-negative.")


@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "fixed"
    interval: int = 8
    scene_change_threshold: float | None = None
    reset_on_scene_change: bool = True
    memory_feedback: str = "clean"

    def __post_init__(self):
        if self.mode not in ("small", "big", "fixed") or self.interval < 1:
            raise ValueError("Invalid mode or interval.")
        if self.scene_change_threshold is not None:
            probability(self.scene_change_threshold, "Scene-change threshold")
        if self.memory_feedback not in ("clean", "none", "corrupted"):
            raise ValueError("memory_feedback must be clean, none or corrupted.")


def config_dict(config):
    return asdict(config)

"""Core interfaces for the CHACAM framework baseline."""

from .schema import (
    AmbiguityGroup,
    CCIConfig,
    Inputs,
    RunConfig,
    load_config,
    load_inputs,
    load_truth,
)
from .synthetic import SyntheticData, make_synthetic, write_synthetic

__all__ = [
    "AmbiguityGroup",
    "CCIConfig",
    "Inputs",
    "RunConfig",
    "SyntheticData",
    "load_config",
    "load_inputs",
    "load_truth",
    "make_synthetic",
    "write_synthetic",
]

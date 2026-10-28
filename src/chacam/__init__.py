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
from .assignment import AssignmentResult, capacities, greedy_assign
from .attention import AttentionResult, attention, l1_normalize, softmax_normalize
from .cci import TopKSelection, train_topk
from .evaluation import compute_metrics, evaluate
from .reference import ReferenceProfiles, binary_marker, training_mean
from .pipeline import FoldFit, RunResult, fit_fold, run

__all__ = [
    "AmbiguityGroup",
    "CCIConfig",
    "AssignmentResult",
    "AttentionResult",
    "FoldFit",
    "Inputs",
    "RunConfig",
    "SyntheticData",
    "ReferenceProfiles",
    "RunResult",
    "TopKSelection",
    "attention",
    "binary_marker",
    "capacities",
    "compute_metrics",
    "evaluate",
    "fit_fold",
    "greedy_assign",
    "l1_normalize",
    "load_config",
    "load_inputs",
    "load_truth",
    "make_synthetic",
    "softmax_normalize",
    "train_topk",
    "training_mean",
    "run",
    "write_synthetic",
]

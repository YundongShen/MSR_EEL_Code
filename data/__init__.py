"""Data pipeline for Edit Entailment Learning."""

from .data_loader import DataLoader, DataSample, GitHubDataLoader
from .entailment_dataset import EntailmentDataset, EntailmentPair
from .tier_labeler import tier_of_file
from .utils import hash_signature, normalize_diff, tokenize_diff_hunks

__all__ = [
    # SWE-bench loading (kept for LLM pair generation pipeline)
    "DataLoader",
    "DataSample",
    "GitHubDataLoader",
    # New: Edit Entailment
    "EntailmentDataset",
    "EntailmentPair",
    "tier_of_file",
    # Utilities
    "hash_signature",
    "normalize_diff",
    "tokenize_diff_hunks",
]

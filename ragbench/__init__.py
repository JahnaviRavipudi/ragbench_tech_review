"""
RAGBench — Explainable Benchmark for Retrieval-Augmented Generation Systems

Vendored from: https://github.com/rungalileo/ragbench
Paper: Friel et al. (2024), arXiv:2407.11005
"""

from .constants import (
    HUGGINGFACE_REPO_NAME,
    RAGBenchFields,
    TrulensFields,
    RagasFields,
    DEFAULT_OPENAI_MAX_CONCURRENT,
)
from .evaluation import rmse, auroc, calculate_metrics

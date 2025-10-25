"""Core anomaly detection functions for SpectraAI.

Include stubs for computing anomaly scores and simple thresholding.
"""
from typing import List


def compute_score(embedding: List[float]) -> float:
    """Compute a simple anomaly score from an embedding.

    For now, this is a placeholder using L2 norm.
    """
    return sum(x * x for x in embedding) ** 0.5


def is_anomaly(score: float, threshold: float = 1.0) -> bool:
    """Return True if score exceeds threshold."""
    return score > threshold

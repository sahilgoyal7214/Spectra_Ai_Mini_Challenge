"""Synthetic data & embedding generation for SpectraAI anomaly detection.

Contains a simple function stub to generate synthetic prompts and mock embeddings.
"""

from typing import List, Tuple


def generate_synthetic_prompts(n: int = 100) -> List[Tuple[str, List[float]]]:
    """Return n synthetic prompts and fake embeddings.

    Args:
        n: number of samples

    Returns:
        List of (prompt, embedding) tuples where embedding is a small list of floats.
    """
    samples = []
    for i in range(n):
        prompt = f"synthetic prompt {i}"
        embedding = [float((i + j) % 10) / 10.0 for j in range(8)]
        samples.append((prompt, embedding))
    return samples

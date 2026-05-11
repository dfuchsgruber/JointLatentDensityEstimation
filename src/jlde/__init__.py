"""Public API for joint latent density estimation.

The package is intentionally independent of the GNN code in `ufnhg`. It expects
callers to provide logits, labels, train masks, and latent hidden states that
were already produced elsewhere.
"""

from jlde.densities import LatentDensity, NearestNeighborDensity
from jlde.estimators import JointLatentDensityEstimator
from jlde.preprocessing import JointLatentPreprocessor, PCA

__all__ = [
    "JointLatentDensityEstimator",
    "JointLatentPreprocessor",
    "LatentDensity",
    "NearestNeighborDensity",
    "PCA",
]

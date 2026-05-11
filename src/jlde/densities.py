"""Latent density models used by the joint latent estimator.

Density classes operate on already preprocessed latent values. They do not run
the GNN, select hidden-state layers, or fit PCA; those responsibilities belong
to `JointLatentPreprocessor`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDensity(nn.Module):
    """Abstract interface for density scorers over latent tensors."""

    def fit(self, values: Sequence[torch.Tensor], *args, **kwargs) -> LatentDensity:
        """Fit density state from preprocessed training values."""
        raise NotImplementedError

    def predict(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return one scalar confidence/density score per sample."""
        raise NotImplementedError


class NearestNeighborDensity(LatentDensity):
    """KNN confidence score used by the legacy `LatentNearestNeighbor`.

    Args:
        k_neighbors: Number of nearest training samples to average.
        use_energy: If true, divide neighbor distances by the training
            `softplus(logsumexp(logits))` values stored at fit time.
    """

    def __init__(self, k_neighbors: int = 5, use_energy: bool = False):
        """Create an unfitted nearest-neighbor density scorer."""
        super().__init__()
        self.k_neighbors = k_neighbors
        self.use_energy = use_energy
        self.train_prediction_logsumexp = self.register_buffer(
            "train_prediction_logsumexp", None
        )
        self.train_labels = self.register_buffer("train_labels", None)
        self._num_train_values = 0

    def fit(
        self,
        values: Sequence[torch.Tensor],
        labels: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> NearestNeighborDensity:
        """Store training values, labels, and optional energy-normalizer state.

        Args:
            values: Preprocessed training latent tensors. In joint mode this is
                a one-element sequence containing concatenated embeddings; in
                independent mode it contains one tensor per selected state.
            labels: Training labels. KNN does not use them directly, but they
                are preserved for legacy compatibility and future densities.
            logits: Training logits. Required when `use_energy=True`.
        """
        self.train_values = list(values)
        self.train_labels = labels
        if logits is not None:
            self.train_prediction_logsumexp = torch.logsumexp(logits, dim=1)
        elif self.use_energy:
            raise ValueError("logits are required when use_energy=True")
        else:
            self.train_prediction_logsumexp = None
        return self

    def predict(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        """Score samples as negative squared mean top-k distance."""
        values = list(values)
        train_values = self.train_values
        if not train_values:
            raise RuntimeError("NearestNeighborDensity must be fitted before predict")

        confidence = torch.zeros(values[0].shape[0], device=values[0].device)
        for value, train_value in zip(values, train_values):
            distances = torch.cdist(value, train_value)
            topk = distances.topk(self.k_neighbors, largest=False)
            distances = topk.values
            if self.use_energy:
                distances = distances / F.softplus(
                    self.train_prediction_logsumexp[topk.indices]
                )
            confidence -= distances.mean(dim=-1) ** 2
        return confidence

    @property
    def train_values(self) -> list[torch.Tensor | None]:
        """Fitted training latent values, one tensor per density block."""
        return [
            getattr(self, f"layer_{i}_train_values")
            for i in range(self._num_train_values)
        ]

    @train_values.setter
    def train_values(self, values: Sequence[torch.Tensor | None]) -> None:
        """Store training latent values as dynamic buffers."""
        self._num_train_values = len(values)
        for i, value in enumerate(values):
            self._set_buffer(f"layer_{i}_train_values", value)

    def _set_buffer(self, name: str, value: torch.Tensor | None) -> None:
        """Register or replace a tensor buffer used as fitted density state."""
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

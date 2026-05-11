"""Scikit-learn-style estimator for joint latent density estimation.

The estimator consumes GNN outputs that were produced elsewhere: hidden states,
labels, masks, and optionally logits. It owns only post-hoc density estimation
state, making it usable outside the original `ufnhg` model wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from jlde.densities import LatentDensity, NearestNeighborDensity
from jlde.preprocessing import HiddenStates, JointLatentPreprocessor


class JointLatentDensityEstimator(nn.Module):
    """Fit and apply a density estimator on joint latent embeddings.

    Args:
        density: Density backend name or already constructed `LatentDensity`.
            Currently `"knn"`, `"nearest_neighbor"`, and `"nearest_neighbors"`
            create `NearestNeighborDensity`.
        n_components: PCA components or explained-variance threshold. `None`
            disables PCA.
        k_neighbors: Number of neighbors for the KNN density backend.
        layers: Hidden-state layers to use before density fitting.
        scale_type: Optional scaling mode passed to `JointLatentPreprocessor`.
        use_energy: Whether KNN distances are normalized by fit-time logits.
        independent_states: If false, selected hidden states are concatenated
            before scoring; if true, they are scored independently and summed.
    """

    def __init__(
        self,
        *,
        density: str | LatentDensity = "knn",
        n_components: int | float | None = 0.95,
        k_neighbors: int = 5,
        layers: str | Sequence[int] = "all",
        scale_type: str | None = None,
        use_energy: bool = False,
        independent_states: bool = False,
    ):
        """Create an unfitted estimator from explicit hyperparameters."""
        super().__init__()
        self.preprocessor = JointLatentPreprocessor(
            n_components=n_components,
            layers=layers,
            scale_type=scale_type,
            independent_states=independent_states,
        )
        self.density = self._make_density(
            density,
            k_neighbors=k_neighbors,
            use_energy=use_energy,
        )

    @classmethod
    def from_config(cls, config: dict) -> JointLatentDensityEstimator:
        """Build an estimator from the legacy latent post-fit config dict."""
        return cls(
            n_components=config.get("n_components", 0.95),
            k_neighbors=config.get("k_neighbors", 5),
            layers=config.get("layers", "all"),
            scale_type=config.get("scale_type", None),
            use_energy=config.get("use_energy", False),
            independent_states=config.get("independent_states", False),
        )

    def initialize(self, num_hidden_states: int) -> JointLatentDensityEstimator:
        """Resolve layer-dependent state before hidden tensors are available."""
        self.preprocessor.initialize(num_hidden_states)
        return self

    def fit(
        self,
        hidden_states: HiddenStates,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> JointLatentDensityEstimator:
        """Fit preprocessing and density state from GNN outputs.

        Args:
            hidden_states: Hidden states for all samples/nodes.
            labels: Labels for all samples/nodes.
            train_mask: Boolean mask selecting training samples/nodes.
            logits: Logits for all samples/nodes. Required when the density uses
                energy normalization.
        """
        self.preprocessor.fit(hidden_states)
        values = self.preprocessor.transform(hidden_states)
        train_values = [value[train_mask] for value in values]
        train_logits = logits[train_mask] if logits is not None else None
        self.density.fit(
            values=train_values,
            labels=labels[train_mask],
            logits=train_logits,
        )
        return self

    def predict(self, hidden_states: HiddenStates) -> torch.Tensor:
        """Return confidence scores for all provided hidden states."""
        values = self.preprocessor.transform(hidden_states)
        return self.density.predict(values)

    def predict_confidence(self, hidden_states: HiddenStates) -> torch.Tensor:
        """Explicit alias for `predict` used by uncertainty-scoring callers."""
        return self.predict(hidden_states=hidden_states)

    def score_samples(self, hidden_states: HiddenStates) -> torch.Tensor:
        """Alias for scikit-learn-style sample scoring."""
        return self.predict(hidden_states=hidden_states)

    @property
    def layers(self) -> list[int] | None:
        """Resolved hidden-state layer indices, or `None` before initialization."""
        return self.preprocessor.layers

    @property
    def pcas(self) -> nn.ModuleList:
        """PCA modules owned by the preprocessor."""
        return self.preprocessor.pcas

    @property
    def mean_values(self) -> list[torch.Tensor | None]:
        """Scaling means exposed for legacy adapter compatibility."""
        return self.preprocessor.mean_values

    @mean_values.setter
    def mean_values(self, values: Sequence[torch.Tensor | None]) -> None:
        """Replace scaling means on the underlying preprocessor."""
        self.preprocessor.mean_values = values

    @property
    def scale_values(self) -> list[torch.Tensor | None]:
        """Scaling factors exposed for legacy adapter compatibility."""
        return self.preprocessor.scale_values

    @scale_values.setter
    def scale_values(self, values: Sequence[torch.Tensor | None]) -> None:
        """Replace scaling factors on the underlying preprocessor."""
        self.preprocessor.scale_values = values

    @property
    def train_values(self) -> list[torch.Tensor | None]:
        """Training latent values stored by the density backend."""
        return self.density.train_values

    @train_values.setter
    def train_values(self, values: Sequence[torch.Tensor | None]) -> None:
        """Replace training latent values on the density backend."""
        self.density.train_values = values

    @property
    def train_labels(self) -> torch.Tensor | None:
        """Training labels stored by the density backend."""
        return self.density.train_labels

    @train_labels.setter
    def train_labels(self, value: torch.Tensor | None) -> None:
        """Replace training labels on the density backend."""
        self.density.train_labels = value

    @property
    def train_prediction_logsumexp(self) -> torch.Tensor | None:
        """Fit-time `logsumexp(logits)` values used by energy-normalized KNN."""
        return self.density.train_prediction_logsumexp

    @train_prediction_logsumexp.setter
    def train_prediction_logsumexp(self, value: torch.Tensor | None) -> None:
        """Replace fit-time energy-normalizer values on the density backend."""
        self.density.train_prediction_logsumexp = value

    @staticmethod
    def _make_density(
        density: str | LatentDensity,
        *,
        k_neighbors: int,
        use_energy: bool,
    ) -> LatentDensity:
        """Create a density backend from a name or pass through an instance."""
        if isinstance(density, LatentDensity):
            return density
        match density:
            case "knn" | "nearest_neighbor" | "nearest_neighbors":
                return NearestNeighborDensity(
                    k_neighbors=k_neighbors,
                    use_energy=use_energy,
                )
            case _:
                raise ValueError(f"Unknown joint latent density: {density}")

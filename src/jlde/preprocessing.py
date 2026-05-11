"""Preprocessing for joint latent density estimators.

This module contains the data preparation that used to live inside
`LatentNearestNeighbor`: hidden-state layer selection, optional PCA, optional
feature scaling, and optional concatenation into a joint latent embedding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


HiddenStates = Sequence[torch.Tensor] | torch.Tensor


def as_hidden_state_list(hidden_states: HiddenStates) -> list[torch.Tensor]:
    """Normalize a single tensor or tensor sequence to a mutable list."""
    if isinstance(hidden_states, torch.Tensor):
        return [hidden_states]
    return list(hidden_states)


class PCA(nn.Module):
    """Small torch PCA module matching the legacy project implementation.

    Args:
        n_components: Number of retained components. A float is interpreted as
            a cumulative explained-variance threshold. `None` disables PCA and
            makes `transform` the identity.
    """

    def __init__(self, n_components: int | float | None):
        """Create an unfitted PCA transform."""
        super().__init__()
        self.n_components = n_components
        self._n_components = None
        self.singular_values = self.register_buffer("singular_values", None)
        self.mean = self.register_buffer("mean", None)
        self.V = self.register_buffer("V", None)

    def fit(self, data: torch.Tensor) -> PCA:
        """Fit principal components on all provided samples.

        This intentionally fits on the complete hidden-state matrix, not only
        the train mask, because that is the behavior of the legacy post-fit KNN.
        """
        if self.n_components is None:
            return self
        with torch.no_grad():
            self.mean = data.mean(dim=0)
            data = data - self.mean
            if isinstance(self.n_components, float):
                _, S, V = torch.pca_lowrank(data, q=data.size(1), niter=2)
                variances = S**2
                n_components = min(
                    (
                        (variances.cumsum(dim=0) / variances.sum())
                        < self.n_components
                    ).sum().item()
                    + 1,
                    data.size(1),
                )
                self.singular_values = S
                self.V = V[:, :n_components]
                self._n_components = n_components
            else:
                _, S, V = torch.pca_lowrank(data, q=self.n_components, niter=2)
                self.singular_values = S
                self.V = V
                self._n_components = self.n_components
        return self

    @property
    def variances(self) -> torch.Tensor:
        """Return unnormalized principal-component variances."""
        return self.singular_values**2

    @property
    def variance_ratio(self) -> torch.Tensor:
        """Return explained variance ratios for the retained components."""
        variances = self.variances
        return variances[: self._n_components] / variances.sum()

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Project data into the fitted PCA basis, or return data unchanged."""
        if self.n_components is None:
            return data
        return torch.matmul(data - self.mean, self.V)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Alias `transform` so the module composes like a torch layer."""
        return self.transform(data)


class JointLatentPreprocessor(nn.Module):
    """Select, reduce, scale, and optionally concatenate hidden states.

    Args:
        n_components: PCA components or explained-variance threshold. `None`
            disables PCA.
        layers: Hidden-state layers to use. `"all"` selects every layer; integer
            entries are normalized modulo the number of hidden states so legacy
            negative and wraparound indices keep working.
        scale_type: Optional scaling mode. Supported values are `None`,
            `"same_expected_distance"`, and `"isotropic"`.
        independent_states: When false, selected states are concatenated into
            one joint latent embedding. When true, each state remains separate
            for independent density scoring.
    """

    def __init__(
        self,
        n_components: int | float | None = 0.95,
        layers: str | Sequence[int] = "all",
        scale_type: str | None = None,
        independent_states: bool = False,
    ):
        """Create an unfitted preprocessor."""
        super().__init__()
        self.n_components = n_components
        self.requested_layers = layers
        self.scale_type = scale_type
        self.independent_states = independent_states
        self.layers: list[int] | None = None
        self.pcas = nn.ModuleList()
        self._num_selected_layers = 0

    def initialize(self, num_hidden_states: int) -> JointLatentPreprocessor:
        """Resolve layers and allocate per-layer transform/scaling buffers."""
        layers = self._resolve_layers(num_hidden_states)
        self.layers = layers
        self._num_selected_layers = len(layers)
        self.pcas = nn.ModuleList(PCA(self.n_components) for _ in layers)
        for i in range(len(layers)):
            self._set_buffer(f"layer_{i}_mean", None)
            self._set_buffer(f"layer_{i}_scale", None)
        return self

    def fit(self, hidden_states: HiddenStates) -> JointLatentPreprocessor:
        """Fit PCA and scaling statistics on the provided hidden states."""
        hidden_states = as_hidden_state_list(hidden_states)
        self.initialize(len(hidden_states))
        selected = self.select(hidden_states)

        if self.n_components is not None:
            for pca, hidden_state in zip(self.pcas, selected):
                pca.fit(hidden_state)

        if self.scale_type is not None:
            self._set_scale(selected)

        return self

    def select(self, hidden_states: HiddenStates) -> list[torch.Tensor]:
        """Return hidden states in the resolved layer order."""
        hidden_states = as_hidden_state_list(hidden_states)
        if self.layers is None:
            self.initialize(len(hidden_states))
        return [hidden_states[i] for i in self.layers]

    def transform(self, hidden_states: HiddenStates) -> list[torch.Tensor]:
        """Apply layer selection, PCA/scaling, and optional concatenation."""
        values = self.reduce(self.select(hidden_states))
        if not self.independent_states:
            values = [torch.cat(values, dim=-1)]
        return values

    def reduce(self, values: HiddenStates) -> list[torch.Tensor]:
        """Apply fitted PCA and scaling to already selected hidden states."""
        values = as_hidden_state_list(values)
        if self.pcas is not None:
            values = [pca.transform(value) for pca, value in zip(self.pcas, values)]
        return [self.scale(value, i) for i, value in enumerate(values)]

    def scale(self, value: torch.Tensor, i: int) -> torch.Tensor:
        """Apply fitted centering and scaling for one selected state."""
        if self.mean_values and self.mean_values[0] is not None:
            value = value - self.mean_values[i]
        if self.scale_values and self.scale_values[0] is not None:
            value = value * torch.as_tensor(self.scale_values[i])
        return value

    @property
    def mean_values(self) -> list[torch.Tensor | None]:
        """Mean vectors used for optional scaling, one per selected state."""
        return [
            getattr(self, f"layer_{i}_mean")
            for i in range(self._num_selected_layers)
        ]

    @mean_values.setter
    def mean_values(self, values: Sequence[torch.Tensor | None]) -> None:
        """Store scaling means as registered buffers."""
        for i, value in enumerate(values):
            self._set_buffer(f"layer_{i}_mean", value)

    @property
    def scale_values(self) -> list[torch.Tensor | None]:
        """Scale factors used after centering, one per selected state."""
        return [
            getattr(self, f"layer_{i}_scale")
            for i in range(self._num_selected_layers)
        ]

    @scale_values.setter
    def scale_values(self, values: Sequence[torch.Tensor | None]) -> None:
        """Store scale factors as registered buffers."""
        for i, value in enumerate(values):
            self._set_buffer(f"layer_{i}_scale", value)

    def _set_scale(self, hidden_states: Sequence[torch.Tensor]) -> None:
        """Fit centering and scale factors after optional PCA projection."""
        if self.pcas is not None:
            hidden_states = [
                pca.transform(values)
                for pca, values in zip(self.pcas, hidden_states)
            ]
        self.mean_values = [hidden.mean(dim=0) for hidden in hidden_states]
        match self.scale_type:
            case "same_expected_distance":
                self.scale_values = [
                    1 / (hidden - mean).norm(dim=1).mean(dim=-1)
                    for hidden, mean in zip(hidden_states, self.mean_values)
                ]
            case "isotropic":
                self.scale_values = [
                    1 / hidden.std(dim=0) for hidden in hidden_states
                ]
            case _:
                raise ValueError(f"Unkown scale type: {self.scale_type}")

    def _resolve_layers(self, num_hidden_states: int) -> list[int]:
        """Convert the requested layer spec to normalized layer indices."""
        if self.requested_layers == "all":
            layers = list(range(num_hidden_states))
        elif isinstance(self.requested_layers, int):
            layers = [self.requested_layers]
        else:
            layers = list(self.requested_layers)
        return [
            (int(layer) + num_hidden_states) % num_hidden_states
            for layer in layers
        ]

    def _set_buffer(self, name: str, value: torch.Tensor | None) -> None:
        """Register or replace a buffer used for fitted preprocessing state."""
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

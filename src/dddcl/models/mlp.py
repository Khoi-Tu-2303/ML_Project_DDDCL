"""PyTorch Multi-Layer Perceptron (MLP) Classifier for DDDCL experiments.

Follows specifications from docs/03_METHOD_SPEC.md and docs/06_ARCHITECTURE.md:
- Pure PyTorch CPU deterministic implementation.
- Binary classification: single logit output with BCEWithLogitsLoss.
- Multiclass classification: C logits output with CrossEntropyLoss.
- Dropout only active during model.train(), disabled during model.eval().
- Explicit initialization and seed control.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_module(name: str) -> nn.Module:
    """Resolve activation function by name."""
    lower = name.lower()
    if lower == "relu":
        return nn.ReLU()
    elif lower == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif lower == "tanh":
        return nn.Tanh()
    elif lower == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: '{name}'. Supported: relu, leaky_relu, tanh, gelu.")


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for tabular classification.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of target classes (>= 2).
    hidden_dims : Sequence[int], default=(64, 32)
        Dimensions of hidden layers.
    dropout_rate : float, default=0.0
        Dropout probability between hidden layers (only applied during training).
    activation : str, default='relu'
        Activation function between hidden layers.
    seed : int | None, default=None
        Seed for reproducible weight initialization.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout_rate: float = 0.0,
        activation: str = "relu",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.is_binary = (num_classes == 2)
        self.output_dim = 1 if self.is_binary else num_classes

        # Build layer stack
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(_get_activation_module(activation))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        # Final classification head
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights with optional seed
        self.reset_parameters(seed=seed)

    def reset_parameters(self, seed: int | None = None) -> None:
        """Initialize network weights deterministically."""
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming normal initialization for hidden layers, Xavier for output
                is_output = (module.out_features == self.output_dim)
                if is_output:
                    # Xavier uniform for final head
                    bound = math.sqrt(6.0 / (module.in_features + module.out_features))
                    if gen is not None:
                        module.weight.data.uniform_(-bound, bound, generator=gen)
                    else:
                        module.weight.data.uniform_(-bound, bound)
                else:
                    # Kaiming normal
                    std = math.sqrt(2.0 / module.in_features)
                    if gen is not None:
                        module.weight.data.normal_(0.0, std, generator=gen)
                    else:
                        module.weight.data.normal_(0.0, std)
                # Bias initialized to zero
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw logits.
        
        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Raw logits tensor:
            - shape (batch_size, 1) if binary classification.
            - shape (batch_size, num_classes) if multiclass classification.
        """
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}"
            )
        return self.network(x)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute task-appropriate classification loss from raw logits.
        
        Parameters
        ----------
        logits : torch.Tensor
            Logits output from forward(), shape (B, 1) or (B, C).
        targets : torch.Tensor
            Target labels, shape (B,).
        reduction : str, default='mean'
            Reduction applied to the loss: 'none', 'mean', or 'sum'.

        Returns
        -------
        torch.Tensor
            Scalar loss if reduction is 'mean' or 'sum', 1D tensor of shape (B,) if 'none'.
        """
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction '{reduction}'. Must be 'none', 'mean', or 'sum'.")

        if self.is_binary:
            # Binary classification: targets must be float shaped (B, 1)
            targets_bce = targets.float().view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(logits, targets_bce, reduction=reduction)
            if reduction == "none":
                return loss.view(-1)
            return loss
        else:
            # Multiclass classification: targets must be long shaped (B,)
            targets_ce = targets.long().view(-1)
            return F.cross_entropy(logits, targets_ce, reduction=reduction)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Compute predicted class probabilities.
        
        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            Features of shape (N, input_dim).

        Returns
        -------
        np.ndarray
            Probability array of shape (N, num_classes).
        """
        was_training = self.training
        self.eval()
        try:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(np.asarray(x, dtype=np.float32))
            logits = self.forward(x)
            if self.is_binary:
                p1 = torch.sigmoid(logits).squeeze(-1).numpy()
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])
            else:
                probs = torch.softmax(logits, dim=-1).numpy()
                return probs
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Compute predicted discrete class labels.
        
        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            Features of shape (N, input_dim).

        Returns
        -------
        np.ndarray
            1D array of predicted class integers of shape (N,).
        """
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)


def create_mlp_model(
    input_dim: int,
    num_classes: int,
    hidden_dims: Sequence[int] = (64, 32),
    dropout_rate: float = 0.0,
    activation: str = "relu",
    seed: int | None = None,
) -> MLPClassifier:
    """Factory function for creating an MLPClassifier."""
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation=activation,
        seed=seed,
    )


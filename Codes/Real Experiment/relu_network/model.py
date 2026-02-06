"""
model.py
Neural network model definition
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ReLURegNet(nn.Module):
    """Single-hidden-layer ReLU network for regression"""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """
        Initialize the ReLU network

        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
        """
        super().__init__()
        # hidden layer weights w_k and biases b_k
        self.hidden = nn.Linear(input_dim, hidden_dim, bias=True)
        # output weights v_k (no bias)
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network

        Args:
            x: Input tensor [N, input_dim]

        Returns:
            out: Output predictions [N, 1]
            z: Pre-activation values [N, hidden_dim]
        """
        z = self.hidden(x)        # pre-activations [N, K]
        h = self.relu(z)          # activations [N, K]
        out = self.output(h)      # output [N, 1]
        return out, z

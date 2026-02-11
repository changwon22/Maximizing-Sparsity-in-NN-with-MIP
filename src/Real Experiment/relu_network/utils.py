"""
utils.py - Utility functions for analyzing sparse ReLU networks

This module provides functions for computing metrics on trained ReLURegNet models:

1. Path-based metrics: Operate on W * V (hidden weights × output weights),
   capturing the effective contribution of each hidden unit to output.

2. Sparsity metrics: Count parameters below threshold (effectively zero).

3. Neuron metrics: Count active hidden neurons contributing to output.

All functions support the 'incl_bias_sparsity' parameter to optionally
include bias terms in calculations.

Functions:
    lp_path_norm: Compute Lp norm of path-based parameterization
    count_nonzero: Count nonzero parameters (L0 path norm)
    count_active_neurons: Count neurons with nonzero outgoing paths
    get_max_params: Get max absolute parameter values for MILP bounds
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .model import ReLURegNet


def _get_param_product(model: ReLURegNet, incl_bias_sparsity: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract weight and output matrices for path-based calculations.

    This is an internal helper function used by lp_path_norm, count_nonzero,
    and count_active_neurons to avoid code duplication.

    Args:
        model: ReLURegNet model instance
        incl_bias_sparsity: If True, concatenate bias as extra column to weights

    Returns:
        W: Weight matrix, optionally including biases (K x d or K x (d+1))
        V: Output weight vector reshaped to (K x 1)
    """
    if incl_bias_sparsity:
        W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
    else:
        W = model.hidden.weight
    V = model.output.weight.view(-1, 1)
    return W, V


def lp_path_norm(model: ReLURegNet, p: float, incl_bias_sparsity: bool = True) -> float:
    """
    Calculate lp path norm of the network

    Args:
        model: ReLURegNet model
        p: Power for lp norm
        incl_bias_sparsity: Whether to include bias terms

    Returns:
        Lp path norm value
    """
    W, V = _get_param_product(model, incl_bias_sparsity)
    return (W * V).abs().pow(p).sum().item()


def count_nonzero(model: ReLURegNet, eps: float = 1e-6, incl_bias_sparsity: bool = True) -> int:
    """
    Count nonzero parameters (l0 path norm)

    Args:
        model: ReLURegNet model
        eps: Threshold for considering a parameter as nonzero
        incl_bias_sparsity: Whether to include bias terms

    Returns:
        Number of nonzero parameters
    """
    W, V = _get_param_product(model, incl_bias_sparsity)
    return int((W * V).abs().gt(eps).sum().item())


def count_active_neurons(model: ReLURegNet, eps: float = 1e-6, incl_bias_sparsity: bool = True) -> int:
    """
    Count active neurons in the hidden layer

    Args:
        model: ReLURegNet model
        eps: Threshold for considering a neuron as active
        incl_bias_sparsity: Whether to include bias terms

    Returns:
        Number of active neurons
    """
    W, V = _get_param_product(model, incl_bias_sparsity)
    return int(((W * V).abs().ge(eps).any(dim=1)).sum().item())


def get_max_params(model: ReLURegNet) -> tuple[float, float, float]:
    """
    Get maximum absolute values of model parameters
    
    Args:
        model: ReLURegNet model
        
    Returns:
        max_W: Maximum absolute weight value
        max_b: Maximum absolute bias value
        max_v: Maximum absolute output weight value
    """
    with torch.no_grad():
        max_W = float(torch.max(torch.abs(model.hidden.weight)).item())
        max_b = float(torch.max(torch.abs(model.hidden.bias)).item())
        max_v = float(torch.max(torch.abs(model.output.weight)).item())
    
    return max_W, max_b, max_v

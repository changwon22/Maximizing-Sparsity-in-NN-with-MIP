"""
data_generation.py
Functions for generating training data
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import math


def relu(a: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Apply ReLU activation function"""
    return np.maximum(a, 0.0)


def data(N: int, d: int) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32],
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]:
    """
    Generate three different types of datasets

    Args:
        N: Number of samples
        d: Dimension of features

    Returns:
        X1, X2, X3: Three different feature matrices (N x d)
        y1, y2, y3: Three different target vectors (N,)
    """
    np.random.seed(0)
    
    # Case 1: Random - Uniform (-1,1)
    X1 = np.random.uniform(-1, 1, size=(N, d)).astype(np.float32)
    y1 = np.random.uniform(-1, 1, size=(N, 1)).astype(np.float32)

    # Case 2: Functional relationship - Ground-Truth ReLU network
    X2 = np.random.uniform(-1, 1, size=(N, d)).astype(np.float32)
    K_true = math.ceil(N / 2)
    W_true = np.random.randn(d, K_true) * 0.5
    b_true = np.random.randn(K_true) * 0.1
    v_true = np.random.randn(K_true) * 0.5
    hidden_layer = relu(X2 @ W_true + b_true)
    y = hidden_layer @ v_true
    noise = 0.05 * np.random.randn(N)
    y2 = y + noise
    y2 = y2.astype(np.float32)
    
    # Case 3: Low-rank structure
    r = math.ceil(d / 2)
    Z = np.random.randn(N, r)
    A = np.random.randn(d, r)
    X3 = Z @ A.T
    X3 = X3.astype(np.float32)
    w_true = np.zeros(d)
    w_true[:3] = [1.0, -1.5, 0.7]
    y3 = X3 @ w_true + 0.05 * np.random.randn(N)
    y3 = y3.astype(np.float32)

    return X1, X2, X3, y1, y2, y3

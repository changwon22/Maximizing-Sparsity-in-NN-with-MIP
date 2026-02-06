"""
Baseline sparsity computation using reweighted L1 regularization.
This module provides functions to compute sparsity of neural networks
using continuous optimization with path regularization.
"""

import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class ReLURegNet(nn.Module):
    """Single-hidden-layer ReLU network"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim, bias=True)
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.hidden(x)        # pre-activations [N, K]
        h = self.relu(z)          # activations [N, K]
        out = self.output(h)      # output [N, 1]
        return out, z


def count_nonzero(model, eps=1e-6, incl_bias_sparsity=True):
    """Count nonzeros in path norm ||v_k w_k||_0"""
    with torch.no_grad():
        if incl_bias_sparsity:
            W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
        else:
            W = model.hidden.weight
        V = model.output.weight.view(-1, 1)
        return int((W * V).abs().gt(eps).sum().item())


def count_active_neurons(model, eps=1e-6, incl_bias_sparsity=True):
    """Count active neurons"""
    with torch.no_grad():
        if incl_bias_sparsity:
            W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
        else:
            W = model.hidden.weight
        V = model.output.weight.view(-1, 1)
        return int(((W * V).abs().ge(eps).any(dim=1)).sum().item())


def get_max_params(model):
    """Get maximum absolute values of weights and biases"""
    with torch.no_grad():
        W = model.hidden.weight.detach()
        b = model.hidden.bias.detach()
        v = model.output.weight.detach().view(-1)

        max_W = float(W.abs().max().item())
        max_b = float(b.abs().max().item())
        max_v = float(v.abs().max().item())

        return max_W, max_b, max_v


def get_scaled_params(model, eps=1e-12):
    """
    Returns scaled (W, b, v) where v_k in {-1, 0, +1}
    and scaling is pushed into W, b.

    This normalization makes the output weights binary (-1, 0, 1)
    and absorbs their magnitudes into the hidden layer weights.
    """
    with torch.no_grad():
        W = model.hidden.weight.detach().cpu()   # [K, d]
        b = model.hidden.bias.detach().cpu()     # [K]
        v = model.output.weight.detach().cpu().view(-1)  # [K]

        sign_v = torch.sign(v)
        abs_v = v.abs()
        zero_mask = abs_v <= eps

        W_scaled = W * abs_v[:, None]
        b_scaled = b * abs_v
        v_scaled = sign_v

        # Handle v_k = 0 safely
        W_scaled[zero_mask, :] = 0.0
        b_scaled[zero_mask] = 0.0
        v_scaled[zero_mask] = 0.0

    return (
        W_scaled.numpy(),
        b_scaled.numpy(),
        v_scaled.numpy()
    )


def train_reweighted_l1(X, y, N, d, K, p_values=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                        gamma=0.01, lambda_reg=0.005, num_epochs=5000,
                        eps=1e-6, incl_bias_sparsity=True, seed=42, verbose=False):
    """
    Train ReLU networks with reweighted L1 regularization for different p values.

    Args:
        X: Input data, numpy array of shape (N, d)
        y: Output data, numpy array of shape (N,) or (N, 1)
        N: Number of training samples
        d: Input dimension
        K: Number of hidden units
        p_values: List of regularization parameters to try
        gamma: Learning rate
        lambda_reg: Regularization weight
        num_epochs: Number of training epochs
        eps: Threshold for counting nonzeros
        incl_bias_sparsity: Whether to include biases in sparsity measure
        seed: Random seed
        verbose: Whether to print progress

    Returns:
        dict: Dictionary containing results for each p value and the best model
            {
                'best_p': float,
                'best_sparsity': int,
                'best_active_neurons': int,
                'best_mse': float,
                'w_bound': float (max |W| from best model),
                'b_bound': float (max |b| from best model),
                'best_model': ReLURegNet,
                'all_results': dict of results for each p
            }
    """
    torch.manual_seed(seed)

    # Convert data to torch tensors
    if isinstance(X, np.ndarray):
        x_train = torch.from_numpy(X).float()
    else:
        x_train = X.float()

    if isinstance(y, np.ndarray):
        y_train = torch.from_numpy(y).float()
    else:
        y_train = y.float()

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)

    criterion = nn.MSELoss()

    # Store results for each p value
    all_results = {}

    for p_val in p_values:
        if verbose:
            print(f"Training with p={p_val}...")

        # Initialize model and optimizer
        model = ReLURegNet(d, K)
        optimizer = optim.Adam(model.parameters(), lr=gamma)

        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Store previous parameters
            prev_V = model.output.weight.view(-1, 1)
            if incl_bias_sparsity:
                prev_W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
            else:
                prev_W = model.hidden.weight

            # Forward pass and gradient descent
            optimizer.zero_grad()
            y_pred, _ = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            # Reweighted L1 update (after epoch 5)
            if epoch > 5:
                prev_prod = (prev_V * prev_W)
                C = lambda_reg * gamma * p_val * prev_prod.abs().pow(p_val - 1)
                new_V = model.output.weight.view(-1, 1)
                if incl_bias_sparsity:
                    new_W = torch.cat((model.hidden.weight, model.hidden.bias[:, None].data.clone()), dim=1)
                else:
                    new_W = model.hidden.weight
                new_prod = new_V * new_W
                sign = new_prod.sign()
                mag = new_prod.abs()
                u = torch.where(mag <= C, torch.zeros_like(new_prod), new_prod - C * sign)
                u_norms = torch.linalg.vector_norm(u, dim=1, ord=2)
                mask = (u_norms >= eps)
                den = torch.sqrt(u_norms).masked_fill(~mask, 1.0)
                u_div = (torch.abs(u) / den[:, None]).masked_fill(~mask[:, None], 0.0)

                model.output.weight.data = torch.sqrt(u_norms).reshape(model.output.weight.data.shape) * torch.sign(model.output.weight.data)
                if incl_bias_sparsity:
                    model.hidden.weight.data = u_div[:, :-1].reshape(model.hidden.weight.data.shape) * torch.sign(model.hidden.weight.data)
                    model.hidden.bias.data = u_div[:, -1].reshape(model.hidden.bias.data.shape) * torch.sign(model.hidden.bias.data)
                else:
                    model.hidden.weight.data = u_div.reshape(model.hidden.weight.data.shape) * torch.sign(model.hidden.weight.data)

        # Evaluate final model
        with torch.no_grad():
            y_pred, _ = model(x_train)
            final_mse = criterion(y_pred, y_train).item()
            sparsity = count_nonzero(model, eps, incl_bias_sparsity)
            active_neurons = count_active_neurons(model, eps, incl_bias_sparsity)
            max_W, max_b, max_v = get_max_params(model)

        all_results[p_val] = {
            'sparsity': sparsity,
            'active_neurons': active_neurons,
            'mse': final_mse,
            'max_W': max_W,
            'max_b': max_b,
            'max_v': max_v,
            'model': model
        }

        if verbose:
            print(f"  p={p_val}: sparsity={sparsity}, active_neurons={active_neurons}, "
                  f"mse={final_mse:.6e}, max_W={max_W:.3e}, max_b={max_b:.3e}")

    # Select best model by sparsity
    best_p = min(all_results.keys(), key=lambda p: all_results[p]['sparsity'])
    best_result = all_results[best_p]

    # Get scaled parameters for bounds (used in MILP)
    best_model = best_result['model']
    W_scaled, b_scaled, v_scaled = get_scaled_params(best_model, eps=eps)
    w_bound = float(np.max(np.abs(W_scaled)))
    b_bound = float(np.max(np.abs(b_scaled)))

    if verbose:
        print(f"\nBest model: p={best_p}")
        print(f"  Sparsity: {best_result['sparsity']}")
        print(f"  Active neurons: {best_result['active_neurons']}")
        print(f"  MSE: {best_result['mse']:.6e}")
        print(f"  Bounds (scaled): w_bound={w_bound:.3e}, b_bound={b_bound:.3e}")

    return {
        'best_p': best_p,
        'best_sparsity': best_result['sparsity'],
        'best_active_neurons': best_result['active_neurons'],
        'best_mse': best_result['mse'],
        'w_bound': w_bound,
        'b_bound': b_bound,
        'max_W_original': best_result['max_W'],
        'max_b_original': best_result['max_b'],
        'max_v_original': best_result['max_v'],
        'best_model': best_model,
        'all_results': all_results
    }


def compute_baseline_sparsity(X, y, N, d, K=None, p_values=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                               num_epochs=None, verbose=False):
    """
    Convenience function to compute baseline sparsity and bounds for MILP comparison.

    Args:
        X: Input data, numpy array of shape (N, d)
        y: Output data, numpy array of shape (N,) or (N, 1)
        N: Number of training samples
        d: Input dimension
        K: Number of hidden units (default: N)
        p_values: List of regularization parameters
        num_epochs: Number of epochs (default: 1000 for small problems, 100000 for large)
        verbose: Whether to print progress

    Returns:
        dict: Contains sparsity, bounds, and other metrics
    """
    if K is None:
        K = N

    if num_epochs is None:
        # Auto-select epochs based on problem size
        if N <= 20 and d <= 20:
            num_epochs = 5000
        else:
            num_epochs = 100000

    if verbose:
        print(f"Computing baseline with N={N}, d={d}, K={K}, epochs={num_epochs}")

    results = train_reweighted_l1(
        X, y, N, d, K,
        p_values=p_values,
        num_epochs=num_epochs,
        verbose=verbose
    )

    return results


if __name__ == '__main__':
    # Example usage
    print("Example: Computing baseline sparsity")

    # Generate random data
    N, d = 5, 3
    K = N
    np.random.seed(42)
    X = np.random.uniform(-1, 1, size=(N, d)).astype(np.float32)
    y = np.random.uniform(-1, 1, size=(N,)).astype(np.float32)

    # Compute baseline
    results = compute_baseline_sparsity(X, y, N, d, K, verbose=True)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best p value: {results['best_p']}")
    print(f"Sparsity: {results['best_sparsity']}")
    print(f"Active neurons: {results['best_active_neurons']}")
    print(f"MSE: {results['best_mse']:.6e}")
    print(f"\nBounds for MILP (from scaled parameters):")
    print(f"  w_bound: {results['w_bound']:.6f}")
    print(f"  b_bound: {results['b_bound']:.6f}")

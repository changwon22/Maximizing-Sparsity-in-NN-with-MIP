"""
trainer.py
Functions for training neural network models with different regularization strategies
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.typing as npt
import numpy as np
import copy
import os

if TYPE_CHECKING:
    from .model import ReLURegNet

from .model import ReLURegNet
from .utils import lp_path_norm, count_nonzero, count_active_neurons
from .constants import (
    DEFAULT_P1, DEFAULT_P2, DEFAULT_P3,
    DEFAULT_LEARNING_RATE, DEFAULT_NUM_EPOCHS,
    DEFAULT_LAMBDA_REG, DEFAULT_EPSILON,
    METRICS_RECORDING_INTERVAL,
    REWEIGHTED_L1_START_EPOCH,
    DEFAULT_FIG_DIR
)


def train_models(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    p1: float = DEFAULT_P1,
    p2: float = DEFAULT_P2,
    p3: float = DEFAULT_P3,
    gamma: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    eps: float = DEFAULT_EPSILON,
    incl_bias_sparsity: bool = True,
    save_figs: bool = True,
    fig_dir: str = DEFAULT_FIG_DIR,
    verbose: bool = True
) -> tuple[dict[str, ReLURegNet], dict[str, dict[str, list[float]]], list[int]]:
    """
    Train multiple models with different regularization strategies.

    This function trains 5 models simultaneously:
    1. 'none': No regularization (baseline)
    2. 'wd': Weight decay (L2 regularization via AdamW)
    3. 'p1': Lp path norm with p=p1 (default 0.4)
    4. 'p2': Lp path norm with p=p2 (default 0.7)
    5. 'p3': Lp path norm with p=p3 (default 1.0, equivalent to L1)

    Algorithm Details:
    - Models 1-2 use standard gradient descent
    - Models 3-5 use reweighted L1 proximal updates after epoch 5
    - The proximal operator promotes sparsity in the path-based parameterization
    - Metrics are recorded every 5000 epochs and at the final epoch

    Args:
        X: Input features (numpy array, shape N x d)
        y: Target values (numpy array, shape N x 1 or N)
        p1, p2, p3: Lp norm exponents for regularization (0 < p <= 1 for sparsity)
        gamma: Learning rate for Adam/AdamW optimizers
        num_epochs: Number of training epochs
        lambda_reg: Regularization weight (controls sparsity vs fit tradeoff)
        eps: Threshold below which parameters are considered zero
        incl_bias_sparsity: If True, include biases in sparsity calculations
        save_figs: If True, save training plots to fig_dir
        fig_dir: Directory path for saving figures
        verbose: If True, print training progress every 5000 epochs

    Returns:
        models: Dict mapping model names ('none', 'wd', 'p1', 'p2', 'p3') to trained models
        metrics: Dict of dicts containing lists of MSE, sparsity, active neurons over time
        epochs: List of epoch numbers where metrics were recorded

    Example:
        >>> X, _, _, y, _, _ = data(N=10, d=5)
        >>> models, metrics, epochs = train_models(X, y, num_epochs=10000)
        >>> best_model = models['p1']
        >>> final_sparsity = metrics['p1']['spars'][-1]
    """
    torch.manual_seed(20)
    
    # For saving figures
    if save_figs:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    
    N, d = X.shape
    K = N
    
    # Convert numpy -> torch
    x_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)
    
    # Instantiate models
    model_none = ReLURegNet(d, K)
    model_wd = copy.deepcopy(model_none)
    model_p1 = copy.deepcopy(model_none)
    model_p2 = copy.deepcopy(model_none)
    model_p3 = copy.deepcopy(model_none)

    # Optimizers
    opt_none = optim.Adam(model_none.parameters(), lr=gamma)
    opt_wd = optim.AdamW(model_wd.parameters(), lr=gamma, weight_decay=lambda_reg)
    opt_p1 = optim.Adam(model_p1.parameters(), lr=gamma)
    opt_p2 = optim.Adam(model_p2.parameters(), lr=gamma)
    opt_p3 = optim.Adam(model_p3.parameters(), lr=gamma)

    criterion = nn.MSELoss()

    # History storage
    epochs = []
    metrics = {
        'none': {'mse': [], 'spars': [], 'act': []},
        'wd': {'mse': [], 'spars': [], 'act': []},
        'p1': {'mse': [], 'spars': [], 'act': [], 'lp': []},
        'p2': {'mse': [], 'spars': [], 'act': [], 'lp': []},
        'p3': {'mse': [], 'spars': [], 'act': [], 'lp': []},
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # No regularization
        opt_none.zero_grad()
        y_none, _ = model_none(x_train)
        loss_none = criterion(y_none, y_train)
        loss_none.backward()
        opt_none.step()

        # AdamW
        opt_wd.zero_grad()
        y_wd, _ = model_wd(x_train)
        loss_wd = criterion(y_wd, y_train)
        loss_wd.backward()
        opt_wd.step()
        
        p_models_opts = [(p1, model_p1, opt_p1), (p2, model_p2, opt_p2), (p3, model_p3, opt_p3)]
        
        for p, model, opt in p_models_opts:
            prev_V = model.output.weight.view(-1, 1)
            if incl_bias_sparsity:
                prev_W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
            else:
                prev_W = model.hidden.weight
            
            opt.zero_grad()
            y_pred, _ = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            opt.step()
        
            if epoch > REWEIGHTED_L1_START_EPOCH:  # reweighted l1 update
                prev_prod = (prev_V * prev_W)
                C = lambda_reg * gamma * p * prev_prod.abs().pow(p - 1)
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
        
        # Record and print
        if epoch % METRICS_RECORDING_INTERVAL == 0 or epoch == 1 or epoch == num_epochs:
            epochs.append(epoch)
            for name, model in [('none', model_none), ('wd', model_wd), ('p1', model_p1), ('p2', model_p2), ('p3', model_p3)]:
                y_pred, _ = model(x_train)
                metrics[name]['mse'].append(criterion(y_pred, y_train).item())
                metrics[name]['spars'].append(count_nonzero(model, eps, incl_bias_sparsity))
                metrics[name]['act'].append(count_active_neurons(model, eps, incl_bias_sparsity))
                if name == 'p1':
                    metrics[name]['lp'].append(lp_path_norm(model, p1, incl_bias_sparsity))
                elif name == 'p2':
                    metrics[name]['lp'].append(lp_path_norm(model, p2, incl_bias_sparsity))
                elif name == 'p3':
                    metrics[name]['lp'].append(lp_path_norm(model, p3, incl_bias_sparsity))

            if verbose:
                print(f"Epoch {epoch}:  "
                      f"No reg (MSE={metrics['none']['mse'][-1]:.2e},spars={metrics['none']['spars'][-1]},act={metrics['none']['act'][-1]}) | "
                      f"WD (MSE={metrics['wd']['mse'][-1]:.2e},spars={metrics['wd']['spars'][-1]},act={metrics['wd']['act'][-1]}) | "
                      f"p={p1} (MSE={metrics['p1']['mse'][-1]:.2e},spars={metrics['p1']['spars'][-1]},act={metrics['p1']['act'][-1]}) | "
                      f"p={p2} (MSE={metrics['p2']['mse'][-1]:.2e},spars={metrics['p2']['spars'][-1]},act={metrics['p2']['act'][-1]}) | "
                      f"p={p3} (MSE={metrics['p3']['mse'][-1]:.2e},spars={metrics['p3']['spars'][-1]},act={metrics['p3']['act'][-1]}) | ")
    
    models = {
        'none': model_none,
        'wd': model_wd,
        'p1': model_p1,
        'p2': model_p2,
        'p3': model_p3
    }
    
    return models, metrics, epochs

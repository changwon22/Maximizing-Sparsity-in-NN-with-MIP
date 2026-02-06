import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import numpy as np
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train ReLU networks with various regularization parameters')

    # Regularization parameters
    parser.add_argument('--p', type=float, nargs='+', default=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                        help='Array of regularization parameters (e.g., --p 0.05 0.1 0.25 0.5 0.75 1.0)')

    # Network dimensions
    parser.add_argument('--N', type=int, required=False, default=10,
                        help='Number of training samples (default: 10)')
    parser.add_argument('--d', type=int, required=False, default=50,
                        help='Input dimension (default: 50)')
    parser.add_argument('--K', type=int, required=False, default=None,
                        help='Number of hidden units (default: N*10)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs (default: 1000 for small problems, 100000 for large)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--lambda_reg', type=float, default=0.005,
                        help='Regularization weight (default: 0.005)')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='Threshold for counting nonzeros (default: 1e-6)')

    # Other options
    parser.add_argument('--incl_bias_sparsity', action='store_true', default=True,
                        help='Include and penalize biases in sparsity measure (default: True)')
    parser.add_argument('--no_bias_sparsity', dest='incl_bias_sparsity', action='store_false',
                        help='Exclude biases from sparsity measure')
    parser.add_argument('--save_figs', action='store_true', default=True,
                        help='Save plots as .png (default: True)')
    parser.add_argument('--no_save_figs', dest='save_figs', action='store_false',
                        help='Do not save plots')
    parser.add_argument('--fig_dir', type=str, default='figs/high_d/',
                        help='Directory to save figures (default: figs/high_d/)')
    parser.add_argument('--seed', type=int, default=20,
                        help='Random seed (default: 20)')
    parser.add_argument('--log_interval', type=int, default=5000,
                        help='Interval for logging metrics (default: 5000)')

    return parser.parse_args()


# Define single-hidden-layer ReLU network
class ReLURegNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # hidden layer weights w_k and biases b_k
        self.hidden = nn.Linear(input_dim, hidden_dim, bias=True)
        # output weights v_k (no bias)
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.hidden(x)        # pre-activations [N, K]
        h = self.relu(z)          # activations [N, K]
        out = self.output(h)      # output [N, 1]
        return out, z


def lp_path_norm(model, p, incl_bias_sparsity):
    """Calculate lp path norm"""
    if incl_bias_sparsity:
        W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
    else:
        W = model.hidden.weight
    V = model.output.weight.view(-1, 1)
    return (W * V).abs().pow(p).sum().item()


def count_nonzero(model, eps, incl_bias_sparsity):
    """Count nonzeros (l0 path norm)"""
    if incl_bias_sparsity:
        W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
    else:
        W = model.hidden.weight
    V = model.output.weight.view(-1, 1)
    return int((W * V).abs().gt(eps).sum().item())


def count_active_neurons(model, eps, incl_bias_sparsity):
    """Count active neurons"""
    if incl_bias_sparsity:
        W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
    else:
        W = model.hidden.weight
    V = model.output.weight.view(-1, 1)
    return int(((W * V).abs().ge(eps).any(dim=1)).sum().item())


def train_model(args):
    # Set random seed
    torch.manual_seed(args.seed)

    # Network dimensions
    N, d = args.N, args.d
    K = args.K if args.K is not None else N * 10

    # Auto-select number of epochs based on problem size if not specified
    if args.num_epochs is None:
        # Use 1000 epochs for small problems, 100000 for large ones
        if N <= 20 and d <= 20:
            num_epochs = 1000
        else:
            num_epochs = 100000
    else:
        num_epochs = args.num_epochs

    print(f"Network configuration: N={N}, d={d}, K={K}")
    print(f"Training for {num_epochs} epochs")
    print(f"Regularization parameters p: {args.p}")

    # Dataset: N random d-dimensional points and random scalar labels
    x_train = torch.zeros(N, d)
    x_train.uniform_(-1, 1)
    y_train = torch.zeros(N, 1)
    y_train.uniform_(-1, 1)

    # Create figure directory if needed
    if args.save_figs:
        if not os.path.exists(args.fig_dir):
            os.makedirs(args.fig_dir)

    # Instantiate models
    model_none = ReLURegNet(d, K)
    model_wd = copy.deepcopy(model_none)

    # Create models for each p value
    p_models = {}
    for p_val in args.p:
        p_models[p_val] = copy.deepcopy(model_none)

    # Optimizers
    opt_none = optim.Adam(model_none.parameters(), lr=args.gamma)
    opt_wd = optim.AdamW(model_wd.parameters(), lr=args.gamma, weight_decay=args.lambda_reg)

    p_opts = {}
    for p_val in args.p:
        p_opts[p_val] = optim.Adam(p_models[p_val].parameters(), lr=args.gamma)

    criterion = nn.MSELoss()

    # History storage
    epochs = []
    metrics = {
        'none': {'mse': [], 'spars': [], 'act': []},
        'wd': {'mse': [], 'spars': [], 'act': []},
    }

    for p_val in args.p:
        metrics[f'p{p_val}'] = {'mse': [], 'spars': [], 'act': [], 'lp': []}

    # Training loop
    print("\nStarting training...")
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

        # Train models with different p values
        for p_val in args.p:
            model = p_models[p_val]
            opt = p_opts[p_val]

            prev_V = model.output.weight.view(-1, 1)
            if args.incl_bias_sparsity:
                prev_W = torch.cat((model.hidden.weight, model.hidden.bias[:, None]), dim=1)
            else:
                prev_W = model.hidden.weight

            opt.zero_grad()
            y_pred, _ = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            opt.step()

            if epoch > 5:  # reweighted l1 update
                prev_prod = (prev_V * prev_W)
                C = args.lambda_reg * args.gamma * p_val * prev_prod.abs().pow(p_val - 1)
                new_V = model.output.weight.view(-1, 1)
                if args.incl_bias_sparsity:
                    new_W = torch.cat((model.hidden.weight, model.hidden.bias[:, None].data.clone()), dim=1)
                else:
                    new_W = model.hidden.weight
                new_prod = new_V * new_W
                sign = new_prod.sign()
                mag = new_prod.abs()
                u = torch.where(mag <= C, torch.zeros_like(new_prod), new_prod - C * sign)
                u_norms = torch.linalg.vector_norm(u, dim=1, ord=2)
                mask = (u_norms >= args.eps)
                den = torch.sqrt(u_norms).masked_fill(~mask, 1.0)
                u_div = (torch.abs(u) / den[:, None]).masked_fill(~mask[:, None], 0.0)

                model.output.weight.data = torch.sqrt(u_norms).reshape(model.output.weight.data.shape) * torch.sign(model.output.weight.data)
                if args.incl_bias_sparsity:
                    model.hidden.weight.data = u_div[:, :-1].reshape(model.hidden.weight.data.shape) * torch.sign(model.hidden.weight.data)
                    model.hidden.bias.data = u_div[:, -1].reshape(model.hidden.bias.data.shape) * torch.sign(model.hidden.bias.data)
                else:
                    model.hidden.weight.data = u_div.reshape(model.hidden.weight.data.shape) * torch.sign(model.hidden.weight.data)

        # Record and print
        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.num_epochs:
            epochs.append(epoch)

            # Record metrics for no regularization
            y_pred, _ = model_none(x_train)
            metrics['none']['mse'].append(criterion(y_pred, y_train).item())
            metrics['none']['spars'].append(count_nonzero(model_none, args.eps, args.incl_bias_sparsity))
            metrics['none']['act'].append(count_active_neurons(model_none, args.eps, args.incl_bias_sparsity))

            # Record metrics for AdamW
            y_pred, _ = model_wd(x_train)
            metrics['wd']['mse'].append(criterion(y_pred, y_train).item())
            metrics['wd']['spars'].append(count_nonzero(model_wd, args.eps, args.incl_bias_sparsity))
            metrics['wd']['act'].append(count_active_neurons(model_wd, args.eps, args.incl_bias_sparsity))

            # Record metrics for each p value
            for p_val in args.p:
                model = p_models[p_val]
                y_pred, _ = model(x_train)
                key = f'p{p_val}'
                metrics[key]['mse'].append(criterion(y_pred, y_train).item())
                metrics[key]['spars'].append(count_nonzero(model, args.eps, args.incl_bias_sparsity))
                metrics[key]['act'].append(count_active_neurons(model, args.eps, args.incl_bias_sparsity))
                metrics[key]['lp'].append(lp_path_norm(model, p_val, args.incl_bias_sparsity))

            # Print progress
            log_str = f"Epoch {epoch}:  "
            log_str += f"No reg (MSE={metrics['none']['mse'][-1]:.2e},spars={metrics['none']['spars'][-1]},act={metrics['none']['act'][-1]}) | "
            log_str += f"WD (MSE={metrics['wd']['mse'][-1]:.2e},spars={metrics['wd']['spars'][-1]},act={metrics['wd']['act'][-1]}) | "

            for p_val in args.p:
                key = f'p{p_val}'
                log_str += f"p={p_val} (MSE={metrics[key]['mse'][-1]:.2e},spars={metrics[key]['spars'][-1]},act={metrics[key]['act'][-1]}) | "

            print(log_str)

    # Plot sparsity over time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['none']['spars'], '-', label='no regularization')
    plt.plot(epochs, metrics['wd']['spars'], '-', label='AdamW WD')

    line_styles = ['-', '-.', ':', '--']
    for i, p_val in enumerate(args.p):
        key = f'p{p_val}'
        style = line_styles[i % len(line_styles)]
        if i == len(args.p) - 1 and len(args.p) > 2:
            plt.plot(epochs, metrics[key]['spars'], ':', color='black', label=f'reweighted $\ell^1$ (p = {p_val})')
        else:
            plt.plot(epochs, metrics[key]['spars'], style, label=f'reweighted $\ell^1$ (p = {p_val})')

    plt.xlabel('epoch')
    plt.ylabel('sparsity ($\Sigma_k \| v_k w_k \|_0$)')
    plt.yscale('log')
    plt.axhline(y=2 * N, color='gray', linestyle='--', label=f'$2N$ (upper bound on true min sparsity)')
    plt.title(f'sparsity over time: $N = {N}, d = {d}$, data/labels in Unif[-1,1]')
    plt.legend()

    if args.save_figs:
        fig_path = os.path.join(args.fig_dir, 'sparsity_over_time_high_d.png')
        plt.savefig(fig_path, dpi=300)
        print(f"\nFigure saved to: {fig_path}")

    plt.show()

    return metrics, epochs


def main():
    args = parse_args()
    metrics, epochs = train_model(args)
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
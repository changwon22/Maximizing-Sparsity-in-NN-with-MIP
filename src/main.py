import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import time
from datetime import datetime
import signal
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Import MILP solver
from functions import solve_MILP

# Import baseline sparsity computation
from baseline import compute_baseline_sparsity

# Import data generation
from data import data


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for time limiting operations"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def run_baseline(X, y, N, d, K, p_values=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                 num_epochs=1000, timeout_seconds=300):
    """
    Run baseline reweighted L1 method with timeout

    Args:
        X: Input data (N, d)
        y: Output data (N,) or (N, 1)
        N: Number of samples
        d: Input dimension
        K: Number of hidden units
        p_values: List of regularization parameters
        num_epochs: Number of training epochs
        timeout_seconds: Time limit in seconds (default: 300 = 5 minutes)

    Returns:
        dict: Results including sparsity, bounds, time, status
    """
    start_time = time.time()
    status = 'success'

    try:
        with time_limit(timeout_seconds):
            results = compute_baseline_sparsity(
                X, y, N, d, K,
                p_values=p_values,
                num_epochs=num_epochs,
                verbose=True
            )
            training_time = time.time() - start_time

            return {
                'sparsity': results['best_sparsity'],
                'active_neurons': results['best_active_neurons'],
                'mse': results['best_mse'],
                'w_bound': results['w_bound'],
                'b_bound': results['b_bound'],
                'best_p': results['best_p'],
                'time': training_time,
                'status': status
            }

    except TimeoutException:
        status = 'timeout'
        training_time = time.time() - start_time

        return {
            'sparsity': float('nan'),
            'active_neurons': float('nan'),
            'mse': float('nan'),
            'w_bound': 1.0,  # default bound
            'b_bound': 1.0,  # default bound
            'best_p': float('nan'),
            'time': training_time,
            'status': status
        }


def run_milp(X, y, w_bound=1.0, b_bound=1.0, timeout_seconds=300):
    """
    Run MILP method with timeout - simplified wrapper around functions.solve_MILP()

    Args:
        X: Input data (N, d)
        y: Output data (N,) or (N, 1)
        w_bound: Bound for |w_ik|
        b_bound: Bound for |b_k|
        timeout_seconds: Time limit in seconds (default: 300 = 5 minutes)

    Returns:
        sparsity: Number of nonzero weights
        obj_val: Objective value
        training_time: Time taken
        status: Status string
    """
    from gurobipy import GRB

    # Ensure y is 1D
    if len(y.shape) > 1:
        y = y.flatten()

    start_time = time.time()

    try:
        # Call solve_MILP from functions.py with timeout
        _, _, _, obj_val, gurobi_status = solve_MILP(
            X, y, w_bound, b_bound,
            output_flag=1,
            timeout_seconds=timeout_seconds
        )

        training_time = time.time() - start_time

        # Convert Gurobi status to string
        if gurobi_status == GRB.OPTIMAL:
            status = 'optimal'
        elif gurobi_status == GRB.TIME_LIMIT:
            status = 'timeout'
        elif gurobi_status == GRB.SUBOPTIMAL:
            status = 'suboptimal'
        else:
            status = 'infeasible'

        sparsity = obj_val

    except Exception:
        training_time = time.time() - start_time
        sparsity = float('inf')
        obj_val = float('inf')
        status = 'error'

    return sparsity, obj_val, training_time, status


def compare_methods(N_values, d_values, K_values=None, p_values=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                   num_epochs_baseline=1000, timeout_seconds=300,
                   output_file='comparison_results.xlsx', data_case=1):
    """
    Compare MILP and Baseline (Reweighted L1) methods across different problem sizes

    Args:
        N_values: List of N values (number of samples)
        d_values: List of d values (input dimensions)
        K_values: List of K values (hidden units), if None use K=N
        p_values: List of regularization parameters for baseline
        num_epochs_baseline: Number of epochs for baseline method
        timeout_seconds: Time limit per method per iteration (default: 300 = 5 minutes)
        output_file: Output Excel file name
        data_case: Which data case to use (1, 2, or 3)
                   1: Random uniform data
                   2: Ground-truth ReLU network with noise
                   3: Low-rank linear data
    """
    results = []

    for N in N_values:
        for d in d_values:
            K = K_values if K_values is not None else N

            # Generate data using data() function
            X1, X2, X3, y1, y2, y3 = data(N, d)

            # Select appropriate case
            if data_case == 1:
                X = X1
                y = y1.flatten() if len(y1.shape) > 1 else y1
            elif data_case == 2:
                X = X2
                y = y2.flatten() if len(y2.shape) > 1 else y2
            elif data_case == 3:
                X = X3
                y = y3.flatten() if len(y3.shape) > 1 else y3
            else:
                raise ValueError(f"Invalid data_case: {data_case}. Must be 1, 2, or 3.")

            result = {
                'N': N,
                'd': d,
                'K': K,
                'data_case': data_case,
            }

            # Run Baseline first to get bounds for MILP
            try:
                baseline_result = run_baseline(
                    X, y, N, d, K,
                    p_values=p_values,
                    num_epochs=num_epochs_baseline,
                    timeout_seconds=timeout_seconds
                )
                result['Baseline_sparsity'] = baseline_result['sparsity']
                result['Baseline_active_neurons'] = baseline_result['active_neurons']
                result['Baseline_mse'] = baseline_result['mse']
                result['Baseline_w_bound'] = baseline_result['w_bound']
                result['Baseline_b_bound'] = baseline_result['b_bound']
                result['Baseline_best_p'] = baseline_result['best_p']
                result['Baseline_time'] = baseline_result['time']
                result['Baseline_status'] = baseline_result['status']

                # Get bounds with epsilon margin for MILP
                epsilon = 1.0
                w_bound = baseline_result['w_bound'] + epsilon
                b_bound = baseline_result['b_bound'] + epsilon

            except Exception:
                result['Baseline_sparsity'] = float('nan')
                result['Baseline_active_neurons'] = float('nan')
                result['Baseline_mse'] = float('nan')
                result['Baseline_w_bound'] = float('nan')
                result['Baseline_b_bound'] = float('nan')
                result['Baseline_best_p'] = float('nan')
                result['Baseline_time'] = float('nan')
                result['Baseline_status'] = 'error'
                w_bound = 1.0
                b_bound = 1.0

            # Run MILP with bounds from baseline
            try:
                milp_sparsity, milp_obj, milp_time, milp_status = run_milp(
                    X, y, w_bound=w_bound, b_bound=b_bound,
                    timeout_seconds=timeout_seconds
                )
                result['MILP_sparsity'] = milp_sparsity
                result['MILP_obj_val'] = milp_obj
                result['MILP_time'] = milp_time
                result['MILP_status'] = milp_status
                result['MILP_w_bound'] = w_bound
                result['MILP_b_bound'] = b_bound
            except Exception:
                result['MILP_sparsity'] = float('nan')
                result['MILP_obj_val'] = float('nan')
                result['MILP_time'] = float('nan')
                result['MILP_status'] = 'error'
                result['MILP_w_bound'] = w_bound
                result['MILP_b_bound'] = b_bound

            # Calculate comparison metrics
            if result['MILP_sparsity'] != float('nan') and result['Baseline_sparsity'] != float('nan'):
                result['sparsity_diff'] = result['Baseline_sparsity'] - result['MILP_sparsity']
                result['sparsity_ratio'] = result['Baseline_sparsity'] / result['MILP_sparsity'] if result['MILP_sparsity'] > 0 else float('inf')
                result['time_diff'] = result['Baseline_time'] - result['MILP_time']
            else:
                result['sparsity_diff'] = float('nan')
                result['sparsity_ratio'] = float('nan')
                result['time_diff'] = float('nan')

            results.append(result)

            # Save intermediate results
            # df = pd.DataFrame(results)
            # df.to_excel(output_file, index=False)

    # Create summary
    df = pd.DataFrame(results)

    # Save final results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = output_file.replace('.xlsx', f'_{timestamp}.xlsx')
    df.to_excel(final_output, index=False)

    return df


if __name__ == '__main__':
    # Configuration
    N_values = list(range(2, 7))                # Number of training samples: 1 to 5
    d_values = list(range(2, 7))                # Input dimensions: 1 to 5
    K_values = None                              # If None, K = N for each iteration

    p_values = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]  # Regularization parameters for baseline
    num_epochs_baseline = 5000                    # Number of epochs for baseline
    timeout_seconds = 300                         # 5 minutes time limit per method

    # Data generation case:
    # 1 = Random uniform [-1, 1]
    # 2 = Ground-truth ReLU network with noise
    # 3 = Low-rank linear data
    data_case = 1

    output_file = 'comparison_results.xlsx'

    # Run comparison
    results_df = compare_methods(
        N_values=N_values,
        d_values=d_values,
        K_values=K_values,
        p_values=p_values,
        num_epochs_baseline=num_epochs_baseline,
        timeout_seconds=timeout_seconds,
        output_file=output_file,
        data_case=data_case
    )

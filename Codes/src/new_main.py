import numpy as np
import pandas as pd
import time
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB
from data import data
from baseline import compute_baseline_sparsity
from src.formulation import MINLP1, MINLP2, MILP1, MILP2

def run_milp(X, y, w_bound=1.0, b_bound=1.0, timeout_seconds=300):
    from gurobipy import GRB

    if len(y.shape) > 1:
        y = y.flatten()

    start_time = time.time()

    try:
        _, _, _, obj_val, gurobi_status = MILP1(
            X, y, w_bound, b_bound,
            output_flag=1,
            timeout_seconds=timeout_seconds
        )
        training_time = time.time() - start_time

        if gurobi_status == GRB.OPTIMAL:
            status = 'optimal'
        elif gurobi_status == GRB.TIME_LIMIT:
            status = 'timeout'
        elif gurobi_status == GRB.SUBOPTIMAL:
            status = 'suboptimal'
        else:
            status = 'infeasible'

        sparsity = obj_val

    except Exception as e:
        training_time = time.time() - start_time
        sparsity = float('inf')
        status = f'error: {e}'

    return sparsity, training_time, status


if __name__ == '__main__':
    N_values = list(range(2, 7))
    d_values = list(range(2, 7))
    data_case = 1
    w_bound = 2.0
    b_bound = 2.0
    timeout_seconds = 300

    results = []

    for N in N_values:
        for d in d_values:
            X1, X2, X3, y1, y2, y3 = data(N, d)

            if data_case == 1:
                X, y = X1, y1.flatten() if len(y1.shape) > 1 else y1
            elif data_case == 2:
                X, y = X2, y2.flatten() if len(y2.shape) > 1 else y2
            else:
                X, y = X3, y3.flatten() if len(y3.shape) > 1 else y3

            sparsity, milp_time, status = run_milp(
                X, y, w_bound=w_bound, b_bound=b_bound,
                timeout_seconds=timeout_seconds
            )

            results.append({
                'N': N, 'd': d, 'K': N,
                'sparsity': sparsity,
                'time': milp_time,
                'status': status,
            })

            print(f"N={N}, d={d} | sparsity={sparsity:.2f} | time={milp_time:.2f}s | {status}")

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'milp_results_{timestamp}.csv', index=False)
    print(df)
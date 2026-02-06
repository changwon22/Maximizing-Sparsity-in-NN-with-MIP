"""
milp_solver.py
MILP solver for finding optimal sparse ReLU networks
"""

from __future__ import annotations
from typing import Any
import numpy as np
import numpy.typing as npt
import gurobipy as gp
from gurobipy import GRB

# Type alias for Gurobi variable dictionaries
GurobiVarDict = Any  # gp.tupledict of Vars


def solve_MILP(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    w_Bound: float,
    b_Bound: float,
    output_flag: int = 1,
    time_limit: float | None = None
) -> tuple[GurobiVarDict | None, GurobiVarDict | None, GurobiVarDict | None, float | None, str]:
    """
    Solve MILP for finding the sparsest ReLU network that fits data exactly.

    Mathematical Formulation:
        minimize    sum_{i,k} t_{i,k} + sum_k s_k
        subject to  y_n = sum_k (2*r_{n,k} - p_{n,k})  for all n
                    W^T x_n + b = p_{n,k} - q_{n,k}    for all n,k
                    p_{n,k} <= M * z_{n,k}              for all n,k
                    q_{n,k} <= M * (1 - z_{n,k})        for all n,k
                    r_{n,k} = v'_k * p_{n,k}            (McCormick relaxation)
                    -w_Bound * t_{i,k} <= W_{i,k} <= w_Bound * t_{i,k}
                    -b_Bound * s_k <= b_k <= b_Bound * s_k

    Variables:
        W_{i,k}: Continuous weights
        b_k: Continuous biases
        v'_k: Binary output indicators
        t_{i,k}, s_k: Binary sparsity indicators
        p_{n,k}, q_{n,k}: ReLU decomposition
        z_{n,k}: Binary ReLU activation indicators
        r_{n,k}: Auxiliary for bilinear product

    Big-M constant: M = d * w_Bound * max|X| + b_Bound

    Args:
        X: Input features (N x d numpy array)
        y: Target values (N x 1 or 1D numpy array)
        w_Bound: Upper bound for weight magnitudes
        b_Bound: Upper bound for bias magnitudes
        output_flag: Show Gurobi output (1=show, 0=hide)
        time_limit: Time limit in seconds (None=unlimited)

    Returns:
        W: Weight matrix Gurobi variables (or None if failed)
        b: Bias vector Gurobi variables (or None if failed)
        v_: Binary output weights (or None if failed)
        ObjVal: Total nonzero parameters (or None if failed)
        status: Optimization status ('optimal', 'time_limit', etc.)

    Example:
        >>> X, _, _, y, _, _ = data(N=5, d=3)
        >>> W, b, v_, obj, status = solve_MILP(X, y, w_Bound=2.0, b_Bound=1.0)
        >>> if obj is not None:
        ...     W_sol, b_sol, v_sol = extract_solution(W, b, v_, d=3, K=5)
        ...     print(f"Found solution with {obj} nonzero parameters")
    """

    # Input validation
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be numpy array, got {type(X)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be numpy array, got {type(y)}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")

    if w_Bound <= 0:
        raise ValueError(f"w_Bound must be positive, got {w_Bound}")
    if b_Bound <= 0:
        raise ValueError(f"b_Bound must be positive, got {b_Bound}")

    if time_limit is not None and time_limit <= 0:
        raise ValueError(f"time_limit must be positive, got {time_limit}")

    # Ensure y is 1D
    if len(y.shape) > 1:
        y = y.flatten()

    if len(y) != X.shape[0]:
        raise ValueError(f"X and y must have same number of samples: X has {X.shape[0]}, y has {len(y)}")
    
    # Input X has d dimension and N number of data
    N, d = X.shape
    # Width K = N
    K = N

    # Create Gurobi model
    m = gp.Model("Sparse_ReLU_Network")

    # Set time limit if specified
    if time_limit is not None:
        m.setParam('TimeLimit', time_limit)

    # 1. Decision Variables
    # Input weights
    W = m.addVars(d, K, lb=-w_Bound, ub=w_Bound, vtype=GRB.CONTINUOUS, name="W")
    # Biases
    b = m.addVars(K, lb=-b_Bound, ub=b_Bound, vtype=GRB.CONTINUOUS, name="b")
    # Output weights (binary: -1, 0, or +1)
    v_ = m.addVars(K, vtype=GRB.BINARY, name="v'")

    # ReLU: positive part
    p = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    # ReLU: negative part
    q = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    # Neuron activation indicator
    z = m.addVars(N, K, vtype=GRB.BINARY, name="z")
    # Nonzero weight indicator
    t = m.addVars(d, K, vtype=GRB.BINARY, name="t")
    # Nonzero bias indicator
    s = m.addVars(K, vtype=GRB.BINARY, name="s")
    # Auxiliary variable (for McCormick relaxation)
    r = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="r")

    # 2. Objective Function: Minimize total number of nonzero parameters
    m.setObjective(
        gp.quicksum(t[i, k] for i in range(d) for k in range(K))
        + gp.quicksum(s[k] for k in range(K)),
        GRB.MINIMIZE
    )

    # 3. Constraints
    
    # Linear combination constraint: W^T x + b = p - q
    for n in range(N):
        for k in range(K):
            m.addConstr(
                gp.quicksum(W[i, k] * X[n, i] for i in range(d)) + b[k]
                == p[n, k] - q[n, k],
                name=f"linear_{n}_{k}"
            )
    
    # ReLU constraints using Big-M method
    X_absmax = np.max(np.abs(X)) if X.size > 0 else 0.0
    M = d * w_Bound * X_absmax + b_Bound
    
    for n in range(N):
        for k in range(K):
            # If z[n,k] = 1, then p[n,k] can be positive (q[n,k] must be 0)
            m.addConstr(p[n, k] <= M * z[n, k], name=f"relu_pos_{n}_{k}")
            # If z[n,k] = 0, then q[n,k] can be positive (p[n,k] must be 0)
            m.addConstr(q[n, k] <= M * (1 - z[n, k]), name=f"relu_neg_{n}_{k}")
    
    # Data fitting constraint: sum_k (2*r[n,k] - p[n,k]) = y[n]
    for n in range(N):
        m.addConstr(
            gp.quicksum(2 * r[n, k] - p[n, k] for k in range(K)) == y[n],
            name=f"fit_{n}"
        )
    
    # McCormick envelope constraints for r[n,k] = v_[k] * p[n,k]
    for n in range(N):
        for k in range(K):
            m.addConstr(r[n, k] <= M * v_[k], name=f"mccormick1_{n}_{k}")
            m.addConstr(p[n, k] - M * (1 - v_[k]) <= r[n, k], name=f"mccormick2_{n}_{k}")
            m.addConstr(r[n, k] <= p[n, k], name=f"mccormick3_{n}_{k}")
    
    # Sparsity indicator constraints
    # If W[i,k] != 0, then t[i,k] = 1
    for i in range(d):
        for k in range(K):
            m.addConstr(-w_Bound * t[i, k] <= W[i, k], name=f"sparse_w_lb_{i}_{k}")
            m.addConstr(W[i, k] <= w_Bound * t[i, k], name=f"sparse_w_ub_{i}_{k}")
    
    # If b[k] != 0, then s[k] = 1
    for k in range(K):
        m.addConstr(-b_Bound * s[k] <= b[k], name=f"sparse_b_lb_{k}")
        m.addConstr(b[k] <= b_Bound * s[k], name=f"sparse_b_ub_{k}")

    # Set output flag
    if output_flag == 0:
        m.setParam('OutputFlag', 0)
    
    # Optimize
    m.optimize()

    # Determine optimization status
    if m.status == GRB.OPTIMAL:
        status = "optimal"
    elif m.status == GRB.TIME_LIMIT:
        status = "time_limit"
    elif m.status == GRB.SUBOPTIMAL:
        status = "suboptimal"
    elif m.status == GRB.INFEASIBLE:
        status = "infeasible"
    elif m.status == GRB.UNBOUNDED:
        status = "unbounded"
    else:
        status = f"other_{m.status}"

    # Print results
    print(f"\nModel status: {m.status}")
    if m.status == GRB.OPTIMAL:
        print(f"Status: Optimal solution found")
        print(f"Objective value (total sparsity): {m.ObjVal:.6f}")
    elif m.status == GRB.SUBOPTIMAL:
        print(f"Status: Suboptimal solution found")
        print(f"Objective value (total sparsity): {m.ObjVal:.6f}")
    elif m.status == GRB.TIME_LIMIT:
        if hasattr(m, 'ObjVal'):
            print(f"Status: Time limit reached with feasible solution")
            print(f"Objective value (total sparsity): {m.ObjVal:.6f}")
        else:
            print("Status: Time limit reached without feasible solution")
    elif m.status == GRB.INFEASIBLE:
        print("Status: Model is infeasible")
    elif m.status == GRB.UNBOUNDED:
        print("Status: Model is unbounded")
    else:
        print(f"Status: Optimization ended with status {m.status}")

    # Return solution with status
    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        try:
            return W, b, v_, m.ObjVal, status
        except AttributeError:
            # TIME_LIMIT without feasible solution
            return None, None, None, None, status
    else:
        return None, None, None, None, status


def extract_solution(
    W: GurobiVarDict,
    b: GurobiVarDict,
    v_: GurobiVarDict,
    d: int,
    K: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Extract the solution from Gurobi variables

    Args:
        W: Weight Gurobi variables
        b: Bias Gurobi variables
        v_: Output weight Gurobi variables
        d: Input dimension
        K: Number of hidden units

    Returns:
        W_sol: Weight matrix as numpy array (d x K)
        b_sol: Bias vector as numpy array (K,)
        v_sol: Output weights as numpy array (K,)
    """
    W_sol = np.zeros((d, K))
    for i in range(d):
        for k in range(K):
            W_sol[i, k] = W[i, k].X

    b_sol = np.array([b[k].X for k in range(K)])
    v_sol = np.array([v_[k].X for k in range(K)])

    return W_sol, b_sol, v_sol

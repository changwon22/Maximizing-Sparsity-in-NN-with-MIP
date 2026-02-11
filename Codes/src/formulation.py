"""
    Inputs:
        X : numpy array of shape (N, d), input x^n
        y : numpy array of shape (N,), output y^n
        K : int, width of single-hidden-layer
        w_Bound: bound for |w_ik|
        b_Bound: bound for |b_k|

    Returns:
        W      : Gurobi Var dicts for input weight
        b      : Gurobi Var dicts for bias
        v_     : Gurobi Var dict for output weight
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB

# MINLP 1 : Fischetti & Jo
def MINLP1(X, y, w_Bound, b_Bound, v_Bound, output_flag=1, timeout_seconds=None):

    N, d = X.shape
    K = N

    m = gp.Model()

    W = m.addVars(d, K, lb=-w_Bound, ub=w_Bound, vtype=GRB.CONTINUOUS, name="W")
    b = m.addVars(K, lb=-b_Bound, ub=b_Bound, vtype=GRB.CONTINUOUS, name="b")
    v = m.addVars(K, lb=-v_Bound, ub=v_Bound, vtype=GRB.CONTINUOUS, name="v")


# MINLP 2 : output weight Normalized to {-1,+1}
def MINLP2(X, y, w_Bound, b_Bound, output_flag=1, timeout_seconds=None):
    return 0

# MILP 1 : McCormick
def MILP1(X, y, w_Bound, b_Bound, output_flag=1, timeout_seconds=None):
    
    # Input X has d dimension and N number of data
    N, d = X.shape
    # Width K = N
    K = N

    # GurobiPy
    m = gp.Model()

    # 1. Decision Variables
    # input weight
    W = m.addVars(d, K, lb=-w_Bound, ub=w_Bound, vtype=GRB.CONTINUOUS, name="W")
    # bias
    b = m.addVars(K, lb=-b_Bound, ub=b_Bound, vtype=GRB.CONTINUOUS, name="b")
    # output weight
    v_ = m.addVars(K, vtype=GRB.BINARY, name="v'")

    # ReLU: positive
    p = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    # ReLU: negative
    q = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    # Neuron Activation
    z = m.addVars(N, K, vtype=GRB.BINARY, name="z")
    # Number of Nonzero Weight
    t = m.addVars(d, K, vtype=GRB.BINARY, name="t")
    # Number of Nonzero Bias
    s = m.addVars(K, vtype=GRB.BINARY, name="s")
    # Auxiliary Variable (for McCormick)
    r = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="r")

    # 2. Objective Functions
    m.setObjective(
        gp.quicksum(t[i, k] for i in range(d) for k in range(K))
        +gp.quicksum(s[k] for k in range(K)),
        GRB.MINIMIZE
    )

    # 3. Constraints
    # Output of linear combination
    for n in range(N):
        for k in range(K):
            m.addConstr(
                gp.quicksum(W[i, k] * X[n, i] for i in range(d)) + b[k]
                == p[n, k] - q[n, k]
            )
    # Output of ReLU   
        # Big-M
    X_absmax = np.max(np.abs(X)) if X.size > 0 else 0.0
    M = d * w_Bound * X_absmax + b_Bound
    for n in range(N):
        for k in range(K):
            m.addConstr(p[n, k] <= M * z[n, k])
            m.addConstr(q[n, k] <= M * (1 - z[n, k]))
    # Data Fitting
    for n in range(N):
        m.addConstr(
            gp.quicksum(2 * r[n, k] - p[n, k] for k in range(K)) == y[n]
        )
    # McCormick
    for n in range(N):
        for k in range(K):
            m.addConstr(r[n, k] <= M * v_[k])
            m.addConstr(p[n, k] - M * (1 - v_[k]) <= r[n, k])
            m.addConstr(r[n, k] <= p[n, k])
    # Nonzero Weight/Bias Indicator
    for i in range(d):
        for k in range(K):
            m.addConstr(-w_Bound * t[i, k] <= W[i, k])
            m.addConstr(W[i, k] <= w_Bound * t[i, k])
    for k in range(K):
        m.addConstr(-b_Bound * s[k] <= b[k])
        m.addConstr(b[k] <= b_Bound * s[k])

    # Set output flag
    m.setParam('OutputFlag', output_flag)

    # Set timeout if specified
    if timeout_seconds is not None:
        m.setParam('TimeLimit', timeout_seconds)

    # Solve
    m.optimize()

    # Print
    if output_flag:
        print(f"\nModel status: {m.status}")
        if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            print(f"Objective value (sum t_ik): {m.ObjVal:.6f}")

    # Return results with status
    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return W, b, v_, m.ObjVal, m.status
    elif m.status == GRB.TIME_LIMIT:
        # Return best solution found so far
        obj_val = m.ObjVal if m.SolCount > 0 else float('inf')
        return W, b, v_, obj_val, m.status
    else:
        return W, b, v_, float('inf'), m.status

# MILP 2 : Symmetry Handling
def MILP2(X, y, w_Bound, b_Bound, output_flag=1, timeout_seconds=None):
    N, d = X.shape
    K = N
    m = gp.Model()

    W = m.addVars(d, K, lb=-w_Bound, ub=w_Bound, vtype=GRB.CONTINUOUS, name="W")
    b = m.addVars(K, lb=-b_Bound, ub=b_Bound, vtype=GRB.CONTINUOUS, name="b")
    v_ = m.addVars(K, vtype=GRB.BINARY, name="v'")
    p = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    q = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    z = m.addVars(N, K, vtype=GRB.BINARY, name="z")
    t = m.addVars(d, K, vtype=GRB.BINARY, name="t")
    s = m.addVars(K, vtype=GRB.BINARY, name="s")
    r = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="r")

    m.setObjective(
        gp.quicksum(t[i, k] for i in range(d) for k in range(K))
        +gp.quicksum(s[k] for k in range(K)),
        GRB.MINIMIZE
    )

    for n in range(N):
        for k in range(K):
            m.addConstr(
                gp.quicksum(W[i, k] * X[n, i] for i in range(d)) + b[k]
                == p[n, k] - q[n, k]
            )

    X_absmax = np.max(np.abs(X)) if X.size > 0 else 0.0
    M = d * w_Bound * X_absmax + b_Bound
    for n in range(N):
        for k in range(K):
            m.addConstr(p[n, k] <= M * z[n, k])
            m.addConstr(q[n, k] <= M * (1 - z[n, k]))

    for n in range(N):
        m.addConstr(
            gp.quicksum(2 * r[n, k] - p[n, k] for k in range(K)) == y[n]
        )

    for n in range(N):
        for k in range(K):
            m.addConstr(r[n, k] <= M * v_[k])
            m.addConstr(p[n, k] - M * (1 - v_[k]) <= r[n, k])
            m.addConstr(r[n, k] <= p[n, k])
    
    for i in range(d):
        for k in range(K):
            m.addConstr(-w_Bound * t[i, k] <= W[i, k])
            m.addConstr(W[i, k] <= w_Bound * t[i, k])
    for k in range(K):
        m.addConstr(-b_Bound * s[k] <= b[k])
        m.addConstr(b[k] <= b_Bound * s[k])

    # Symmetry Handling
    m.setParam('Symmetry', 2)
    for k in range(K - 1):
        m.addConstr(
            W[0, k] <= W[0, k + 1],
            name=f"sym_W_{k}"
        )


    m.setParam('OutputFlag', output_flag)
    if timeout_seconds is not None:
        m.setParam('TimeLimit', timeout_seconds)
    m.optimize()

    # Print
    if output_flag:
        print(f"\nModel status: {m.status}")
        if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            print(f"Objective value (sum t_ik): {m.ObjVal:.6f}")

    # Return results with status
    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return W, b, v_, m.ObjVal, m.status
    elif m.status == GRB.TIME_LIMIT:
        # Return best solution found so far
        obj_val = m.ObjVal if m.SolCount > 0 else float('inf')
        return W, b, v_, obj_val, m.status
    else:
        return W, b, v_, float('inf'), m.status
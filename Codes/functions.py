import numpy as np
import gurobipy as gp
from gurobipy import GRB

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

# 1. First Stage (Feasibility) Problem: Finding feasible width K

# 2. Second Stage (Sparsity) Problem: For a given K, sparsify the NN weights

# 2.1. MINLP

# 2.2. MILP
def solve_MILP(X, y, w_Bound, b_Bound, output_flag=1):
    
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

    # Solve
    # m.setParam('OutputFlag', 0)
    m.optimize()

    # Print
    print(f"\nModel status: {m.status}")
    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"Objective value (sum t_ik): {m.ObjVal:.6f}")

    return W, b, v_, m.ObjVal
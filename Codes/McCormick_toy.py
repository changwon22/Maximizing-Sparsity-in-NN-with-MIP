import numpy as np
import gurobipy as gp
from gurobipy import GRB


def print_solution_and_check_new(m, X, y, W, b, vprime, v, p, q, z, r, t):
    """
    Simple sanity checks for the 'New MILP Formulation'.
    """
    if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"Model status is {m.status}; no solution values to print.")
        return

    N, d = X.shape
    K = len(b)

    # Extract solution
    W_opt = np.array([[W[i, k].X for i in range(d)] for k in range(K)])  # shape (K, d)
    b_opt = np.array([b[k].X for k in range(K)])                          # shape (K,)
    vprime_opt = np.array([vprime[k].X for k in range(K)])                # shape (K,)
    v_opt = np.array([v[k].X for k in range(K)])                          # shape (K,)

    P = np.array([[p[n, k].X for k in range(K)] for n in range(N)])       # (N, K)
    Q = np.array([[q[n, k].X for k in range(K)] for n in range(N)])       # (N, K)
    Z = np.array([[z[n, k].X for k in range(K)] for n in range(N)])       # (N, K)
    R = np.array([[r[n, k].X for k in range(K)] for n in range(N)])       # (N, K)
    T = np.array([[t[i, k].X for k in range(K)] for i in range(d)])       # (d, K)

    print("\n=== Optimized Parameters (New MILP Formulation) ===")
    print("W (K x d) =\n", W_opt)
    print("b (K,)    =\n", b_opt)
    print("v' (K,)   =\n", vprime_opt)
    print("v  (K,)   =\n", v_opt)
    print("t (d x K) =\n", T)

    # Check v == 2 v' - 1
    print("\nChecking v = 2 v' - 1:")
    for k in range(K):
        lhs = v_opt[k]
        rhs = 2 * vprime_opt[k] - 1
        print(f"k={k}: v={lhs:.3f}, 2v'-1={rhs:.3f}, diff={abs(lhs-rhs):.3e}")

    # Check affine split: w_k^T x^n + b_k = p^n_k - q^n_k
    print("\nChecking affine constraints (w_k^T x^n + b_k ?= p^n_k - q^n_k):")
    for n in range(N):
        lhs = W_opt @ X[n] + b_opt           # shape (K,)
        rhs = P[n] - Q[n]                    # shape (K,)
        err = np.max(np.abs(lhs - rhs))
        print(f"sample {n}: max|lhs-rhs| = {err:.3e}")

    # Big-M consistency for ReLU split
    print("\nBig-M ReLU split consistency:")
    for n in range(N):
        # p should be ~0 if z=0; q should be ~0 if z=1
        leak_p = np.max(P[n] * (1 - Z[n]))   # p shouldn't be active if z=0
        leak_q = np.max(Q[n] * Z[n])         # q shouldn't be active if z=1
        print(f"sample {n}: max(p*(1-z))={leak_p:.3e}, max(q*z)={leak_q:.3e}")

    # Check McCormick output vs y: sum_k (2 r^n_k - p^n_k) == y^n
    print("\nChecking outputs (sum_k (2 r[n,k] - p[n,k]) ?= y[n]):")
    for n in range(N):
        pred = np.sum(2 * R[n] - P[n])
        print(f"sample {n}: y_true={y[n]:.6f}, y_pred={pred:.6f}, err={abs(pred-y[n]):.3e}")


def solve_new_milp(X, y, K, Bw=1.0, Bb=1.0, M=None, output_flag=0):
    """
    Solve the 'New MILP Formulation' as specified in the LaTeX code.

    Inputs:
        X : numpy array of shape (N, d), features x^n
        y : numpy array of shape (N,), targets y^n
        K : int, number of hidden neurons
        Bw: bound for |w_{ik}|
        Bb: bound for |b_k|
        M : Big-M for affine/ReLU/McCormick. If None, computed from data.
        output_flag: pass to Gurobi's OutputFlag parameter.

    Returns:
        m      : Gurobi model
        W, b   : Gurobi Var dicts for weights/bias
        vprime : Gurobi Var dict for v'_k
        v      : Gurobi Var dict for v_k
        p, q   : Gurobi Var dicts for positive/negative parts
        z      : Gurobi Var dict for ReLU indicator
        r      : Gurobi Var dict for McCormick product
        t      : Gurobi Var dict for nonzero-path indicator
    """
    N, d = X.shape

    m = gp.Model("New_MILP_Formulation")

    # === Variables ===
    # W: w_{ik} \in R, i in [d], k in [K]
    W = m.addVars(d, K, lb=-Bw, ub=Bw, vtype=GRB.CONTINUOUS, name="W")

    # b: b_k \in R
    b = m.addVars(K, lb=-Bb, ub=Bb, vtype=GRB.CONTINUOUS, name="b")

    # v'_k \in {0,1}, v_k = 2 v'_k - 1, v_k \in [-1,1]
    vprime = m.addVars(K, vtype=GRB.BINARY, name="vprime")
    v = m.addVars(K, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="v")

    # p^n_k, q^n_k >= 0
    p = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    q = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="q")

    # z^n_k \in {0,1}
    z = m.addVars(N, K, vtype=GRB.BINARY, name="z")

    # t_{ik} \in {0,1}
    t = m.addVars(d, K, vtype=GRB.BINARY, name="t")

    # r^n_k, McCormick product for r^n_k = p^n_k * v'_k
    r = m.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name="r")

    # === Big-M selection ===
    # Use bound on |w| and |b| plus max|x| to bound |w^T x + b|
    X_absmax = np.max(np.abs(X)) if X.size > 0 else 0.0
    M_affine = d * Bw * X_absmax + Bb
    if M_affine <= 0:
        M_affine = 1.0
    if M is not None:
        M_affine = M

    # For weight-path indicator we can use M_w = Bw
    M_w = Bw

    # === Constraints ===

    # v_k = 2 v'_k - 1
    for k in range(K):
        m.addConstr(v[k] == 2 * vprime[k] - 1, name=f"sign_{k}")

    # 1) Output of linear combination:
    #    sum_i w_{ik} x_i^n + b_k = p^n_k - q^n_k
    for n in range(N):
        for k in range(K):
            m.addConstr(
                gp.quicksum(W[i, k] * X[n, i] for i in range(d)) + b[k]
                == p[n, k] - q[n, k],
                name=f"affine_{n}_{k}"
            )

    # 2) ReLU activation big-M part:
    #    p^n_k, q^n_k >= 0 (already via lb)
    #    p^n_k <= M z^n_k
    #    q^n_k <= M (1 - z^n_k)
    for n in range(N):
        for k in range(K):
            m.addConstr(p[n, k] <= M_affine * z[n, k],            name=f"pBigM_{n}_{k}")
            m.addConstr(q[n, k] <= M_affine * (1 - z[n, k]),      name=f"qBigM_{n}_{k}")

    # 3) Exact data fitting:
    #    sum_k (2 r^n_k - p^n_k) = y^n
    for n in range(N):
        m.addConstr(
            gp.quicksum(2 * r[n, k] - p[n, k] for k in range(K)) == y[n],
            name=f"fit_{n}"
        )

    # 4) McCormick constraints for r^n_k = p^n_k * v'_k with v'_k in {0,1}:
    #    0 <= r^n_k <= M v'_k
    #    p^n_k - M (1 - v'_k) <= r^n_k <= p^n_k
    for n in range(N):
        for k in range(K):
            m.addConstr(r[n, k] <= M_affine * vprime[k],           name=f"r_up_{n}_{k}")
            # lb r >= 0 is already set
            m.addConstr(p[n, k] - M_affine * (1 - vprime[k]) <= r[n, k],
                        name=f"r_lowMc_{n}_{k}")
            m.addConstr(r[n, k] <= p[n, k],                         name=f"r_upMc_{n}_{k}")

    # 5) Nonzero weight path indicator:
    #    -M t_{ik} <= w_{ik} <= M t_{ik}
    for i in range(d):
        for k in range(K):
            m.addConstr(-M_w * t[i, k] <= W[i, k], name=f"path_lo_{i}_{k}")
            m.addConstr(W[i, k] <= M_w * t[i, k],  name=f"path_hi_{i}_{k}")

    # === Objective: Sparsity ===
    #    min sum_{i=1}^d sum_{k=1}^K t_{ik}
    m.setObjective(
        gp.quicksum(t[i, k] for i in range(d) for k in range(K)),
        GRB.MINIMIZE
    )

    # Solve
    m.setParam('OutputFlag', 1)
    m.optimize()

    print(f"\nModel status: {m.status}")
    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"Objective value (sum t_ik): {m.ObjVal:.6f}")
        print_solution_and_check_new(m, X, y, W, b, vprime, v, p, q, z, r, t)

    return m, W, b, vprime, v, p, q, z, r, t


# ============================
# Small example / usage
# ============================
if __name__ == "__main__":
    np.random.seed(0)

    # Data generation
    N = 7
    d = 5
    K = 7  # number of hidden neurons

    X = np.random.rand(N, d)
    y = np.random.rand(N)

    print(f"Generated {N} random data points with {d} dimensions:\n")
    for n in range(N):
        print(f"Sample {n}: X = {X[n]}, y = {y[n]}")

    # Solve the new MILP formulation
    m, W, b, vprime, v, p, q, z, r, t = solve_new_milp(X, y, K, Bw=1.0, Bb=1.0, M=None, output_flag=0)
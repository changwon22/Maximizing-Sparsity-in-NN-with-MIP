import numpy as np
import gurobipy as gp
from gurobipy import GRB

np.random.seed(0)
    
# Second Stage (Sparsity) Problem

# Solution Verification

def print_solution_and_check(m, X, y, W, b, v, p, q, z, N, d, l1):
    # Make sure the model is solved
    if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"Model status is {m.status}; no solution values to print.")
        return

    # Extract solution values to NumPy
    W_opt = np.array([[W[i, j].X for j in range(d)] for i in range(l1)])
    b_opt = np.array([b[i].X for i in range(l1)])
    v_opt = np.array([v[i].X for i in range(l1)])

    P = np.array([[p[n, i].X for i in range(l1)] for n in range(N)])
    Q = np.array([[q[n, i].X for i in range(l1)] for n in range(N)])
    Z = np.array([[z[n, i].X for i in range(l1)] for n in range(N)])

    print("\nOptimized Parameters:")
    print("W =\n", W_opt)
    print("b =\n", b_opt)
    print("v =\n", v_opt)

    # Sanity checks on affine split: W x + b = p - q
    print("\nChecking affine constraints (W x + b ?= p - q):")
    for n in range(N):
        lhs = W_opt @ X[n] + b_opt           # shape (l1,)
        rhs = P[n] - Q[n]                    # shape (l1,)
        err = np.max(np.abs(lhs - rhs))
        print(f"sample {n}: max|lhs-rhs| = {err:.3e}")

    # Big-M split consistency (optional)
    print("\nBig-M split consistency (should be small if tight):")
    for n in range(N):
        # q_i should be ~0 when z=0; p_i ~0 when z=1
        leak_q = np.max(Q[n] * (1 - Z[n]))   # q shouldn't be active if z=0
        leak_p = np.max(P[n] * Z[n])         # p shouldn't be active if z=1
        print(f"sample {n}: max(q*(1-z))={leak_q:.3e}, max(p*z)={leak_p:.3e}")

    # Output check: sum_i p[n,i] * v[i] == y[n]
    print("\nChecking outputs (sum_i p[n,i]*v[i] ?= y[n]):")
    pred = P @ v_opt                         # shape (N,)
    for n in range(N):
        print(f"sample {n}: y_true={y[n]:.6f}, y_pred={pred[n]:.6f}, err={abs(pred[n]-y[n]):.3e}")

def second_stage(N,d,K,X,y):

    m = gp.Model()

    Bw = 1.1
    W = m.addVars(l1, d, lb=-Bw, ub=Bw, vtype=GRB.CONTINUOUS, name="W")
    Bb = 1.1
    b = m.addVars(l1,lb=-Bb, ub=Bb, vtype=GRB.CONTINUOUS, name="b")
    Bv = 1.1
    v = m.addVars(l1, lb=-Bv, ub=Bv, vtype=GRB.CONTINUOUS, name="v")

    p = m.addVars(N, l1, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    q = m.addVars(N, l1, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    z = m.addVars(N, l1, vtype=GRB.BINARY, name="z")

    Ms = Bw * Bv
    s = m.addVars(l1, d, lb=-Ms, ub=Ms, vtype=GRB.CONTINUOUS, name="s")
    t = m.addVars(l1, d, vtype=GRB.BINARY, name="t")

    m.setObjective(gp.quicksum(t[i,j] for i in range(l1) for j in range(d)), GRB.MINIMIZE)

    for n in range(N):
        for i in range(l1):
            m.addConstr(
                gp.quicksum(W[i,j] * X[n, j] for j in range(d)) + b[i]
                == p[n, i] - q[n, i],
                name=f"affine_{n}_{i}"
            )

    M = d*Bw + Bb
    for n in range(N):
        for i in range(l1):
            m.addConstr(p[n,i] <= M * (1 - z[n,i]), name=f"pBigM_{n}_{i}")
            m.addConstr(q[n,i] <= M * z[n,i],       name=f"qBigM_{n}_{i}")

    for n in range(N):
        m.addConstr(
            gp.quicksum(p[n,i] * v[i] for i in range(l1)) == y[n],
            name=f"out_{n}"
        )

    for i in range(l1):
        for j in range(d):
            m.addConstr(s[i,j] == W[i,j] * v[i],       name=f"s_def_{i}_{j}")
            m.addConstr( s[i,j] <=  Ms * t[i,j],        name=f"s_up_{i}_{j}")
            m.addConstr(-s[i,j] <=  Ms * t[i,j],        name=f"s_lo_{i}_{j}")

    # m.Params.NonConvex = 2
    m.setParam('OutputFlag', 0)
    m.optimize()
    print(m.status)
    print_solution_and_check(m, X, y, W, b, v, p, q, z, N, d, l1)
    return m.ObjVal, W, b, v

# Data Generation

# Random_Case 1

N = 5
d = 3

X = np.random.rand(N, d)
y = np.random.rand(N)

print(f"Generated {N} random data points with {d} dimensions:\n")

for i in range(N):
    print(f"Sample {i+1}: X = {X[i]}, y = {y[i]}")


###

i = 1
optimal_value = -1

while optimal_value != 0:
    optimal_value = first_stage(N, d, i, X, y)

    print(f"Iteration with i = {i},  Optimal Value: {optimal_value}")

    if optimal_value == -1:
        print("Model was infeasible or an error occurred. Stopping loop.")
        break

    if i > N:
        print("Reached iteration limit. Stopping loop.")
        break

    if optimal_value == 0:
        l1=i
        break

    i += 1

# second_stage

objval, W, b, v = second_stage(N,d,l1+5,X,y)
print(objval)

# Parameters

print("\nOptimized Parameters:")
print(W)
# W_opt = np.array([[W[i, j].X for j in range(d)] for i in range(l1)])
# b_opt = np.array([b[i].X for i in range(l1)])
# v_opt = np.array([v[i].X for i in range(l1)])

# print("W =", W_opt)
# print("b =", b_opt)
# print("v =", v_opt)

# print("\nChecking predictions against input data:")
# for n in range(N):
#     # affine layer values (like ReLU split variables)
#     affine_vals = [sum(W_opt[i, j] * X[n, j] for j in range(d)) + b_opt[i] for i in range(l1)]
#     output_val = sum(max(0, affine_vals[i]) * v_opt[i] for i in range(l1))  # simplified check
#     print(f"Sample {n}: y_true = {y[n]:.4f}, model_output ≈ {output_val:.4f}")
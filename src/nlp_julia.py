import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_nlp(m,K):

    # t = np.linspace(0, 1, m + 2)
    t = np.random.uniform(0, 1, m)   # m random points in (0,1)
    t = np.concatenate(([0], t, [1]))  # add endpoints
    t.sort()
    print("t =", t)
    y = np.random.rand(m+2)
    print("y =", y)

    # M[i,j] = max(0, t[i] - t[j])
    M = np.maximum(0, t[:, None] - t[None, :])

    #GurobiPy
    model = gp.Model()

    # 1.Decision Variables
    v = model.addVars(K,vtype=GRB.CONTINUOUS, name="v")
    # |v|=z
    z = model.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    u = model.addVars(m+2,K,vtype=GRB.CONTINUOUS, name="u")
    u_ = model.addVars(m+2,K,vtype=GRB.CONTINUOUS, name="u_")

    # |u|=w, |u_|=w_
    w = model.addVars(m+2,K,lb=0.0,vtype=GRB.CONTINUOUS, name="w")
    w_ = model.addVars(m+2,K,lb=0.0,vtype=GRB.CONTINUOUS, name="w_")

    # 2.Objective Function
    model.setObjective(gp.quicksum(z[k] for k in range(K)), GRB.MINIMIZE)

    # 3.Constraints
    # Absolute Values for v, u, u_tilde
    for k in range(K):
        model.addConstr(v[k]<=z[k])
        model.addConstr(-z[k]<=v[k])
    for i in range(m+2):
        for k in range(K):
            model.addConstr(u[i,k]<=w[i,k])
            model.addConstr(-w[i,k]<=u[i,k])
            model.addConstr(u_[i,k]<=w_[i,k])
            model.addConstr(-w_[i,k]<=u_[i,k])

    # Constraint (1)
    for k in range(K):
        model.addConstr(
            (gp.quicksum(w[i,k]+w_[i,k] for i in range(m+2)))<=1
        )
    
    # Constraint (2)
    for j in range(m+2):
        model.addConstr(
            gp.quicksum(v[k]*gp.quicksum(u[k,i]*M[j,i]+u_[k,i]*M[i,j] for i in range (m+2)) for k in range(K))==y[j]
        )
        
    # Constraint (3)
    for k in range(K):
        for j in range(m+1):
            model.addConstr(
                gp.quicksum(u[k,i]*M[j,i]+u_[k,i]*M[i,j] for i in range(m+2))*gp.quicksum(u[k,i]*M[j+1,i]+u_[k,i]*M[i,j+1]for i in range(m+2))>=0
            )

    # Solve
    model.optimize()
    # Print
    print(f"\nModel status: {model.status}")
    if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"Objective value (sum t_ik): {model.ObjVal:.6f}")

    v_val  = [v[k].X for k in range(K)]
    u_val  = [[u[i, k].X for k in range(K)] for i in range(m + 2)]
    u_val_ = [[u_[i, k].X for k in range(K)] for i in range(m + 2)]

    return opt_val, v_val, u_val, u_val_


# Example
m=3
K=m+2
    
opt_val,v,u,u_=solve_nlp(m,K)

print("\noptimal(minimum) value:", opt_val)

print("\nv:")
for k in range(K):
    print(f"v[{k}] = {v[k]}")

print("\nu:")
for i in range(m + 2):
    print([u[i][k] for k in range(K)])

print("\nu_:")
for i in range(m + 2):
    print([u_[i][k] for k in range(K)])
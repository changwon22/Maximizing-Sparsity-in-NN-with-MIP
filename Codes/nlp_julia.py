import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Given Values
m=3
K=m+2
t = np.linspace(0, 1, m + 2)
y = np.random.rand(m+2)

# GurobiPy
m = gp.Model()

# Decision Variables
v = m.addVars(K,vtype=GRB.CONTINUOUS, name="v")
# |v|=z
z = m.addVars(K,lb=0.0, vtype=GRB.CONTINUOUS, name="z")

u = m.addVars(m+1,K,vtype=GRB.CONTINUOUS, name="u")
u_ = m.addVars(m+1,K,vtype=GRB.CONTINUOUS, name="u_")

# |u|=w, |u_|=w_
w = m.addVars(m+1,K,lb=0.0,vtype=GRB.CONTINUOUS, name="u")
w_ = m.addVars(m+1,K,lb=0.0,vtype=GRB.CONTINUOUS, name="u_")

# Objective Function
m.setObjective(gp.quicksum((z[k]) for k in range(K)), GRB.MINIMIZE)

# Constraint 1
for k in range(K):
    m.addConstr(
                gp.quicksum(W[i,j] * X[n, j] for j in range(d)) + b[i]
                == p[n, i] - q[n, i],
                name=f"affine_{n}_{i}"
            )

# Constraint 2
 
# Constraint 3            

for n in range(N):
        for i in range(l1):
            m.addConstr(
                gp.quicksum(W[i,j] * X[n, j] for j in range(d)) + b[i]
                == p[n, i] - q[n, i],
                name=f"affine_{n}_{i}"
            )

m.addConstr(p[n,i] <= M * (1 - z[n,i]))


    Bw = 1.1
    W = m.addVars(l1, d, lb=-Bw, ub=Bw, vtype=GRB.CONTINUOUS, name="W")
    Bb = 1.1
    b = m.addVars(l1,lb=-Bb, ub=Bb, vtype=GRB.CONTINUOUS, name="b")
    Bv = 1.1
    v = m.addVars(l1, lb=-Bv, ub=Bv, vtype=GRB.CONTINUOUS, name="v")

    p = m.addVars(N, l1, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    q = m.addVars(N, l1, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    z = m.addVars(N, l1, vtype=GRB.BINARY, name="z")

    f = m.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    g = m.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="g")

    

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
            gp.quicksum(p[n,i] * v[i] for i in range(l1)) - y[n]==f[n]-g[n],
            name=f"out_{n}"
        )

m.optimize()
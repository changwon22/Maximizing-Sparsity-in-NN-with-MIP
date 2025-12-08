import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Data Generation
from data import data
N, d = 10, 3
X1, X2, X3, y1, y2, y3 = data(N, d)

# First Stage (Feasibility) Problem
    
# Second Stage (Sparsity) Problem
from functions import solve_MILP
w_Bound=1.0
b_Bound=1.0
W, b, v_, ObjVal = solve_Symmetry(X3, y3, w_Bound, b_Bound)
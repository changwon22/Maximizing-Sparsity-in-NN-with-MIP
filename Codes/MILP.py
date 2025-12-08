import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Data Generation
from data import data
N, d = 7, 3
X1, X2, X3, y1, y2, y3 = data(N, d)

# First Stage (Feasibility) Problem
    
# Second Stage (Sparsity) Problem
from functions import solve_MILP
w_Bound=3.0
b_Bound=3.0
W, b, v_, ObjVal = solve_MILP(X1, y1, w_Bound, b_Bound)
import numpy as np
import math

def relu(a):
    return np.maximum(a, 0.0)

# Data Generation
def data(N,d):
    np.random.seed(0)
    
    # Case 1: Random - Uniform (-1,1)
    X1 = np.random.uniform(-1, 1, size=(N, d))
    y1 = np.random.uniform(-1, 1, size=(N, 1))

    # Case 2: Functional relationship - Ground-Truth ReLU network
    X2 = np.random.uniform(-1, 1, size=(N, d))
    K_true = math.ceil(N / 2)
    W_true = np.random.randn(d, K_true) * 0.5
    b_true = np.random.randn(K_true) * 0.1
    v_true  = np.random.randn(K_true) * 0.5
    hidden_layer = relu(X2 @ W_true + b_true)
    y = hidden_layer @ v_true
    noise = 0.05 * np.random.randn(N)
    y2 = y + noise
    
    # Case 3:
    r = math.ceil(d / 2)
    Z = np.random.randn(N, r)
    A = np.random.randn(d, r)
    X3 = Z @ A.T
    w_true = np.zeros(d)
    w_true[:3] = [1.0, -1.5, 0.7]
    y3= X3 @ w_true + 0.05 * np.random.randn(N)

    return X1, X2, X3, y1, y2, y3
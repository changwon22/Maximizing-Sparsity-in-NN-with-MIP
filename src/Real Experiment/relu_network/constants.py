"""
constants.py - Configuration constants for the relu_network package

Centralizes magic numbers and default configurations used throughout
the package, making them easy to find and modify.
"""

# Training defaults
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_EPOCHS = 100000
DEFAULT_LAMBDA_REG = 0.005
DEFAULT_EPSILON = 1e-6

# Regularization p-values
DEFAULT_P1 = 0.4
DEFAULT_P2 = 0.7
DEFAULT_P3 = 1.0

# MILP defaults
DEFAULT_MILP_EPSILON_MARGIN = 1.0

# Training algorithm parameters
REWEIGHTED_L1_START_EPOCH = 5
METRICS_RECORDING_INTERVAL = 5000

# Plotting defaults
DEFAULT_FIG_DIR = 'figs/high_d/'

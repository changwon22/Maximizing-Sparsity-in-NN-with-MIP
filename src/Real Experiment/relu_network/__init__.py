"""
relu_network package
A modular implementation for training and optimizing sparse ReLU networks
"""

from .data_generation import data, relu
from .model import ReLURegNet
from .utils import lp_path_norm, count_nonzero, count_active_neurons, get_max_params
from .trainer import train_models
from .milp_solver import solve_MILP, extract_solution

__all__ = [
    'data',
    'relu',
    'ReLURegNet',
    'lp_path_norm',
    'count_nonzero',
    'count_active_neurons',
    'get_max_params',
    'train_models',
    'solve_MILP',
    'extract_solution'
]

__version__ = '1.0.0'

# Sparse ReLU Network Package

A modular Python package for training and optimizing sparse single-hidden-layer ReLU neural networks using continuous optimization and Mixed-Integer Linear Programming (MILP).

## Package Structure

```
.
├── main.py                          # Main execution script
├── README.md                        # This file
└── relu_network/                    # Main package directory
    ├── __init__.py                  # Package initialization
    ├── data_generation.py           # Data generation functions
    ├── model.py                     # Neural network model definition
    ├── utils.py                     # Utility functions for analysis
    ├── trainer.py                   # Training functions
    └── milp_solver.py              # MILP solver for optimal sparse networks
```

## Module Descriptions

### 1. `data_generation.py`
Contains functions for generating synthetic datasets:
- `relu(a)`: Apply ReLU activation
- `data(N, d)`: Generate three types of datasets (random, functional ReLU, low-rank)

### 2. `model.py`
Defines the neural network architecture:
- `ReLURegNet`: Single-hidden-layer ReLU network class

### 3. `utils.py`
Utility functions for model analysis:
- `lp_path_norm()`: Calculate lp path norm
- `count_nonzero()`: Count nonzero parameters (l0 norm)
- `count_active_neurons()`: Count active neurons
- `get_max_params()`: Extract maximum parameter values

### 4. `trainer.py`
Training functionality with multiple regularization strategies:
- `train_models()`: Train models with no regularization, weight decay, and three lp penalties

### 5. `milp_solver.py`
MILP optimization for finding optimal sparse networks:
- `solve_MILP()`: Formulate and solve MILP using Gurobi
- `extract_solution()`: Extract solution from Gurobi variables

### 6. `main.py`
Main execution script that orchestrates the entire pipeline:
1. Data generation
2. Model training with continuous optimization
3. Best model selection
4. MILP optimization for optimal sparse solution

## Usage

### Basic Usage

Run the main script:

```bash
python main.py
```

### Using Individual Modules

You can also import and use individual modules:

```python
from relu_network import data, ReLURegNet, train_models, solve_MILP

# Generate data
X1, X2, X3, y1, y2, y3 = data(N=7, d=5)

# Train models
models, metrics, epochs = train_models(X1, y1)

# Solve MILP
W, b, v_, obj_val = solve_MILP(X1, y1, w_Bound=2.0, b_Bound=1.0)
```

### Custom Training Example

```python
from relu_network import data, train_models, get_max_params

# Generate data
X, _, _, y, _, _ = data(N=10, d=8)

# Train with custom parameters
models, metrics, epochs = train_models(
    X, y,
    p1=0.3,
    p2=0.5,
    p3=1.0,
    gamma=0.005,
    num_epochs=50000,
    lambda_reg=0.01,
    verbose=True
)

# Analyze best model
best_model = models['p1']
max_W, max_b, max_v = get_max_params(best_model)
print(f"Max weight: {max_W:.4f}")
```

## Requirements

- Python 3.7+
- NumPy
- PyTorch
- Gurobi (with valid license)
- Pandas (optional, imported but not actively used in current version)

## Installation

1. Install dependencies:
```bash
pip install numpy torch pandas
```

2. Install Gurobi:
   - Download from https://www.gurobi.com
   - Install gurobipy: `pip install gurobipy`
   - Obtain and activate a license

## Features

- **Multiple Regularization Strategies**: Compare no regularization, weight decay, and lp path norm penalties
- **Modular Design**: Easy to extend and modify individual components
- **MILP Optimization**: Find provably optimal sparse solutions using Gurobi
- **Comprehensive Metrics**: Track MSE, sparsity, active neurons, and lp norms during training

## Algorithm Details

### Continuous Optimization
The package trains models using:
- **No regularization**: Standard MSE loss
- **Weight decay**: L2 regularization via AdamW
- **lp path norm**: Reweighted l1 proximal updates for sparsity

### MILP Formulation
The MILP finds the sparsest network by:
- Minimizing the number of nonzero weights and biases
- Subject to exact data fitting constraints
- Using Big-M method for ReLU activation
- McCormick relaxation for bilinear terms

## License

This code is based on research in sparse neural network optimization.

## Contact

For questions or issues, please refer to the original Jupyter notebook or contact the authors.

# Project Structure Overview

## File Organization

```
outputs/
│
├── main.py                          # Main execution script
├── example_usage.py                 # Usage examples
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
│
└── relu_network/                    # Core package
    ├── __init__.py                  # Package initialization & exports
    ├── data_generation.py           # Data generation utilities
    ├── model.py                     # Neural network architecture
    ├── utils.py                     # Analysis utilities
    ├── trainer.py                   # Training algorithms
    └── milp_solver.py              # MILP optimization
```

## Module Dependencies

```
main.py
  │
  ├─→ relu_network.data_generation
  ├─→ relu_network.train_models
  ├─→ relu_network.solve_MILP
  ├─→ relu_network.extract_solution
  └─→ relu_network.get_max_params

trainer.py
  │
  ├─→ model.ReLURegNet
  └─→ utils.{lp_path_norm, count_nonzero, count_active_neurons}

milp_solver.py
  │
  └─→ gurobipy (external)

example_usage.py
  │
  └─→ All relu_network modules
```

## Key Functions by Module

### data_generation.py
- `relu(a)` - Apply ReLU activation
- `data(N, d)` - Generate 3 types of datasets

### model.py
- `ReLURegNet` - Single hidden layer ReLU network
  - `__init__(input_dim, hidden_dim)`
  - `forward(x)` → (output, pre_activation)

### utils.py
- `lp_path_norm(model, p, incl_bias_sparsity)` - Calculate lp path norm
- `count_nonzero(model, eps, incl_bias_sparsity)` - Count nonzero params
- `count_active_neurons(model, eps, incl_bias_sparsity)` - Count active neurons
- `get_max_params(model)` - Get max absolute parameter values

### trainer.py
- `train_models(X, y, ...)` - Train 5 models with different regularizers
  - Returns: models dict, metrics dict, epochs list

### milp_solver.py
- `solve_MILP(X, y, w_Bound, b_Bound, output_flag)` - Solve MILP
- `extract_solution(W, b, v_, d, K)` - Extract numpy arrays from Gurobi

## Data Flow

```
1. Data Generation
   data(N, d) → X, y

2. Model Training
   train_models(X, y) → models, metrics, epochs

3. Model Selection
   Select best model based on sparsity

4. Parameter Extraction
   get_max_params(best_model) → max_W, max_b, max_v

5. MILP Optimization
   solve_MILP(X, y, bounds) → W, b, v_, obj_val

6. Solution Extraction
   extract_solution(W, b, v_) → numpy arrays
```

## Import Examples

### Example 1: Import entire package
```python
import relu_network
X, _, _, y, _, _ = relu_network.data(7, 5)
```

### Example 2: Import specific functions
```python
from relu_network import data, train_models
X, _, _, y, _, _ = data(7, 5)
models, metrics, epochs = train_models(X, y)
```

### Example 3: Import from submodules
```python
from relu_network.model import ReLURegNet
from relu_network.utils import count_nonzero
model = ReLURegNet(5, 7)
sparsity = count_nonzero(model)
```

## Running the Code

### Option 1: Run main script
```bash
python main.py
```

### Option 2: Run examples
```bash
python example_usage.py
```

### Option 3: Interactive Python
```python
from relu_network import *
X, _, _, y, _, _ = data(N=7, d=5)
models, metrics, epochs = train_models(X, y, num_epochs=10000)
```

## Customization Points

1. **Change data generation**: Modify `data_generation.py`
2. **Change network architecture**: Modify `model.py`
3. **Add new metrics**: Modify `utils.py`
4. **Change training algorithm**: Modify `trainer.py`
5. **Modify MILP constraints**: Modify `milp_solver.py`
6. **Change hyperparameters**: Modify `main.py` or pass to functions

# Quick Start Guide - Running Experiments

## Overview

This guide will help you run experiments with different N (number of samples) and d (number of features) values, with a 1-minute time limit per experiment. All results are automatically saved to an Excel spreadsheet.

## Files

- **`run_experiments_crossplatform.py`** - Main experiment script (works on Windows, Mac, Linux)
- **`experiment_results.xlsx`** - Output file with all results

## Setup

### 1. Install Dependencies

```bash
pip install numpy torch pandas gurobipy matplotlib openpyxl
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Set up Gurobi License

Gurobi requires a license (free for academic use):
1. Go to https://www.gurobi.com/academia/academic-program-and-licenses/
2. Register and download a license
3. Follow installation instructions

## Running Experiments

### Basic Usage

```bash
python run_experiments_crossplatform.py
```

This will:
1. Run experiments for all combinations of N and d values
2. Apply a 1-minute time limit to each MILP optimization
3. Save results to `experiment_results.xlsx` after each experiment
4. Print progress and summary statistics

### Customize N and d Values

Edit the script to change which values to test:

```python
# In run_experiments_crossplatform.py, around line 335:
N_values = [5, 7, 10, 15]     # Your N values
d_values = [3, 5, 7, 10]       # Your d values
```

### Customize Time Limit

Change the time limit for MILP optimization:

```python
time_limit_seconds = 60  # 1 minute (default)
# or
time_limit_seconds = 300  # 5 minutes
```

### Customize Training Epochs

Reduce training epochs for faster experiments:

```python
train_epochs = 50000   # Full training (default)
# or
train_epochs = 10000   # Faster for testing
```

## Understanding the Output

### Console Output

For each experiment, you'll see:
```
*** Experiment 1/9 ***
======================================================================
Running experiment: N=5, d=3
======================================================================
[1/3] Generating data...
  Data generated in 0.001s
[2/3] Training models (epochs=50000)...
  Training completed in 45.321s
  Best model: p1 with sparsity=12
[3/3] Solving MILP (time limit=60s)...
  MILP completed in 23.456s (status: optimal)
  Objective: 8.0
  Improvement: 4.0 (33.3%)
```

### Excel Spreadsheet Columns

The output spreadsheet contains:

**Experiment Info:**
- `N` - Number of samples
- `d` - Number of features
- `timestamp` - When experiment ran
- `train_epochs` - Training epochs used
- `time_limit` - MILP time limit

**Training Results (for each model: none, wd, p1, p2, p3):**
- `{model}_mse` - Mean squared error
- `{model}_sparsity` - Number of nonzero parameters
- `{model}_active_neurons` - Number of active neurons

**Best Model:**
- `best_model` - Which model had lowest sparsity
- `best_sparsity` - Sparsity of best model
- `max_W`, `max_b`, `max_v` - Maximum parameter values

**MILP Results:**
- `milp_status` - optimal, suboptimal, time_limit, or error
- `milp_time` - Time taken
- `milp_objective` - Optimal sparsity found
- `milp_nonzero_weights` - Nonzero weights in MILP solution
- `milp_nonzero_biases` - Nonzero biases in MILP solution
- `milp_active_neurons` - Active neurons in MILP solution
- `improvement` - Sparsity improvement over best trained model
- `improvement_percent` - Improvement as percentage

**Status:**
- `success` - Whether experiment completed
- `error` - Error message if failed

## Example Experiments

### Quick Test (Small Scale)
```python
N_values = [5, 7]
d_values = [3, 5]
time_limit_seconds = 30
train_epochs = 10000
```
Runs 4 experiments in ~5-10 minutes

### Medium Scale
```python
N_values = [5, 7, 10]
d_values = [3, 5, 7]
time_limit_seconds = 60
train_epochs = 50000
```
Runs 9 experiments in ~30-60 minutes

### Large Scale
```python
N_values = [5, 7, 10, 15, 20]
d_values = [3, 5, 7, 10, 15]
time_limit_seconds = 300
train_epochs = 100000
```
Runs 25 experiments in several hours

## Tips

1. **Start Small**: Test with small N and d values first
2. **Monitor Progress**: Results are saved after each experiment
3. **Time Estimates**: 
   - Data generation: <1 second
   - Training: ~1 second per 1000 epochs
   - MILP: Up to time limit (may finish early)
4. **Interrupting**: You can stop the script (Ctrl+C) and results up to that point are saved

## Troubleshooting

### "No module named 'gurobipy'"
```bash
pip install gurobipy
```
Then set up a license from Gurobi website.

### "No module named 'openpyxl'"
```bash
pip install openpyxl
```

### Excel file won't open
If Excel output fails, the script automatically saves to CSV instead:
```
experiment_results.csv
```

### MILP always times out
- Increase `time_limit_seconds`
- Or reduce N and d values
- MILP is computationally expensive for large problems

### Training takes too long
- Reduce `train_epochs` (e.g., from 50000 to 10000)
- This may reduce model quality slightly

## Analyzing Results

Open `experiment_results.xlsx` in Excel, Google Sheets, or Python:

```python
import pandas as pd
df = pd.read_excel('experiment_results.xlsx')

# Filter successful MILP runs
successful = df[df['milp_status'].str.contains('optimal')]

# Average improvement
print(f"Average improvement: {successful['improvement_percent'].mean():.2f}%")

# Plot results
import matplotlib.pyplot as plt
plt.scatter(df['N'], df['improvement_percent'])
plt.xlabel('N')
plt.ylabel('Improvement %')
plt.show()
```

## Next Steps

1. Run initial experiments with default settings
2. Analyze results in Excel
3. Adjust N, d, time limits based on findings
4. Run larger batch experiments if needed

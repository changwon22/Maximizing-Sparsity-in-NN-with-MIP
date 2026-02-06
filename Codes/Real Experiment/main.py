import sys
import os

# Add the parent directory to the path to import relu_network package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relu_network import data, train_models, solve_MILP, extract_solution, get_max_params


def main():
    
    # Training hyperparameters
    p1 = 0.4          # lp regularization parameter 1
    p2 = 0.7          # lp regularization parameter 2
    p3 = 1.0          # lp regularization parameter 3 (L1)
    gamma = 0.01      # learning rate
    num_epochs = 100000  # number of training epochs
    lambda_reg = 0.005   # regularization weight
    eps = 1e-6           # threshold for counting nonzeros
    incl_bias_sparsity = True  # include biases in sparsity measure
    save_figs = True     # save plots as .png
    
    # Data parameters
    N = 7  # number of samples
    d = 5  # dimension of features
    
    print("=" * 80)
    print("Sparse ReLU Network Training and Optimization")
    print("=" * 80)
    
    # ==============================================================================
    # Step 1: Generate Data
    # ==============================================================================
    
    print("\n[Step 1] Generating data...")
    X1, X2, X3, y1, y2, y3 = data(N, d)
    print(f"Generated data with N={N} samples, d={d} features")
    print(f"  X1 shape: {X1.shape}, y1 shape: {y1.shape}")
    
    # ==============================================================================
    # Step 2: Train Models with Continuous Optimization
    # ==============================================================================
    
    print("\n[Step 2] Training models with continuous optimization...")
    print(f"Training parameters:")
    print(f"  - Learning rate: {gamma}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Regularization weight: {lambda_reg}")
    print(f"  - p values: {p1}, {p2}, {p3}")
    print()
    
    models, metrics, epochs = train_models(
        X1, y1,
        p1=p1, p2=p2, p3=p3,
        gamma=gamma,
        num_epochs=num_epochs,
        lambda_reg=lambda_reg,
        eps=eps,
        incl_bias_sparsity=incl_bias_sparsity,
        save_figs=save_figs,
        verbose=True
    )
    
    # ==============================================================================
    # Step 3: Select Best Model
    # ==============================================================================
    
    print("\n[Step 3] Selecting best model based on sparsity...")
    
    # Find model with lowest sparsity at final epoch
    final_sparsity = {
        name: metrics[name]['spars'][-1] 
        for name in ['none', 'wd', 'p1', 'p2', 'p3']
    }
    
    best_model_name = min(final_sparsity, key=final_sparsity.get)
    best_model = models[best_model_name]
    
    print(f"\nFinal sparsity for each model:")
    for name, spars in final_sparsity.items():
        marker = " <-- BEST" if name == best_model_name else ""
        print(f"  {name:6s}: {spars:3d} nonzero parameters{marker}")
    
    # Get maximum parameter values
    max_W, max_b, max_v = get_max_params(best_model)
    
    print(f"\nBest model ({best_model_name}) parameter bounds:")
    print(f"  max |W| = {max_W:.6f}")
    print(f"  max |b| = {max_b:.6f}")
    print(f"  max |v| = {max_v:.6f}")
    
    # ==============================================================================
    # Step 4: Solve MILP for Optimal Sparse Solution
    # ==============================================================================
    
    print("\n[Step 4] Solving MILP for optimal sparse solution...")
    
    # Set bounds with epsilon margin
    epsilon = 1.0
    w_Bound = max_W + epsilon
    b_Bound = max_b + epsilon
    
    print(f"MILP bounds:")
    print(f"  w_Bound = {w_Bound:.6f}")
    print(f"  b_Bound = {b_Bound:.6f}")
    print()
    
    W, b, v_, obj_val = solve_MILP(X1, y1.flatten(), w_Bound, b_Bound, output_flag=1)
    
    if obj_val is not None:
        print(f"\nMILP optimization successful!")
        print(f"Optimal sparsity: {obj_val:.0f} nonzero parameters")
        
        # Extract solution
        K = N
        W_sol, b_sol, v_sol = extract_solution(W, b, v_, d, K)
        
        print(f"\nSolution statistics:")
        print(f"  Nonzero weights: {(abs(W_sol) > 1e-6).sum()}")
        print(f"  Nonzero biases: {(abs(b_sol) > 1e-6).sum()}")
        print(f"  Active output neurons: {(v_sol > 0.5).sum()}")
    else:
        print("\nMILP optimization failed or was interrupted.")
    
    # ==============================================================================
    # Summary
    # ==============================================================================
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Best continuous model: {best_model_name} with {final_sparsity[best_model_name]} nonzero params")
    if obj_val is not None:
        print(f"MILP optimal solution: {obj_val:.0f} nonzero params")
        improvement = final_sparsity[best_model_name] - obj_val
        print(f"Improvement: {improvement:.0f} parameters ({improvement/final_sparsity[best_model_name]*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()

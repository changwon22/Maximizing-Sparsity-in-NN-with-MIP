"""
example_usage.py
Examples of how to use the relu_network package modules
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def example_1_basic_training():
    """Example 1: Basic model training"""
    print("=" * 60)
    print("Example 1: Basic Model Training")
    print("=" * 60)
    
    from relu_network import data, train_models
    
    # Generate small dataset
    X, _, _, y, _, _ = data(N=5, d=3)
    
    # Train models with fewer epochs for quick demo
    models, metrics, epochs = train_models(
        X, y,
        num_epochs=10000,
        verbose=False  # Suppress detailed output
    )
    
    # Print final results
    print("\nFinal Results:")
    for name in ['none', 'wd', 'p1', 'p2', 'p3']:
        mse = metrics[name]['mse'][-1]
        spars = metrics[name]['spars'][-1]
        act = metrics[name]['act'][-1]
        print(f"  {name:6s}: MSE={mse:.2e}, Sparsity={spars:2d}, Active={act}")
    
    return models


def example_2_custom_model():
    """Example 2: Create and use custom model"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Model Creation")
    print("=" * 60)
    
    import torch
    from relu_network import ReLURegNet, count_nonzero, lp_path_norm
    
    # Create a custom model
    model = ReLURegNet(input_dim=4, hidden_dim=6)
    
    # Create dummy input
    x = torch.randn(10, 4)
    
    # Forward pass
    output, pre_activation = model(x)
    
    print(f"\nModel created with:")
    print(f"  Input dimension: 4")
    print(f"  Hidden dimension: 6")
    print(f"  Output shape: {output.shape}")
    
    # Analyze sparsity
    nonzeros = count_nonzero(model)
    l1_norm = lp_path_norm(model, p=1)
    
    print(f"\nModel statistics:")
    print(f"  Nonzero parameters: {nonzeros}")
    print(f"  L1 path norm: {l1_norm:.4f}")
    
    return model


def example_3_data_generation():
    """Example 3: Explore different data types"""
    print("\n" + "=" * 60)
    print("Example 3: Data Generation")
    print("=" * 60)
    
    from relu_network import data
    import numpy as np
    
    # Generate different datasets
    X1, X2, X3, y1, y2, y3 = data(N=8, d=4)
    
    print("\nDataset 1 (Random):")
    print(f"  X1 shape: {X1.shape}, range: [{X1.min():.3f}, {X1.max():.3f}]")
    print(f"  y1 shape: {y1.shape}, range: [{y1.min():.3f}, {y1.max():.3f}]")
    
    print("\nDataset 2 (Functional ReLU):")
    print(f"  X2 shape: {X2.shape}, range: [{X2.min():.3f}, {X2.max():.3f}]")
    print(f"  y2 shape: {y2.shape}, mean: {y2.mean():.3f}, std: {y2.std():.3f}")
    
    print("\nDataset 3 (Low-rank):")
    print(f"  X3 shape: {X3.shape}, range: [{X3.min():.3f}, {X3.max():.3f}]")
    print(f"  y3 shape: {y3.shape}, mean: {y3.mean():.3f}, std: {y3.std():.3f}")


def example_4_milp_small():
    """Example 4: Small MILP optimization"""
    print("\n" + "=" * 60)
    print("Example 4: Small MILP Optimization")
    print("=" * 60)
    
    from relu_network import data, solve_MILP, extract_solution
    
    # Generate small dataset (MILP is computationally expensive)
    X, _, _, y, _, _ = data(N=4, d=3)
    
    print(f"\nSolving MILP for N={X.shape[0]}, d={X.shape[1]}...")
    print("(This may take a while...)\n")
    
    # Solve MILP with reasonable bounds
    try:
        W, b, v_, obj_val = solve_MILP(
            X, y.flatten(),
            w_Bound=2.0,
            b_Bound=1.0,
            output_flag=0  # Suppress Gurobi output
        )
        
        if obj_val is not None:
            # Extract solution
            W_sol, b_sol, v_sol = extract_solution(W, b, v_, d=3, K=4)
            
            print(f"\nOptimization Results:")
            print(f"  Optimal sparsity: {obj_val:.0f}")
            print(f"  Nonzero weights: {(abs(W_sol) > 1e-6).sum()}")
            print(f"  Nonzero biases: {(abs(b_sol) > 1e-6).sum()}")
            print(f"  Active neurons: {(v_sol > 0.5).sum()}")
    except Exception as e:
        print(f"MILP optimization failed: {e}")
        print("(This might happen if Gurobi is not properly installed)")


def example_5_module_composition():
    """Example 5: Compose multiple modules"""
    print("\n" + "=" * 60)
    print("Example 5: Module Composition")
    print("=" * 60)
    
    from relu_network import data, train_models, get_max_params, count_active_neurons
    
    # Generate data
    X, _, _, y, _, _ = data(N=6, d=4)
    
    # Train models
    print("\nTraining models...")
    models, metrics, epochs = train_models(
        X, y,
        num_epochs=5000,
        verbose=False
    )
    
    # Analyze all models
    print("\nModel Analysis:")
    for name, model in models.items():
        max_W, max_b, max_v = get_max_params(model)
        active = count_active_neurons(model)
        
        print(f"\n  {name.upper()}:")
        print(f"    Max |W|: {max_W:.4f}")
        print(f"    Max |b|: {max_b:.4f}")
        print(f"    Max |v|: {max_v:.4f}")
        print(f"    Active neurons: {active}")


def main():
    """Run all examples"""
    print("\n")
    print("#" * 60)
    print("# ReLU Network Package - Usage Examples")
    print("#" * 60)
    
    # Run examples
    example_1_basic_training()
    example_2_custom_model()
    example_3_data_generation()
    
    # MILP example (commented out by default as it's slow)
    # Uncomment to run:
    # example_4_milp_small()
    
    example_5_module_composition()
    
    print("\n" + "#" * 60)
    print("# All examples completed!")
    print("#" * 60)
    print()


if __name__ == "__main__":
    main()

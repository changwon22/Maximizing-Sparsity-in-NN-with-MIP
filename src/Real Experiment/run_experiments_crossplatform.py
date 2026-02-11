"""
run_experiments_crossplatform.py
Cross-platform version: Run experiments with different N and d values
Works on Windows, Mac, and Linux
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relu_network import data, train_models, get_max_params, solve_MILP, extract_solution


def run_single_experiment(N, d, time_limit_seconds=60, train_epochs=50000):
    """
    Run a single experiment with given N and d
    
    Args:
        N: Number of samples
        d: Number of features
        time_limit_seconds: Time limit for MILP in seconds
        train_epochs: Number of training epochs
        
    Returns:
        results: Dictionary containing experiment results
    """
    results = {
        'N': N,
        'd': d,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_epochs': train_epochs,
        'time_limit': time_limit_seconds,
    }
    
    try:
        print(f"\n{'='*70}")
        print(f"Running experiment: N={N}, d={d}")
        print(f"{'='*70}")
        
        # Step 1: Generate data
        print(f"[1/3] Generating data...")
        start_time = time.time()
        X1, X2, X3, y1, y2, y3 = data(N, d)
        data_time = time.time() - start_time
        results['data_gen_time'] = data_time
        print(f"  Data generated in {data_time:.3f}s")
        
        # Step 2: Train models
        print(f"[2/3] Training models (epochs={train_epochs})...")
        start_time = time.time()
        models, metrics, epochs = train_models(
            X1, y1,
            p1=0.4,
            p2=0.7,
            p3=1.0,
            gamma=0.01,
            num_epochs=train_epochs,
            lambda_reg=0.005,
            eps=1e-6,
            incl_bias_sparsity=True,
            save_figs=False,
            verbose=False
        )
        train_time = time.time() - start_time
        results['train_time'] = train_time
        print(f"  Training completed in {train_time:.3f}s")
        
        # Record results for each model
        for name in ['none', 'wd', 'p1', 'p2', 'p3']:
            results[f'{name}_mse'] = metrics[name]['mse'][-1]
            results[f'{name}_sparsity'] = metrics[name]['spars'][-1]
            results[f'{name}_active_neurons'] = metrics[name]['act'][-1]
        
        # Find best model
        final_sparsity = {
            name: metrics[name]['spars'][-1] 
            for name in ['none', 'wd', 'p1', 'p2', 'p3']
        }
        best_model_name = min(final_sparsity, key=final_sparsity.get)
        best_model = models[best_model_name]
        
        results['best_model'] = best_model_name
        results['best_sparsity'] = final_sparsity[best_model_name]
        
        # Get parameter bounds
        max_W, max_b, max_v = get_max_params(best_model)
        results['max_W'] = max_W
        results['max_b'] = max_b
        results['max_v'] = max_v
        
        print(f"  Best model: {best_model_name} with sparsity={final_sparsity[best_model_name]}")
        
        # Step 3: Solve MILP with time limit
        print(f"[3/3] Solving MILP (time limit={time_limit_seconds}s)...")
        epsilon = 1.0
        w_Bound = max_W + epsilon
        b_Bound = max_b + epsilon
        
        results['w_bound'] = w_Bound
        results['b_bound'] = b_Bound
        
        start_time = time.time()
        
        try:
            W, b, v_, obj_val, milp_status = solve_MILP(
                X1, y1.flatten(),
                w_Bound,
                b_Bound,
                output_flag=0,  # Suppress output in batch mode
                time_limit=time_limit_seconds
            )
            
            milp_time = time.time() - start_time
            results['milp_time'] = milp_time
            results['milp_status'] = milp_status
            
            if obj_val is not None:
                results['milp_objective'] = obj_val
                
                # Extract solution
                K = N
                W_sol, b_sol, v_sol = extract_solution(W, b, v_, d, K)
                
                results['milp_nonzero_weights'] = int((abs(W_sol) > 1e-6).sum())
                results['milp_nonzero_biases'] = int((abs(b_sol) > 1e-6).sum())
                results['milp_active_neurons'] = int((v_sol > 0.5).sum())
                
                improvement = results['best_sparsity'] - obj_val
                results['improvement'] = improvement
                results['improvement_percent'] = (improvement / results['best_sparsity'] * 100) if results['best_sparsity'] > 0 else 0
                
                print(f"  MILP completed in {milp_time:.3f}s (status: {milp_status})")
                print(f"  Objective: {obj_val:.0f}")
                print(f"  Improvement: {improvement:.0f} ({results['improvement_percent']:.1f}%)")
            else:
                print(f"  MILP did not find solution (status: {milp_status}, time: {milp_time:.3f}s)")
                
        except Exception as e:
            milp_time = time.time() - start_time
            results['milp_time'] = milp_time
            results['milp_status'] = f"error: {str(e)[:50]}"
            print(f"  MILP error: {e}")
        
        results['success'] = True
        results['error'] = None
        
    except Exception as e:
        print(f"  Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    return results


def run_experiments(N_values, d_values, time_limit_seconds=60, train_epochs=50000, 
                   output_file='experiment_results.xlsx'):
    """
    Run experiments for all combinations of N and d values
    
    Args:
        N_values: List of N values to test
        d_values: List of d values to test
        time_limit_seconds: Time limit for MILP per experiment
        train_epochs: Number of training epochs
        output_file: Output Excel file name
    """
    all_results = []
    total_experiments = len(N_values) * len(d_values)
    experiment_num = 0
    
    print("\n" + "="*70)
    print(f"Starting batch experiments: {total_experiments} total")
    print(f"N values: {N_values}")
    print(f"d values: {d_values}")
    print(f"MILP time limit: {time_limit_seconds}s per experiment")
    print(f"Training epochs: {train_epochs}")
    print("="*70)
    
    start_time_total = time.time()
    
    for N in N_values:
        for d in d_values:
            experiment_num += 1
            print(f"\n\n*** Experiment {experiment_num}/{total_experiments} ***")
            
            results = run_single_experiment(
                N=N, 
                d=d, 
                time_limit_seconds=time_limit_seconds,
                train_epochs=train_epochs
            )
            
            all_results.append(results)
            
            # Save intermediate results after each experiment
            df = pd.DataFrame(all_results)
            try:
                df.to_excel(output_file, index=False, engine='openpyxl')
                print(f"\n  → Results saved to {output_file}")
            except Exception as e:
                # Fallback to CSV if Excel fails
                csv_file = output_file.replace('.xlsx', '.csv')
                df.to_csv(csv_file, index=False)
                print(f"\n  → Results saved to {csv_file} (Excel failed: {e})")
    
    total_time = time.time() - start_time_total
    
    # Final summary
    print("\n" + "="*70)
    print("BATCH EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Total experiments: {total_experiments}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average time per experiment: {total_time/total_experiments:.2f}s")
    print(f"\nResults saved to: {output_file}")
    print("="*70)
    
    # Print summary statistics
    df = pd.DataFrame(all_results)
    print("\nSummary Statistics:")
    print(f"  Successful experiments: {df['success'].sum()}/{len(df)}")
    
    if 'milp_status' in df.columns:
        milp_success = df['milp_status'].str.contains('optimal|suboptimal', na=False).sum()
        milp_timeout = df['milp_status'].str.contains('time_limit', na=False).sum()
        print(f"  MILP optimal/suboptimal: {milp_success}/{len(df)}")
        print(f"  MILP timeouts: {milp_timeout}/{len(df)}")
    
    if 'improvement_percent' in df.columns:
        successful_improvements = df[df['milp_status'].str.contains('optimal|suboptimal', na=False)]['improvement_percent']
        if len(successful_improvements) > 0:
            print(f"  Average improvement: {successful_improvements.mean():.2f}%")
            print(f"  Max improvement: {successful_improvements.max():.2f}%")
    
    return df


if __name__ == "__main__":
    # Define the N and d values to test
    # Start with small values to test quickly
    N_values = [5, 7, 10]        # Number of samples
    d_values = [3, 5, 7]          # Number of features
    
    # Configuration
    time_limit_seconds = 60       # 1 minute time limit for MILP
    train_epochs = 50000          # Training epochs (reduce for faster testing, e.g., 10000)
    output_file = 'experiment_results.xlsx'
    
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"This will run {len(N_values) * len(d_values)} experiments")
    print(f"Each experiment includes:")
    print(f"  - Data generation")
    print(f"  - Training 5 models for {train_epochs} epochs")
    print(f"  - MILP optimization (up to {time_limit_seconds}s)")
    print(f"\nEstimated time: {len(N_values) * len(d_values) * (train_epochs * 0.001 + time_limit_seconds + 10):.0f}s")
    print("="*70)
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Run experiments
    results_df = run_experiments(
        N_values=N_values,
        d_values=d_values,
        time_limit_seconds=time_limit_seconds,
        train_epochs=train_epochs,
        output_file=output_file
    )
    
    print(f"\n✓ All results saved to {output_file}")
    print(f"✓ You can open this file in Excel or any spreadsheet application")

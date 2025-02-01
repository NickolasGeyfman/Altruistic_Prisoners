#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alphasweep3continued.py

This module performs a parameter sweep over different values of the altruism parameter (α)
and the discount factor (γ) for agents playing the repeated Prisoner's Dilemma. For each
(α, γ) pair, multiple seeds are run in parallel. This version resumes the sweep by pre-filling
the results arrays with data already computed (from a previous run) and computing the missing
parameter combinations. The final cooperation rates are then aggregated and visualized using
side-by-side heatmaps for mean cooperation and its standard deviation.

Updated parameters:
  - Seeds: 10
  - Number of agents: 30
  - α: from 0.0 to 0.6 in 0.05 increments
  - γ: from 0.0 to 0.90 in 0.05 increments, then 0.95 and 0.99
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Import the testing framework from three_steptester.py.
import three_steptester  # Ensure three_steptester.py is in the same directory.

def main():
    # ===== Hyperparameters for the sweep =====
    alpha_values = np.arange(0.0, 0.6 + 0.05, 0.05).tolist()  # 0.00, 0.05, 0.10, ..., 0.60
    discount_values = list(np.arange(0.0, 0.95, 0.05)) + [0.95, 0.99]  # 0.00, 0.05, ... , 0.90, 0.95, 0.99
    num_seeds = 10       # Number of seeds per (α, γ) pair
    num_agents = 30      # Number of agents
    alpha_std = 0.1      # Standard deviation for α (set to 0 if α == 0)
    learning_rate = 0.1
    initial_epsilon = 0.1
    min_epsilon = 0.01
    decay_rate = 0.999
    train_episodes = 2000
    eval_episodes = 10
    prob_continue = 0.99
    mode = "ring"        # Pairing mode: "random" or "ring"
    min_rounds = 10      # Minimum rounds per episode
    # ==========================================
    
    n_alpha = len(alpha_values)
    n_gamma = len(discount_values)
    
    # Initialize results arrays with NaN so we can tell what hasn't been computed.
    results_mean = np.full((n_alpha, n_gamma), np.nan)
    results_std = np.full((n_alpha, n_gamma), np.nan)
    
    # --- Pre-fill known data from previous runs ---
    # Row 0 corresponds to α = 0.00.
    # (Fill these arrays with your already-computed values.)
    mean_row0 = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
                          0.003, 0.003, 0.003, 0.023, 0.041, 0.124, 0.205, 0.321, 
                          0.524, 0.569, 0.676, 0.795, 0.915])
    std_row0 = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
                         0.010, 0.010, 0.010, 0.018, 0.031, 0.067, 0.064, 0.050, 
                         0.022, 0.048, 0.062, 0.043, 0.024])
    results_mean[0, :] = mean_row0
    results_std[0, :] = std_row0

    # Row 1 corresponds to α = 0.05.
    mean_row1 = np.array([0.058, 0.060, 0.060, 0.058, 0.062, 0.060, 0.066, 0.075, 
                          0.079, 0.105, 0.145, 0.156, 0.244, 0.332, 0.450, 0.542, 
                          0.624, 0.698, 0.739, 0.813, 0.921])
    std_row1 = np.array([0.034, 0.035, 0.033, 0.034, 0.033, 0.034, 0.034, 0.034, 
                         0.038, 0.058, 0.069, 0.048, 0.058, 0.058, 0.043, 0.055, 
                         0.035, 0.042, 0.025, 0.025, 0.018])
    results_mean[1, :] = mean_row1
    results_std[1, :] = std_row1

    # Row 2 corresponds to α = 0.10.
    # We have computed for discount indices 0 through 4.
    mean_row2_partial = np.array([0.139, 0.139, 0.140, 0.143, 0.145])
    std_row2_partial = np.array([0.044, 0.044, 0.044, 0.043, 0.045])
    results_mean[2, 0:5] = mean_row2_partial
    results_std[2, 0:5] = std_row2_partial

    # --- End Pre-fill ---

    # Now, resume the simulation for any cell that is still NaN.
    for i, alpha in enumerate(alpha_values):
        local_alpha_std = 0.0 if alpha == 0.0 else alpha_std
        for j, gamma in enumerate(discount_values):
            if not np.isnan(results_mean[i, j]):
                # Already computed; skip this cell.
                continue
            
            # Record start time for this (α, γ) pair.
            start_time = time.time()
            
            # Build the argument list for multiprocessing.
            arglist = []
            for seed in range(num_seeds):
                kwargs = {
                    "seed": seed,
                    "num_agents": num_agents,
                    "alpha_mean": alpha,
                    "alpha_std": local_alpha_std,
                    "discount": gamma,
                    "learning_rate": learning_rate,
                    "initial_epsilon": initial_epsilon,
                    "min_epsilon": min_epsilon,
                    "decay_rate": decay_rate,
                    "train_episodes": train_episodes,
                    "eval_episodes": eval_episodes,
                    "prob_continue": prob_continue,
                    "mode": mode,
                    "min_rounds": min_rounds
                }
                arglist.append((seed, kwargs))
            
            # Run the experiments in parallel.
            with mp.Pool(processes=num_seeds) as pool:
                results = pool.map(three_steptester.worker_single, arglist)
            
            elapsed_time = time.time() - start_time
            
            final_coops = [result[0] for result in results]
            mean_coop = np.mean(final_coops)
            std_coop = np.std(final_coops)
            
            results_mean[i, j] = mean_coop
            results_std[i, j] = std_coop
            
            print(f"Alpha = {alpha:.2f} (std = {local_alpha_std:.2f}), Discount = {gamma:.2f}, "
                  f"Seeds = {num_seeds} => Mean coop = {mean_coop:.3f}, Std = {std_coop:.3f}, "
                  f"Time taken = {elapsed_time:.2f} sec")
    
    # Print the aggregated results.
    print("\n=== Results Matrix (Mean Cooperation) ===")
    print(results_mean)
    print("\n=== Results Matrix (Standard Deviation) ===")
    print(results_std)
    
    # Create side-by-side heatmaps.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap for Mean Final Cooperation.
    ax1 = axes[0]
    im1 = ax1.imshow(results_mean, origin='upper', cmap='viridis', aspect='auto')
    ax1.set_title("Mean Final Cooperation")
    ax1.set_xlabel("Discount (γ)")
    ax1.set_ylabel("Alpha")
    ax1.set_xticks(range(n_gamma))
    ax1.set_yticks(range(n_alpha))
    ax1.set_xticklabels([f"{d:.2f}" for d in discount_values])
    ax1.set_yticklabels([f"{a:.2f}" for a in alpha_values])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Heatmap for Standard Deviation of Final Cooperation.
    ax2 = axes[1]
    im2 = ax2.imshow(results_std, origin='upper', cmap='plasma', aspect='auto')
    ax2.set_title("Std of Final Cooperation")
    ax2.set_xlabel("Discount (γ)")
    ax2.set_ylabel("Alpha")
    ax2.set_xticks(range(n_gamma))
    ax2.set_yticks(range(n_alpha))
    ax2.set_xticklabels([f"{d:.2f}" for d in discount_values])
    ax2.set_yticklabels([f"{a:.2f}" for a in alpha_values])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overall title with hyperparameter details.
    hyperparam_info = (
        f"num_agents={num_agents}, train_episodes={train_episodes}, seeds={num_seeds}\n"
        f"prob_continue={prob_continue}, min_rounds={min_rounds}, mode={mode}"
    )
    plt.suptitle(f"Alpha vs. Discount Sweep (Continued)\n{hyperparam_info}", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alphasweep3.py

This module performs a parameter sweep over different values of the altruism parameter (alpha)
and the discount factor (gamma) for agents playing the repeated Prisoner's Dilemma.
For each (alpha, gamma) pair, multiple seeds are run in parallel.
The final cooperation rates are then aggregated and visualized using side-by-side heatmaps
for mean cooperation and its standard deviation.

Updated parameters:
  - Seeds: 10
  - Number of agents: 30
  - Alpha: from 0.0 to 0.6 in 0.05 increments
  - Discount: from 0.0 to 0.90 in 0.05 increments, then 0.95 and 0.99
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Import the testing framework from three_steptester.py.
import three_steptester  # Ensure three_steptester.py is in the same directory.

def main():
    # ===== Hyperparameters for the sweep =====
    # Alpha values: 0.0, 0.05, 0.10, ..., 0.60
    alpha_values = np.arange(0.0, 0.6 + 0.05, 0.05).tolist()
    
    # Discount values: from 0.0 to 0.90 in 0.05 steps, then add 0.95 and 0.99
    discount_values = list(np.arange(0.0, 0.95, 0.05)) + [0.95, 0.99]
    
    num_seeds = 10       # Number of seeds per (alpha, gamma) pair
    num_agents = 30      # Number of agents
    alpha_std = 0.1      # Standard deviation for alpha (set to 0 if alpha == 0)
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
    results_mean = np.zeros((n_alpha, n_gamma))
    results_std = np.zeros((n_alpha, n_gamma))
    
    # Loop over each combination of alpha and discount values.
    for i, alpha in enumerate(alpha_values):
        # Force standard deviation to 0 if alpha is 0.0.
        local_alpha_std = 0.0 if alpha == 0.0 else alpha_std
        
        for j, gamma in enumerate(discount_values):
            # Record start time for this parameter combination.
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
            
            # Run the experiments in parallel using a pool with num_seeds processes.
            with mp.Pool(processes=num_seeds) as pool:
                results = pool.map(three_steptester.worker_single, arglist)
            
            # Calculate the time elapsed for this (alpha, gamma) pair.
            elapsed_time = time.time() - start_time
            
            # Gather the final cooperation rates.
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
    ax1.set_xlabel("Discount (gamma)")
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
    ax2.set_xlabel("Discount (gamma)")
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
    plt.suptitle(f"Alpha vs. Discount Sweep\n{hyperparam_info}", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
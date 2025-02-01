#!/usr/bin/env python3
"""
alphasweep3amarel.py

This script performs a parameter sweep over various values of the altruism factor (alpha)
and discount factor (gamma) for our repeated Prisoner's Dilemma experiment on Amarel.
It is designed to run in parallel using Python's multiprocessing module. In this version,
we run 10 seeds per (alpha, gamma) pair (instead of 50) to speed up the overall run time.

The script automatically uses the number of CPUs allocated by SLURM via the SLURM_CPUS_PER_TASK
environment variable, which allows it to take full advantage of the resources on nodes like sirius3.


"""

import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

# Import experiment helper functions (ensure three_steptester.py is in the same directory)
import three_steptester

def main():
    # === Hyperparameters for the parameter sweep ===
    alpha_values    = [0.0, 0.2, 0.4, 0.6, 0.8]      # Altruism factors
    discount_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]  # Discount factors (gamma)
    num_seeds       = 10                          # Use 10 seeds per (alpha, gamma) pair for faster runs.
    num_agents      = 10
    alpha_std       = 0.1                         # Standard deviation for alpha (set to 0 if alpha==0)
    learning_rate   = 0.1
    initial_epsilon = 0.1
    min_epsilon     = 0.01
    decay_rate      = 0.999
    train_episodes  = 2000
    eval_episodes   = 10
    prob_continue   = 0.99
    mode            = "ring"
    min_rounds      = 10

    # === Determine the number of processes based on allocated CPUs ===
    # SLURM sets the environment variable SLURM_CPUS_PER_TASK if you request CPUs in your job script.
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus:
        num_processes = int(cpus)
    else:
        num_processes = mp.cpu_count()
    print(f"Using {num_processes} parallel processes.")

    # === Prepare arrays to store the aggregated results ===
    n_alpha = len(alpha_values)
    n_gamma = len(discount_values)
    results_mean = np.zeros((n_alpha, n_gamma))
    results_std  = np.zeros((n_alpha, n_gamma))

    # === Loop over each (alpha, gamma) pair ===
    for i, alpha in enumerate(alpha_values):
        local_alpha_std = 0.0 if alpha == 0.0 else alpha_std
        for j, gamma in enumerate(discount_values):
            # Build an argument list for each seed run.
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
                    "min_rounds": min_rounds,
                }
                arglist.append((seed, kwargs))
            
            # Use multiprocessing to run all seeds in parallel.
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(three_steptester.worker_single, arglist)
            
            # Extract the final cooperation rates and compute the mean and standard deviation.
            final_coops = [result[0] for result in results]
            mean_coop = np.mean(final_coops)
            std_coop  = np.std(final_coops)
            results_mean[i, j] = mean_coop
            results_std[i, j]  = std_coop

            print(f"Alpha = {alpha} (std = {local_alpha_std}), Gamma = {gamma} => Mean coop = {mean_coop:.3f}, Std = {std_coop:.3f}")

    # === Plot the results as heatmaps ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(results_mean, origin='upper', cmap='viridis', aspect='auto')
    ax1.set_title("Mean Final Cooperation")
    ax1.set_xlabel("Discount (gamma)")
    ax1.set_ylabel("Alpha")
    ax1.set_xticks(range(n_gamma))
    ax1.set_yticks(range(n_alpha))
    ax1.set_xticklabels(discount_values)
    ax1.set_yticklabels(alpha_values)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(results_std, origin='upper', cmap='plasma', aspect='auto')
    ax2.set_title("Std of Final Cooperation")
    ax2.set_xlabel("Discount (gamma)")
    ax2.set_ylabel("Alpha")
    ax2.set_xticks(range(n_gamma))
    ax2.set_yticks(range(n_alpha))
    ax2.set_xticklabels(discount_values)
    ax2.set_yticklabels(alpha_values)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    hyperparam_info = (
        f"num_agents={num_agents}, train_episodes={train_episodes}, seeds={num_seeds}\n"
        f"prob_continue={prob_continue}, min_rounds={min_rounds}, mode={mode}"
    )
    plt.suptitle(f"Alpha vs. Discount Sweep on Amarel\n{hyperparam_info}", fontsize=12)
    plt.tight_layout()
    plt.savefig("alphasweep3amarel_results.png")
    plt.show()

if __name__ == "__main__":
    main()
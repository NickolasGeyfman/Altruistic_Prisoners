#!/usr/bin/env python3
import random
import os  # NEW: to read environment variable
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from agent import Agent, clamp, run_round
from agent import C, D

# Import new episode routines
from agent import run_episode_stable_pairs, run_episode_ring

num_agents = 200
mean_alpha = 0.3
std_alpha  = 0.1
learning_rate = 0.1
discount    = 0.90
epsilon     = 0.1
rounds_per_episode = 200
train_episodes    = 300
eval_episodes     = 50

def run_experiment(seed, store_training=False, mode="stable_pairs"):
    random.seed(seed)
    np.random.seed(seed)
    
    population = []
    for _ in range(num_agents):
        alpha_i = clamp(random.gauss(mean_alpha, std_alpha), 0, 1)
        agent = Agent(alpha=alpha_i,
                      learning_rate=learning_rate,
                      discount=discount,
                      epsilon=epsilon)
        population.append(agent)

    if mode == "stable_pairs":
        episode_fn = run_episode_stable_pairs
    elif mode == "ring":
        episode_fn = run_episode_ring
    else:
        raise ValueError("Unknown mode for run_experiment")

    # TRAIN
    train_coops = [] if store_training else None
    for _ in range(train_episodes):
        c = episode_fn(population, rounds_per_episode, update_Q_phase=True)
        if store_training:
            train_coops.append(c)

    # EVAL
    eval_coops = []
    for _ in range(eval_episodes):
        c = episode_fn(population, rounds_per_episode, update_Q_phase=False)
        eval_coops.append(c)

    final_coop = np.mean(eval_coops)
    return final_coop, train_coops

# Worker functions
def worker_final_coop_only(args):
    seed, mode = args
    final_coop, _ = run_experiment(seed, store_training=False, mode=mode)
    return final_coop

def worker_with_training_curve(args):
    seed, mode = args
    final_coop, train_curve = run_experiment(seed, store_training=True, mode=mode)
    return (seed, final_coop, train_curve)

def run_n_seeds(n=20, mode="stable_pairs"):
    """
    This function spawns multiple processes in parallel to run seeds.
    We'll dynamically set pool size from SLURM_CPUS_PER_TASK.
    """
    seeds = list(range(n))

    # NEW: read environment variable (default to 5 if not set)
    cpus_str = os.environ.get("SLURM_CPUS_PER_TASK", "5")
    pool_size = int(cpus_str)
    print(f"Using mp.Pool with {pool_size} processes (SLURM_CPUS_PER_TASK={cpus_str})")

    # Use a Pool with 'pool_size'
    with mp.Pool(processes=pool_size) as pool:
        final_coops = pool.map(worker_final_coop_only, [(s, mode) for s in seeds])

    mean_coop = np.mean(final_coops)
    std_coop  = np.std(final_coops)

    print(f"===== {n} Seeds => final eval coop (mode={mode}) =====")
    print(f"Parameters: num_agents={num_agents}, mean_alpha={mean_alpha}, discount={discount}")
    print("Final coops:", final_coops)
    print(f"Mean = {mean_coop:.3f}, Std = {std_coop:.3f}\n")

def run_5_seeds_plot(mode="stable_pairs"):
    seeds = [0,1,2,3,4]

    # same logic for parallel pool
    cpus_str = os.environ.get("SLURM_CPUS_PER_TASK", "5")
    pool_size = int(cpus_str)
    print(f"Using mp.Pool with {pool_size} processes (SLURM_CPUS_PER_TASK={cpus_str})")

    with mp.Pool(processes=pool_size) as pool:
        results = pool.map(worker_with_training_curve, [(s, mode) for s in seeds])

    print(f"===== 5 Seeds => training lines & final coop (mode={mode}) =====")
    plt.figure(figsize=(8,5))
    for (seed, final_coop, curve) in results:
        print(f"Seed={seed}, final_coop={final_coop:.3f}")
        plt.plot(curve, label=f"seed={seed}")

    plt.xlabel("Training Episode")
    plt.ylabel("Cooperation Rate")
    plt.title(f"Training Curves (5 seeds) - {mode}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():

    print("Running ring mode...")
    run_n_seeds(10, mode="ring")
    run_5_seeds_plot(mode="ring")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import random
import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# We assume you have these definitions in agent.py or similarly:
from agent import Agent, clamp, run_round, C, D  # C=0, D=1
# Or define them here if you want.

########################
# Episode Routines
########################

def run_episode_stable_pairs(population, rounds_per_episode=50, update_Q_phase=True):
    """
    Each agent is paired with exactly one partner for the entire 'rounds_per_episode'.
    Pair them up: (0,1), (2,3), etc. Then in each round, each pair plays one round of PD.
    Return the fraction of cooperation in this episode.
    """
    # Reset each agent's state
    for agent in population:
        agent.current_state = 0

    # Shuffle and form stable pairs
    random.shuffle(population)
    pairs = []
    num_agents = len(population)
    for i in range(0, num_agents, 2):
        if i+1 < num_agents:
            pairs.append((population[i], population[i+1]))

    cooperation_count = 0
    total_actions = 0

    for _ in range(rounds_per_episode):
        for (agentA, agentB) in pairs:
            actionA, actionB = run_round(agentA, agentB, update_Q_phase)
            if actionA == C:
                cooperation_count += 1
            if actionB == C:
                cooperation_count += 1
            total_actions += 2

    coop_rate = cooperation_count / total_actions if total_actions else 0.0
    return coop_rate

def run_episode_ring(population, rounds_per_episode=50, update_Q_phase=True):
    """
    Place agents in a ring. Each round, each agent i plays PD with neighbor (i+1) mod N.
    Return cooperation fraction.
    """
    for agent in population:
        agent.current_state = 0

    num_agents = len(population)
    cooperation_count = 0
    total_actions = 0

    for _ in range(rounds_per_episode):
        # For each agent i, the neighbor j = (i+1)%N
        for i in range(num_agents):
            j = (i + 1) % num_agents
            action_i, action_j = run_round(population[i], population[j], update_Q_phase)
            if action_i == C:
                cooperation_count += 1
            if action_j == C:
                cooperation_count += 1
            total_actions += 2

    return cooperation_count / total_actions if total_actions else 0.0

########################
# Equilibrium Check
########################

def check_approx_equilibrium(population, epsilon=0.01):
    """
    Checks if, for each agent, for each state, there's no large difference between
    best Q-value and the second-best Q-value. If best - second_best > epsilon, agent
    would deviate if not playing the best action. If we never find such a gap,
    we say we have approximate equilibrium.
    """
    for agent in population:
        Q = agent.Q  # shape [4,2]
        # states: 0..3, actions: 0..1

        for s in range(4):
            sorted_Q = np.sort(Q[s])  # ascending
            best = sorted_Q[-1]       # largest
            second_best = sorted_Q[-2]
            if best - second_best > epsilon:
                return False
    return True

########################
# Main Q-Learning Logic
########################

# Global Hyperparams for demonstration
num_agents        = 200
mean_alpha        = 0.3
std_alpha         = 0.1
learning_rate     = 0.1
discount          = 0.90
epsilon           = 0.1
rounds_per_episode= 200
train_episodes    = 300
eval_episodes     = 50

def run_experiment(seed, store_training=False, mode="stable_pairs"):
    random.seed(seed)
    np.random.seed(seed)

    # Create population
    population = []
    for _ in range(num_agents):
        alpha_i = clamp(random.gauss(mean_alpha, std_alpha), 0, 1)
        agent = Agent(alpha=alpha_i,
                      learning_rate=learning_rate,
                      discount=discount,
                      epsilon=epsilon)
        population.append(agent)

    # Choose episode function
    if mode == "stable_pairs":
        episode_fn = run_episode_stable_pairs
    elif mode == "ring":
        episode_fn = run_episode_ring
    else:
        raise ValueError("Unknown mode for run_experiment")

    train_coops = [] if store_training else None

    eq_epsilon = 0.01  # threshold for "approx eq"

    for i in range(train_episodes):
        c = episode_fn(population, rounds_per_episode, update_Q_phase=True)
        if store_training:
            train_coops.append(c)

        # Check approximate eq after each training episode
        if check_approx_equilibrium(population, epsilon=eq_epsilon):
            print(f"==> Approx NE reached at training episode {i}, stopping early")
            break

    # Evaluate
    eval_coops = []
    for _ in range(eval_episodes):
        c = episode_fn(population, rounds_per_episode, update_Q_phase=False)
        eval_coops.append(c)

    final_coop = np.mean(eval_coops)
    return final_coop, train_coops

########################
# Example Worker/Plots
########################

def worker_with_training_curve(args):
    seed, mode = args
    final_coop, train_curve = run_experiment(seed, store_training=True, mode=mode)
    return (seed, final_coop, train_curve)

def run_10_seeds(mode="stable_pairs"):
    n = 10
    seeds = list(range(n))

    # Use some pool size or detect CPU
    with mp.Pool(processes=10) as pool:
        results = pool.map(worker_with_training_curve, [(s, mode) for s in seeds])

    # final coops + curves
    final_coops = []
    curves      = []

    for (seed, f_coop, c) in results:
        final_coops.append(f_coop)
        curves.append(c)

    mean_coop = np.mean(final_coops)
    std_coop  = np.std(final_coops)

    print(f"\nMode={mode} => final_coops: {final_coops}")
    print(f"Mean={mean_coop:.3f}, Std={std_coop:.3f}")

    # Plot 1 figure
    plt.figure()
    for i, curve in enumerate(curves):
        plt.plot(curve, label=f"Seed {i}")
    plt.xlabel("Training Episode")
    plt.ylabel("Coop Rate")
    plt.title(f"Training Curves - {mode} (10 seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{mode}_10seeds_curve.png")
    plt.show()

def main():

    print("\n===== Testing ring for 10 seeds =====")
    run_10_seeds(mode="ring")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import random
import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from agent import Agent, clamp, run_round, C

########################
# Random Pair Episode
########################

def run_episode_random(population, rounds_per_episode=50, update_Q_phase=True):
    """
    Each round:
      1. Shuffle the entire population
      2. Pair them as (0,1), (2,3), ...
      3. For each pair, run one PD round
    Returns the fraction of cooperation in this episode.
    """
    num_agents = len(population)
    # Reset each agent's state
    for agent in population:
        agent.current_state = 0

    cooperation_count = 0
    total_actions = 0

    for _ in range(rounds_per_episode):
        random.shuffle(population)
        for i in range(0, num_agents, 2):
            if i + 1 < num_agents:
                agentA = population[i]
                agentB = population[i+1]
                actionA, actionB = run_round(agentA, agentB, update_Q_phase)
                if actionA == C:
                    cooperation_count += 1
                if actionB == C:
                    cooperation_count += 1
                total_actions += 2

    return cooperation_count / total_actions if total_actions else 0.0

########################
# Ring Episode
########################

def run_episode_ring(population, rounds_per_episode=50, update_Q_phase=True):
    """
    Ring: each agent i plays with neighbor (i+1) mod N each round.
    Returns fraction of cooperation in this episode.
    """
    num_agents = len(population)
    # Reset each agent's state
    for agent in population:
        agent.current_state = 0

    cooperation_count = 0
    total_actions = 0

    for _ in range(rounds_per_episode):
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
# "Meet10" Blocked Episode
########################

def run_episode_meet10(population, rounds_per_episode=200, block_size=10, update_Q_phase=True):
    """
    Divides the episode into blocks of 'block_size' sub-rounds each.
    In each block:
      - Shuffle population and pair them as (0,1), (2,3), ...
      - Keep those pairs for 'block_size' sub-rounds
    So each agent stays with the same partner for 'block_size' sub-rounds,
    then switches to a new partner for the next block.

    Returns fraction of cooperation over the entire episode.
    """
    num_agents = len(population)
    # Reset each agent's state once at start
    for agent in population:
        agent.current_state = 0

    total_actions = 0
    cooperation_count = 0

    # Number of blocks
    num_blocks = rounds_per_episode // block_size

    for _ in range(num_blocks):
        # Re-pair them (shuffle)
        random.shuffle(population)
        pairs = []
        for i in range(0, num_agents, 2):
            if i+1 < num_agents:
                pairs.append((population[i], population[i+1]))

        # For block_size sub-rounds, keep these pairs
        for _ in range(block_size):
            for (agentA, agentB) in pairs:
                actionA, actionB = run_round(agentA, agentB, update_Q_phase)
                if actionA == C:
                    cooperation_count += 1
                if actionB == C:
                    cooperation_count += 1
                total_actions += 2

    return cooperation_count / total_actions if total_actions else 0.0

########################
# Main Experiment Logic
########################

def run_experiment_for_alpha(
    seed,
    alpha_float,
    mode="random",        # "random", "ring", or "meet10"
    num_agents=200,
    std_alpha=0.1,
    learning_rate=0.1,
    discount=0.2,
    epsilon=0.1,
    rounds_per_episode=200,
    train_episodes=200,
    eval_episodes=20
):
    """
    For a given alpha_float in [0..1], run an experiment with num_agents.
    Each agent's alpha ~ clamp(N(alpha_float, std_alpha), 0,1).

    mode: "random" => run_episode_random
          "ring"   => run_episode_ring
          "meet10" => run_episode_meet10
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create population
    population = []
    for _ in range(num_agents):
        alpha_i = clamp(random.gauss(alpha_float, std_alpha), 0, 1)
        agent = Agent(alpha=alpha_i,
                      learning_rate=learning_rate,
                      discount=discount,
                      epsilon=epsilon)
        population.append(agent)

    # Pick the episode function
    if mode == "random":
        episode_fn = run_episode_random
    elif mode == "ring":
        episode_fn = run_episode_ring
    elif mode == "meet10":
        episode_fn = run_episode_meet10
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # TRAIN
    for _ in range(train_episodes):
        episode_fn(population, rounds_per_episode, update_Q_phase=True)

    # EVAL
    eval_coops = []
    for _ in range(eval_episodes):
        c = episode_fn(population, rounds_per_episode, update_Q_phase=False)
        eval_coops.append(c)

    final_coop = np.mean(eval_coops)
    return final_coop

def worker_alpha(args):
    """
    Worker for parallel: receives (seed, alpha_float, mode, agent_count) => final_coop
    """
    seed, alpha_float, mode, agent_count = args
    final_coop = run_experiment_for_alpha(
        seed,
        alpha_float,
        mode=mode,
        num_agents=agent_count
    )
    return final_coop

def run_alpha_sweep(mode="random", agent_count=200):
    """
    Sweeps alpha in [0..1] in increments of 0.1. For each alpha => 10 seeds in parallel.
    Gathers final_coops => mean & std. 
    mode can be "random", "ring", or "meet10".
    agent_count can be 10 or 200 (etc).
    """
    alpha_values = range(0, 101, 10)  # 0..100 in steps of 10 => 0.0..1.0

    all_means = []
    all_stds  = []

    for alpha_int in alpha_values:
        alpha_float = alpha_int / 100.0
        # We'll run 10 seeds
        seeds = list(range(10))

        with mp.Pool(processes=10) as pool:
            final_coops = pool.map(
                worker_alpha,
                [(s, alpha_float, mode, agent_count) for s in seeds]
            )

        mean_coop = np.mean(final_coops)
        std_coop  = np.std(final_coops)
        all_means.append(mean_coop)
        all_stds.append(std_coop)

        print(f"\n=== mode={mode}, #agents={agent_count}, alpha={alpha_float:.2f} => final coops => {final_coops}")
        print(f"Mean = {mean_coop:.3f}, Std = {std_coop:.3f}\n")

    # Convert alpha_values to float array
    alpha_array = np.array(list(alpha_values), dtype=float) / 100.0

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # Left: alpha vs mean
    axes[0].plot(alpha_array, all_means, marker='o', color='b')
    axes[0].set_title(f"Mean Coop vs. Alpha ({agent_count} Agents, {mode})")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Mean Cooperation")

    # Right: alpha vs std
    axes[1].plot(alpha_array, all_stds, marker='o', color='r')
    axes[1].set_title(f"Std of Coop vs. Alpha ({agent_count} Agents, {mode})")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Std of Cooperation")

    plt.tight_layout()
    outname = f"alpha_sweep_{agent_count}agents_{mode}.png"
    plt.savefig(outname)
    plt.show()

def main():
    # Examples:
    # 1) Random pairing, 200 agents
    #run_alpha_sweep(mode="random", agent_count=200)

    run_alpha_sweep(mode="random", agent_count=10)

    # 2) Ring pairing, 200 agents
    #run_alpha_sweep(mode="ring", agent_count=200)

    # 3) Meet10 approach, 200 agents
   # run_alpha_sweep(mode="meet10", agent_count=200)


if __name__ == "__main__":
    main()
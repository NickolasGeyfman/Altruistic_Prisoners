#!/usr/bin/env python3
"""
three_steptester.py

This module provides a testing framework for running the repeated Prisoner's Dilemma (PD)
with agents that use 3-step memory and partner-specific Q-learning (as defined in agent3.py).

It includes two episode types:
  - Random pairing: agents are paired randomly each round.
  - Ring pairing: agents are paired in a ring (agent i with agent (i+1) mod N).

It also contains the main experiment function which trains and evaluates the agents,
and plots the training curves.
"""

import random
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# Import Agent3 and associated utilities from agent3.py.
from agent3 import Agent3, run_round, C, D

###############################################################################
# Episode Functions (Indefinite Horizon)
###############################################################################

def run_indefinite_episode_random(population, prob_continue=0.99, update_Q_phase=True, min_rounds=10):
    """
    Run an episode with random pairing of agents.
    
    The episode continues indefinitely with a probability 'prob_continue' after at least
    'min_rounds' rounds have been played. This ensures a minimum interaction period before
    termination can occur.
    
    Parameters:
        population: List of Agent3 instances.
        prob_continue: Probability of continuing after the minimum rounds.
        update_Q_phase: Whether to update Q-values (True for training, False for evaluation).
        min_rounds: Minimum number of rounds before early termination is allowed.
    
    Returns:
        The overall cooperation rate (fraction of cooperative moves).
    """
    num_agents = len(population)
    coop_count = 0
    total_moves = 0
    rounds_so_far = 0

    while True:
        # Ensure a minimum number of rounds.
        if rounds_so_far < min_rounds:
            continue_game = True
        else:
            continue_game = (random.random() < prob_continue)

        if not continue_game:
            break

        # Shuffle the population and pair agents randomly.
        random.shuffle(population)
        for i in range(0, num_agents, 2):
            if i + 1 < num_agents:
                agent_i = population[i]
                agent_j = population[i + 1]
                i_id = agent_i.agent_id
                j_id = agent_j.agent_id

                a_i, a_j = run_round(agent_i, agent_j, i_id, j_id, update_Q_phase)
                if a_i == C:
                    coop_count += 1
                if a_j == C:
                    coop_count += 1
                total_moves += 2

        rounds_so_far += 1

    return coop_count / total_moves if total_moves > 0 else 0.0

def run_indefinite_episode_ring(population, prob_continue=0.99, update_Q_phase=True, min_rounds=10):
    """
    Run an episode with ring pairing.
    
    Each agent i interacts with agent (i+1 mod N) in every round. Similar to the random
    episode, the continuation of the episode is determined probabilistically after a minimum
    number of rounds.
    
    Parameters:
        population: List of Agent3 instances.
        prob_continue: Continuation probability after min_rounds.
        update_Q_phase: Whether to update Q-values.
        min_rounds: Minimum rounds to enforce before termination is possible.
    
    Returns:
        The overall cooperation rate.
    """
    num_agents = len(population)
    coop_count = 0
    total_moves = 0
    rounds_so_far = 0

    while True:
        if rounds_so_far < min_rounds:
            continue_game = True
        else:
            continue_game = (random.random() < prob_continue)

        if not continue_game:
            break

        # Ring pairing: agent i interacts with (i+1) mod num_agents.
        for i in range(num_agents):
            j = (i + 1) % num_agents
            agent_i = population[i]
            agent_j = population[j]
            i_id = agent_i.agent_id
            j_id = agent_j.agent_id

            a_i, a_j = run_round(agent_i, agent_j, i_id, j_id, update_Q_phase)
            if a_i == C:
                coop_count += 1
            if a_j == C:
                coop_count += 1
            total_moves += 2

        rounds_so_far += 1

    return coop_count / total_moves if total_moves > 0 else 0.0

def clamp(x, minimum=0.0, maximum=1.0):
    """
    Clamp a value x so that it falls between the specified minimum and maximum.
    """
    return max(minimum, min(x, maximum))

###############################################################################
# Main Experiment Function
###############################################################################

def run_experiment_single(
    seed,
    num_agents=20,
    alpha_mean=0.0,
    alpha_std=0.0,
    discount=0.95,
    learning_rate=0.1,
    initial_epsilon=0.1,
    min_epsilon=0.01,
    decay_rate=0.999,
    train_episodes=500,
    eval_episodes=10,
    prob_continue=0.99,
    mode="random",
    min_rounds=10
):
    """
    Run a single experiment with a population of agents playing the repeated PD.
    
    The experiment consists of:
      - A training phase (where agents update their Q-values with exploration).
      - An evaluation phase (with no exploration and no Q-updates).
    
    Parameters:
        seed: Random seed for reproducibility.
        num_agents: Number of agents.
        alpha_mean: Mean altruism factor.
        alpha_std: Standard deviation of altruism.
        discount: Discount factor (gamma) for Q-learning.
        learning_rate: Learning rate.
        initial_epsilon: Starting exploration rate.
        min_epsilon: Minimum exploration rate.
        decay_rate: Rate at which epsilon decays.
        train_episodes: Number of training episodes.
        eval_episodes: Number of evaluation episodes.
        prob_continue: Probability to continue an episode after min_rounds.
        mode: Pairing mode ("random" or "ring").
        min_rounds: Minimum rounds per episode.
    
    Returns:
        final_coop: Average cooperation rate during evaluation.
        train_coops: List of cooperation rates per training episode.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Build a population of agents.
    population = []
    for i in range(num_agents):
        # Draw the altruism factor from a normal distribution.
        alpha_i = random.gauss(alpha_mean, alpha_std)
        alpha_i = clamp(alpha_i, 0.0, 1.0)
        agent = Agent3(
            agent_id=i,
            num_agents=num_agents,
            alpha=alpha_i,
            learning_rate=learning_rate,
            discount=discount,
            epsilon=initial_epsilon
        )
        population.append(agent)

    # Choose the appropriate episode function.
    if mode == "random":
        episode_fn = lambda pop, update_Q_phase=True: run_indefinite_episode_random(
            population=pop,
            prob_continue=prob_continue,
            update_Q_phase=update_Q_phase,
            min_rounds=min_rounds
        )
    elif mode == "ring":
        episode_fn = lambda pop, update_Q_phase=True: run_indefinite_episode_ring(
            population=pop,
            prob_continue=prob_continue,
            update_Q_phase=update_Q_phase,
            min_rounds=min_rounds
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_coops = []

    # Training phase.
    epsilon = initial_epsilon
    for ep in range(train_episodes):
        epsilon = max(min_epsilon, epsilon * decay_rate)
        for agent in population:
            agent.epsilon = epsilon
            agent.reset_states()  # Reset memory at the start of each episode.
        coop_rate = episode_fn(population, update_Q_phase=True)
        train_coops.append(coop_rate)

    # Evaluation phase (no exploration, no Q-updates).
    for agent in population:
        agent.epsilon = 0.0
        agent.reset_states()

    eval_coops = []
    for _ in range(eval_episodes):
        coop_rate = episode_fn(population, update_Q_phase=False)
        eval_coops.append(coop_rate)

    final_coop = np.mean(eval_coops)
    return final_coop, train_coops

def worker_single(args):
    """
    Multiprocessing worker that unpacks the arguments and runs a single experiment.
    
    Parameters:
        args: A tuple (seed, kwargs) where kwargs is a dictionary of parameters.
    
    Returns:
        A tuple (final_coop, train_curve) from run_experiment_single.
    """
    (seed, kwargs) = args
    final_coop, train_curve = run_experiment_single(**kwargs)
    return (final_coop, train_curve)

def main():
    """
    Main function to run the experiments.
    
    It sets up the hyperparameters, runs training and evaluation across multiple seeds,
    prints the results, and plots the training curves.
    """
    # ===== Hyperparameters =====
    num_agents       = 10
    alpha_mean       = 0.0
    alpha_std        = 0.0
    discount         = 0.2
    learning_rate    = 0.1
    initial_epsilon  = 0.1
    min_epsilon      = 0.01
    decay_rate       = 0.999
    prob_continue    = 0.99  # Chance to continue after min_rounds
    train_episodes   = 2000
    eval_episodes    = 10
    mode             = "ring"  # "random" or "ring"
    min_rounds       = 10
    num_seeds        = 10
    # ============================
    
    # Build argument list for each seed.
    arglist = []
    for s in range(num_seeds):
        kwargs = dict(
            seed=s,
            num_agents=num_agents,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            discount=discount,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            min_epsilon=min_epsilon,
            decay_rate=decay_rate,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            prob_continue=prob_continue,
            mode=mode,
            min_rounds=min_rounds
        )
        arglist.append((s, kwargs))
    
    # Run experiments in parallel.
    with mp.Pool(processes=num_seeds) as pool:
        results = pool.map(worker_single, arglist)
    
    final_coops = [result[0] for result in results]
    all_train_curves = [result[1] for result in results]
    mean_coop = np.mean(final_coops)
    std_coop = np.std(final_coops)
    
    print("\n==== 3-step (Partner-Specific) Tester ====")
    print(f"Mode: {mode}, alpha_mean: {alpha_mean:.2f}, alpha_std: {alpha_std:.2f}, #agents: {num_agents}")
    print(f"Discount: {discount}, Seeds: {num_seeds}, min_rounds: {min_rounds}")
    print(f"Training episodes: {train_episodes}, indefinite horizon probability: {prob_continue}")
    print("Final Cooperation Rates per Seed:", final_coops)
    print(f"Mean final cooperation = {mean_coop:.3f}, Std = {std_coop:.3f}\n")
    
    # Plot the training curves.
    plt.figure(figsize=(8, 5))
    for i, curve in enumerate(all_train_curves):
        plt.plot(curve, label=f"Seed {i} (final={final_coops[i]:.2f})")
    plt.xlabel("Training Episode")
    plt.ylabel("Cooperation Rate")
    plt.title(
        f"3-step PD with Indefinite Horizon (min_rounds={min_rounds})\n"
        f"Mean Final Cooperation = {mean_coop:.2f} ± {std_coop:.2f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("3step_minrounds_example.png")
    plt.show()

if __name__ == "__main__":
    main()
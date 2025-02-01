#!/usr/bin/env python3
"""
agent3.py

This module defines an Agent3 class for playing the repeated Prisoner's Dilemma (PD)
using partner-specific Q-learning with a 3-step memory. Each agent tracks a separate
memory and Q-table for every other potential opponent. The reward signal can be
"altruistic" – blending its own payoff with that of its opponent.
"""

import random
import numpy as np

# Define possible actions
C = 0  # Cooperate
D = 1  # Defect

# Payoff matrices for the PD game.
# For agent A (row player):
#   PAYOFFS_A[action_A][action_B] gives the payoff.
# For agent B (column player):
#   PAYOFFS_B[action_A][action_B] gives the payoff.
#
# For instance:
#   - (C, C): both get 3.
#   - (C, D): A gets 0, B gets 5.
#   - (D, C): A gets 5, B gets 0.
#   - (D, D): both get 1.
PAYOFFS_A = [
    [3, 0],  # When agent A cooperates: (C,C)=3, (C,D)=0
    [5, 1]   # When agent A defects:   (D,C)=5, (D,D)=1
]
PAYOFFS_B = [
    [3, 5],  # For agent B, note the swapped outcomes
    [0, 1]
]

def altruistic_reward(action_i, action_j, alpha_i):
    """
    Compute the reward for an agent using a blend of its own payoff and its opponent's payoff.
    
    The formula used is:
        reward = (1 - alpha_i)*Pi + alpha_i*Pj,
    where Pi is the agent's own payoff (from PAYOFFS_A) and Pj is the opponent's payoff (from PAYOFFS_B).
    
    Parameters:
        action_i: Action taken by the agent (C or D).
        action_j: Action taken by the opponent.
        alpha_i: The altruism factor (0.0 means selfish; higher values mean more caring).
    
    Returns:
        The blended reward.
    """
    p_i = PAYOFFS_A[action_i][action_j]
    p_j = PAYOFFS_B[action_i][action_j]
    return (1 - alpha_i) * p_i + alpha_i * p_j

def outcome_index(action_i, action_j):
    """
    Convert a pair of actions (agent's and opponent's) into a unique index (0 to 3).
    
    We use the following encoding:
        (C, C) -> 0, (C, D) -> 1, (D, C) -> 2, (D, D) -> 3.
    This compact index helps us represent a 2-bit outcome within a 3-step memory.
    """
    return action_i * 2 + action_j

def next_state_3step(old_state, new_outcome):
    """
    Update the agent's 3-step memory state by adding a new outcome.
    
    The memory is encoded as an integer from 0 to 63 (4^3 states). We shift out the oldest
    outcome and append the new outcome (which is a number in 0..3) on the right.
    
    Parameters:
        old_state: The previous memory state.
        new_outcome: The new outcome (0, 1, 2, or 3).
    
    Returns:
        The updated memory state.
    """
    # (old_state % 16) drops the oldest outcome (keeping the last 2 outcomes),
    # then multiplying by 4 shifts left, and we add the new outcome.
    return (old_state % 16) * 4 + new_outcome

class Agent3:
    """
    An agent that learns via Q-learning with partner-specific memory.
    
    Each agent maintains:
      - A separate 3-step memory for each opponent.
      - A separate Q-table (64 states x 2 actions) for each opponent.
    """
    def __init__(self, agent_id, num_agents, alpha=0.0, learning_rate=0.1, discount=0.9, epsilon=0.1):
        """
        Initialize the Agent.
        
        Parameters:
            agent_id: A unique identifier for the agent.
            num_agents: The total number of agents in the population.
            alpha: Altruism factor (how much the agent cares about its partner's payoff).
            learning_rate: Q-learning learning rate.
            discount: Discount factor (gamma) for future rewards.
            epsilon: Exploration rate for the epsilon-greedy policy.
        """
        self.agent_id = agent_id
        self.num_agents = num_agents

        self.alpha = alpha
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

        # Create one Q-table per opponent. Each Q-table has 64 states (3-step memory) and 2 possible actions.
        self.Q = [np.zeros((64, 2)) for _ in range(num_agents)]
        # Initialize memory state for each opponent to 0.
        self.current_state = [0] * num_agents

    def choose_action(self, opponent_id, update_Q_phase=True):
        """
        Choose an action for the given opponent using an epsilon-greedy strategy.
        
        Parameters:
            opponent_id: The id of the opponent.
            update_Q_phase: Whether we're in training mode (True) or evaluation mode (False).
            
        Returns:
            The chosen action (C or D).
        """
        state = self.current_state[opponent_id]
        if update_Q_phase:
            if random.random() < self.epsilon:
                # Explore: choose randomly.
                return random.choice([C, D])
            else:
                # Exploit: choose the best known action.
                return np.argmax(self.Q[opponent_id][state])
        else:
            # During evaluation, no exploration.
            return np.argmax(self.Q[opponent_id][state])

    def update_Q(self, opponent_id, old_state, action_i, action_j, reward_i, new_state, update_Q_phase=True):
        """
        Perform a standard Q-learning update for the Q-table corresponding to the given opponent.
        
        Parameters:
            opponent_id: The id of the opponent.
            old_state: The memory state before the round.
            action_i: The action chosen by this agent.
            action_j: The opponent's action (primarily for record keeping).
            reward_i: The reward received after taking the action.
            new_state: The memory state after the round.
            update_Q_phase: Whether to perform an update (skip during evaluation).
        """
        if not update_Q_phase:
            return

        best_next = np.max(self.Q[opponent_id][new_state])
        td_target = reward_i + self.gamma * best_next
        td_error = td_target - self.Q[opponent_id][old_state, action_i]
        self.Q[opponent_id][old_state, action_i] += self.lr * td_error

    def transition_state(self, opponent_id, action_i, action_j):
        """
        Update the memory state for a specific opponent based on the actions taken.
        
        Parameters:
            opponent_id: The id of the opponent.
            action_i: The action taken by this agent.
            action_j: The action taken by the opponent.
            
        Returns:
            A tuple (old_state, new_state).
        """
        old_state = self.current_state[opponent_id]
        new_outcome = outcome_index(action_i, action_j)
        new_state = next_state_3step(old_state, new_outcome)
        self.current_state[opponent_id] = new_state
        return old_state, new_state

    def reset_states(self):
        """
        Reset the memory states for all opponents.
        
        Useful at the beginning of a new training episode.
        """
        self.current_state = [0] * self.num_agents

def run_round(agent_i, agent_j, i_id, j_id, update_Q_phase=True):
    """
    Run one round of the PD game between two agents.
    
    Steps:
      1. Retrieve each agent's current memory state (specific to the opponent).
      2. Each agent selects an action using its Q-table.
      3. Both agents update their memory based on the joint outcome.
      4. Compute rewards using the altruistic_reward function.
      5. Update Q-values if in training mode.
      
    Parameters:
        agent_i, agent_j: The two Agent3 instances.
        i_id, j_id: Their corresponding identifiers.
        update_Q_phase: Whether to update Q-values (True in training, False in evaluation).
        
    Returns:
        A tuple (action_i, action_j) with the chosen actions.
    """
    old_state_i = agent_i.current_state[j_id]
    old_state_j = agent_j.current_state[i_id]

    action_i = agent_i.choose_action(j_id, update_Q_phase)
    action_j = agent_j.choose_action(i_id, update_Q_phase)

    _, new_state_i = agent_i.transition_state(j_id, action_i, action_j)
    _, new_state_j = agent_j.transition_state(i_id, action_j, action_i)

    r_i = altruistic_reward(action_i, action_j, agent_i.alpha)
    r_j = altruistic_reward(action_j, action_i, agent_j.alpha)

    agent_i.update_Q(j_id, old_state_i, action_i, action_j, r_i, new_state_i, update_Q_phase)
    agent_j.update_Q(i_id, old_state_j, action_j, action_i, r_j, new_state_j, update_Q_phase)

    return action_i, action_j

if __name__ == '__main__':
    # A simple test to ensure a round runs without issues.
    num_agents = 2
    agents = [Agent3(i, num_agents) for i in range(num_agents)]
    a, b = run_round(agents[0], agents[1], 0, 1)
    print(f"Test round actions: Agent0 -> {a}, Agent1 -> {b}")
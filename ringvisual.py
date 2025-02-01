import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np

from agent import Agent, clamp, run_round, C, D

def run_one_round_ring(population, update_Q_phase=True):
    """
    Performs exactly ONE round of PD in a ring:
      each agent i plays with neighbor (i+1) mod N.
    Returns total cooperations in this round (just for info).
    """
    num_agents = len(population)
    cooperation_count = 0
    
    # We'll do each pair (i, i+1) exactly once
    for i in range(num_agents):
        j = (i + 1) % num_agents
        action_i, action_j = run_round(population[i], population[j], update_Q_phase)
        if action_i == C:
            cooperation_count += 1
        if action_j == C:
            cooperation_count += 1

    return cooperation_count

def visualize_ring_evolution(
    num_agents=20,
    alpha=0.2,
    learning_rate=0.8,
    discount=0.95,
    epsilon=0.1,
    rounds=50,
    update_Q_phase=True
):
    """
    Creates 'num_agents' in a ring, then animates each round in real-time.
    Each node is colored green (C) or red (D) based on last action.
    """
    # 1) Create population
    population = []
    for _ in range(num_agents):
        agent = Agent(alpha=alpha,
                      learning_rate=learning_rate,
                      discount=discount,
                      epsilon=epsilon)
        population.append(agent)

    # 2) Build ring graph in NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_agents))
    for i in range(num_agents):
        j = (i+1) % num_agents
        G.add_edge(i, j)

    # Precompute a nice circular layout
    pos = nx.circular_layout(G)

    # 3) Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title("Ring PD Evolution")

    # We'll draw the graph once, then update node colors on each frame
    nx.draw_networkx_edges(G, pos=pos, ax=ax)
    # We'll create a list of node patches that we can recolor each frame
    node_patches = nx.draw_networkx_nodes(G, pos=pos, node_color="white", ax=ax)
    # We'll add labels separately if you want
    nx.draw_networkx_labels(G, pos=pos, ax=ax, font_color="black")

    ax.set_axis_off()

    # 4) Animation update function
    def update(frame):
        # Do ONE round of PD in the ring
        run_one_round_ring(population, update_Q_phase)

        # Now recolor nodes based on their last action
        colors = []
        for agent in population:
            if agent.last_action == C:
                colors.append("green")
            else:
                colors.append("red")

        # Update node colors
        node_patches.set_color(colors)
        # Optionally update the title to show current round
        ax.set_title(f"Ring PD Evolution - Round {frame+1}/{rounds}")

    # 5) Create animation
    ani = FuncAnimation(fig, update, frames=rounds, interval=800, repeat=False)

    plt.show()

    # After animation, you could check final Q-values or cooperation, etc.

if __name__ == "__main__":
    # Example usage
    visualize_ring_evolution(
        num_agents=20,
        alpha=0.1,
        learning_rate=0.1,
        discount=0.95,
        epsilon=0.2,
        rounds=60,
        update_Q_phase=True
    )
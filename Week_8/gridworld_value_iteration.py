"""
gridworld_value_iteration.py
Value iteration for the 4x3 stochastic Gridworld.
Saves value grids and policy arrows as PNGs for each reward.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- PARAMETERS ---
GAMMA = 0.99
THETA = 1e-4
REWARDS = [-2.0, -0.04, 0.02, 0.1, 1.0]   # r(s) values to try
OUT_DIR = "grid_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Grid layout: rows x cols (we use row 0 at top to match figure)
rows, cols = 3, 4

# Cells: ' ' normal, 'W' wall, '+1' terminal, '-1' terminal, 'S' start
# Layout adapted to standard example (3 rows x 4 cols)
# We'll index with (r,c): r in 0..2, c in 0..3
grid = [
    [' ', ' ', ' ', '+1'],
    [' ', 'W', ' ', '-1'],
    ['S', ' ', ' ', ' ']
]

ACTIONS = ['U','D','L','R']
action_vectors = {'U':(-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}

# transition probabilities: intended with 0.8, perpendicular slips 0.1 each
P_INTENDED = 0.8
P_SLIP = 0.1

def valid_cell(r,c):
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'W'

def move_from(r,c,action):
    dr, dc = action_vectors[action]
    nr, nc = r+dr, c+dc
    if not valid_cell(nr,nc):
        return r,c
    return nr,nc

def all_successors(r,c,action):
    # returns list of (prob, (nr,nc))
    succ = {}
    # intended
    nr,nc = move_from(r,c,action)
    succ[(nr,nc)] = succ.get((nr,nc),0) + P_INTENDED
    # perpendiculars
    if action in ('U','D'):
        for a in ('L','R'):
            nr,nc = move_from(r,c,a)
            succ[(nr,nc)] = succ.get((nr,nc),0) + P_SLIP
    else:
        for a in ('U','D'):
            nr,nc = move_from(r,c,a)
            succ[(nr,nc)] = succ.get((nr,nc),0) + P_SLIP
    items = [(p, s) for s,p in succ.items()]
    return items

def is_terminal(r,c):
    return grid[r][c] in ('+1','-1')

def value_iteration(step_reward):
    V = np.zeros((rows,cols))
    iterations = 0
    while True:
        delta = 0.0
        iterations += 1
        V_old = V.copy()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 'W' or is_terminal(r,c):
                    continue
                best = -1e9
                for a in ACTIONS:
                    ssum = 0.0
                    for prob, (nr,nc) in all_successors(r,c,a):
                        reward = 0.0
                        if grid[nr][nc] == '+1':
                            reward = 1.0
                        elif grid[nr][nc] == '-1':
                            reward = -1.0
                        else:
                            reward = step_reward
                        ssum += prob * (reward + GAMMA * V_old[nr,nc])
                    if ssum > best:
                        best = ssum
                V[r,c] = best
                delta = max(delta, abs(V[r,c] - V_old[r,c]))
        if delta < THETA:
            break
        if iterations > 10000:
            print("Warning: value iteration exceeded iteration limit.")
            break
    return V, iterations

def greedy_policy_from_V(V, step_reward):
    policy = np.full((rows,cols),' ')
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'W':
                policy[r,c] = 'W'
                continue
            if is_terminal(r,c):
                policy[r,c] = grid[r][c]
                continue
            best = -1e9
            best_action = ' '
            for a in ACTIONS:
                ssum = 0.0
                for prob, (nr,nc) in all_successors(r,c,a):
                    if grid[nr][nc] == '+1':
                        reward = 1.0
                    elif grid[nr][nc] == '-1':
                        reward = -1.0
                    else:
                        reward = step_reward
                    ssum += prob * (reward + GAMMA * V[nr,nc])
                if ssum > best:
                    best = ssum
                    best_action = a
            policy[r,c] = best_action
    return policy

def plot_value(V, reward_label):
    fig, ax = plt.subplots()
    im = ax.imshow(V, cmap='coolwarm', interpolation='none')
    for (j,i),label in np.ndenumerate(V):
        text = ''
        if grid[j][i] == 'W':
            text = 'W'
        else:
            text = f"{label:.2f}"
        ax.text(i, j, text, ha='center', va='center', color='black', fontsize=10)
    ax.set_title(f"Gridworld Values r={reward_label}")
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(OUT_DIR, f"values_r_{str(reward_label).replace('.','p')}.png"), bbox_inches='tight')
    plt.close(fig)

def plot_policy(policy, reward_label):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    for r in range(rows):
        for c in range(cols):
            cell = policy[r,c]
            if cell == 'W':
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5),1,1, color='gray'))
            elif cell in ('+1','-1'):
                ax.text(c, r, cell, fontsize=12, ha='center', va='center')
            elif cell == 'S':
                ax.text(c, r, 'S', fontsize=10, ha='center', va='center')
            else:
                if cell == 'U':
                    dx,dy = 0,-0.3
                    ax.arrow(c, r+0.2, 0, -0.4, head_width=0.15, head_length=0.1)
                if cell == 'D':
                    ax.arrow(c, r-0.2, 0, 0.4, head_width=0.15, head_length=0.1)
                if cell == 'L':
                    ax.arrow(c+0.2, r, -0.4, 0, head_width=0.15, head_length=0.1)
                if cell == 'R':
                    ax.arrow(c-0.2, r, 0.4, 0, head_width=0.15, head_length=0.1)
    ax.set_title(f"Greedy Policy r={reward_label}")
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"policy_r_{str(reward_label).replace('.','p')}.png"), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    for r in REWARDS:
        V, iters = value_iteration(r)
        print(f"Reward {r}: converged in {iters} iterations")
        plot_value(V, r)
        policy = greedy_policy_from_V(V, r)
        plot_policy(policy, r)
    print("Gridworld outputs saved to:", OUT_DIR)

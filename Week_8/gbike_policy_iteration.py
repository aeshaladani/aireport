"""
gbike_policy_iteration.py
Policy iteration for Gbike bicycle rental (classic Jack's car rental variant).
Includes an option for the modified problem with:
 - one free bike moved 1->2 per night
 - parking penalty INR 4 if bikes > 10 at a location after moving
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from math import exp, factorial

OUT_DIR = "gbike_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters
MAX_BIKES = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
PARKING_PENALTY = 4  # applied if >10 bikes after moving at a location
FREE_SHUTTLE = True  # set True to enable free shuttle (Problem 3)
DISCOUNT = 0.9
THETA = 1e-2   # evaluation threshold
POISSON_UPPER = 11  # up to which value we compute Poisson probabilities (you can increase)
# Poisson means
req_mean1 = 3
req_mean2 = 4
ret_mean1 = 3
ret_mean2 = 2

# Precompute Poisson probabilities
def poisson_pmf(n, lam):
    return exp(-lam) * (lam**n) / factorial(n)

poisson_req1 = [poisson_pmf(i, req_mean1) for i in range(POISSON_UPPER)]
poisson_req2 = [poisson_pmf(i, req_mean2) for i in range(POISSON_UPPER)]
poisson_ret1 = [poisson_pmf(i, ret_mean1) for i in range(POISSON_UPPER)]
poisson_ret2 = [poisson_pmf(i, ret_mean2) for i in range(POISSON_UPPER)]

# To ensure they sum ~1, add remainder onto last bucket
def normalize(ps):
    s = sum(ps)
    ps[-1] += (1.0 - s)
    return ps

poisson_req1 = normalize(poisson_req1)
poisson_req2 = normalize(poisson_req2)
poisson_ret1 = normalize(poisson_ret1)
poisson_ret2 = normalize(poisson_ret2)

# State space: (b1, b2)
V = np.zeros((MAX_BIKES+1, MAX_BIKES+1))
policy = np.zeros_like(V, dtype=int)

def expected_return(b1, b2, action, V, modified=False):
    # action = bikes moved from 1->2 (positive) or 2->1 (negative)
    # enforce feasible action
    b1_after_move = min(MAX_BIKES, max(0, b1 - action))
    b2_after_move = min(MAX_BIKES, max(0, b2 + action))

    # Cost for move
    move_cost = MOVE_COST * abs(action)
    if modified and FREE_SHUTTLE:
        # if moving from 1->2 and at least 1 moved, one is free
        if action > 0:
            move_cost = MOVE_COST * max(0, abs(action)-1)
    # parking cost
    parking_cost = 0
    if modified:
        if b1_after_move > 10:
            parking_cost += PARKING_PENALTY
        if b2_after_move > 10:
            parking_cost += PARKING_PENALTY

    expected = -move_cost - parking_cost  # immediate move/parking penalty part (other parts add via rentals)
    total = 0.0
    # iterate over possible rental requests and returns (use truncated Poisson)
    for req1, p_req1 in enumerate(poisson_req1):
        for req2, p_req2 in enumerate(poisson_req2):
            prob_req = p_req1 * p_req2
            # bikes rented equals min(bikes available after move, requests)
            rentals1 = min(b1_after_move, req1)
            rentals2 = min(b2_after_move, req2)
            reward_from_rentals = RENTAL_REWARD * (rentals1 + rentals2)
            # bikes left after rentals
            b1_after_rent = b1_after_move - rentals1
            b2_after_rent = b2_after_move - rentals2
            # now returns
            for ret1, p_ret1 in enumerate(poisson_ret1):
                for ret2, p_ret2 in enumerate(poisson_ret2):
                    prob = prob_req * p_ret1 * p_ret2
                    new_b1 = min(MAX_BIKES, b1_after_rent + ret1)
                    new_b2 = min(MAX_BIKES, b2_after_rent + ret2)
                    total += prob * (reward_from_rentals + DISCOUNT * V[new_b1, new_b2])
    return expected + total

def policy_evaluation(V, policy, modified=False):
    while True:
        delta = 0.0
        for b1 in range(MAX_BIKES+1):
            for b2 in range(MAX_BIKES+1):
                v = V[b1,b2]
                a = policy[b1,b2]
                V[b1,b2] = expected_return(b1,b2,a,V,modified=modified)
                delta = max(delta, abs(v - V[b1,b2]))
        if delta < THETA:
            break
    return V

def policy_improvement(V, policy, modified=False):
    policy_stable = True
    for b1 in range(MAX_BIKES+1):
        for b2 in range(MAX_BIKES+1):
            old_action = policy[b1,b2]
            best_value = -1e9
            best_action = old_action
            for a in range(-MAX_MOVE, MAX_MOVE+1):
                # action feasible?
                if (0 <= b1 - a <= MAX_BIKES) and (0 <= b2 + a <= MAX_BIKES):
                    val = expected_return(b1,b2,a,V,modified=modified)
                    if val > best_value:
                        best_value = val
                        best_action = a
            policy[b1,b2] = best_action
            if best_action != old_action:
                policy_stable = False
    return policy_stable, policy

def policy_iteration(modified=False):
    # initialize value and policy
    V = np.zeros((MAX_BIKES+1, MAX_BIKES+1))
    policy = np.zeros_like(V, dtype=int)
    iter_count = 0
    while True:
        iter_count += 1
        V = policy_evaluation(V, policy, modified=modified)
        stable, policy = policy_improvement(V, policy, modified=modified)
        print(f"[Policy Iter] iter {iter_count} stable={stable}")
        if stable or iter_count > 20:
            break
    return V, policy

def save_heatmap(mat, name):
    plt.figure(figsize=(6,5))
    plt.imshow(mat, origin='lower', aspect='auto')
    plt.colorbar()
    plt.title(name)
    plt.xlabel("b2")
    plt.ylabel("b1")
    plt.savefig(os.path.join(OUT_DIR, name.replace(" ", "_") + ".png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Running Gbike base problem (no free shuttle, no parking penalty)...")
    V_base, policy_base = policy_iteration(modified=False)
    save_heatmap(V_base, "Value_Base")
    save_heatmap(policy_base, "Policy_Base")
    np.savetxt(os.path.join(OUT_DIR, "policy_base.txt"), policy_base, fmt='%d')

    print("Running Gbike modified problem (free shuttle; parking penalty)...")
    V_mod, policy_mod = policy_iteration(modified=True)
    save_heatmap(V_mod, "Value_Modified")
    save_heatmap(policy_mod, "Policy_Modified")
    np.savetxt(os.path.join(OUT_DIR, "policy_modified.txt"), policy_mod, fmt='%d')

    print("Gbike outputs saved to:", OUT_DIR)

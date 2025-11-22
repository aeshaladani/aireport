import numpy as np
import matplotlib.pyplot as plt

def modified_epsilon_greedy(steps=10000, eps=0.1, alpha=0.7, seed=0):
    rng = np.random.RandomState(seed)
    k = 10
    true_means = np.ones(k)
    Q = np.zeros(k)
    rewards = np.zeros(steps)
    optimal_selected = np.zeros(steps, dtype=int)

    for t in range(steps):
        if rng.rand() > eps:
            action = np.argmax(Q)
        else:
            action = rng.randint(k)
        # random walk for true means
        true_means += rng.normal(0, 0.01, size=k)
        reward = rng.normal(true_means[action], 1.0)
        Q[action] += alpha * (reward - Q[action])  # constant step-size update
        rewards[t] = reward
        optimal_selected[t] = 1 if action == np.argmax(true_means) else 0

    avg_reward = np.cumsum(rewards) / (np.arange(steps) + 1)
    return avg_reward, optimal_selected, rewards

if __name__ == '__main__':
    avg_reward_alpha, opt_selected_alpha, _ = modified_epsilon_greedy(steps=10000, eps=0.1, alpha=0.7, seed=123)
    # Compare with sample-average baseline (alpha=1/n) quickly for reference
    avg_reward_sa, opt_selected_sa, _ = __import__('nonstationary_10arm').nonstationary_10_arm(steps=10000, eps=0.1, seed=123)

    plt.plot(avg_reward_alpha, label='alpha=0.7 (modified)')
    plt.plot(avg_reward_sa, label='sample-average (1/n)')
    plt.title("Average Reward: modified vs sample-average")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

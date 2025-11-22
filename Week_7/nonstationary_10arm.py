import numpy as np
import matplotlib.pyplot as plt

def nonstationary_10_arm(steps=10000, eps=0.1, seed=0):
    rng = np.random.RandomState(seed)
    k = 10
    # initialize true means to 1.0 (as in your lab)
    true_means = np.ones(k)
    Q = np.zeros(k)
    N = np.zeros(k, dtype=int)
    rewards = np.zeros(steps)
    optimal_selected = np.zeros(steps, dtype=int)

    for t in range(steps):
        if rng.rand() > eps:
            action = np.argmax(Q)
        else:
            action = rng.randint(k)
        # before sampling reward, update true means with small gaussian step
        true_means += rng.normal(0, 0.01, size=k)
        reward = rng.normal(true_means[action], 1.0)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards[t] = reward
        # track whether chosen action is optimal according to current true_means
        optimal_action = np.argmax(true_means)
        optimal_selected[t] = 1 if action == optimal_action else 0

    avg_reward = np.cumsum(rewards) / (np.arange(steps) + 1)
    return avg_reward, optimal_selected, rewards

if __name__ == '__main__':
    avg_reward, optimal_selected, rewards = nonstationary_10_arm(steps=10000, eps=0.1, seed=42)
    # Plot average reward
    plt.plot(avg_reward)
    plt.title("Non-Stationary 10-Arm: Average Reward")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()

    # Plot % optimal action chosen (smoothed)
    window = 100
    pct_opt = np.convolve(optimal_selected, np.ones(window)/window, mode='valid') * 100
    plt.plot(np.arange(len(pct_opt)) + window//2, pct_opt)
    plt.title("% Optimal Action (moving average window=100)")
    plt.xlabel("Step")
    plt.ylabel("% Optimal Action")
    plt.grid(True)
    plt.show()

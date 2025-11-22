import numpy as np
import matplotlib.pyplot as plt

def run_binary_bandit(p1=0.6, p2=0.4, eps=0.1, steps=1000, seed=0):
    np.random.seed(seed)
    Q = np.array([0.0, 0.0])
    N = np.array([0,0], dtype=int)
    rewards = np.zeros(steps)
    for t in range(steps):
        if np.random.rand() > eps:
            action = np.argmax(Q)
        else:
            action = np.random.choice([0,1])
        # sample reward from Bernoulli
        prob = p1 if action == 0 else p2
        r = 1 if np.random.rand() < prob else 0
        N[action] += 1
        Q[action] += (r - Q[action]) / N[action]  # incremental mean
        rewards[t] = r
    return Q, N, rewards

if __name__ == '__main__':
    Q, N, rewards = run_binary_bandit(p1=0.7, p2=0.3, eps=0.1, steps=1000, seed=1)
    print("Final estimates Q:", Q, "Counts:", N)
    # plot average reward over time
    avg_reward = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.plot(avg_reward)
    plt.title("Binary Bandit: Average Reward over Time")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()

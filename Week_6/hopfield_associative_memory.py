import numpy as np
import random

def to_bipolar(x):
    return np.where(x == 0, -1, 1)

def hebbian_train(patterns):
    N = patterns[0].size
    W = np.zeros((N, N))
    for p in patterns:
        bp = to_bipolar(p)
        W += np.outer(bp, bp)
    np.fill_diagonal(W, 0)
    return W / N

def hopfield_update(W, state):
    new_state = state.copy()
    for i in range(len(state)):
        s = np.dot(W[i], new_state)
        new_state[i] = 1 if s >= 0 else -1
    return new_state

def recall(W, pattern, steps=5):
    state = to_bipolar(pattern)
    for _ in range(steps):
        state = hopfield_update(W, state)
    return state

if __name__ == "__main__":
    patterns = [
        np.random.randint(0, 2, (100,)),
        np.random.randint(0, 2, (100,))
    ]

    W = hebbian_train(patterns)

    noisy = patterns[0].copy()
    flip_positions = random.sample(range(100), 10)
    for pos in flip_positions:
        noisy[pos] ^= 1

    recalled = recall(W, noisy)
    print("Recalled Pattern:")
    print(recalled.reshape(10,10))

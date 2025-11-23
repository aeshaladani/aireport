import numpy as np
import random
from hopfield_associative_memory import hebbian_train, recall

def test_error_tolerance(pattern, W, trials=20):
    N = pattern.size
    max_flips = 0

    for flips in range(1, 30):
        success = 0
        for _ in range(trials):
            noisy = pattern.copy()
            idx = random.sample(range(N), flips)
            for i in idx:
                noisy[i] ^= 1
            if np.array_equal(recall(W, noisy), np.where(pattern==0,-1,1)):
                success += 1
        if success < trials / 2:
            break
        max_flips = flips

    return max_flips

if __name__ == "__main__":
    p = np.random.randint(0,2,(100,))
    W = hebbian_train([p])
    print("Error correction capability:", test_error_tolerance(p, W))

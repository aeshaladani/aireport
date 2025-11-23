import numpy as np
import random

def energy(board):
    E = 0
    for r in range(8):
        E += (np.sum(board[r]) - 1)**2
    for c in range(8):
        E += (np.sum(board[:,c]) - 1)**2
    return E

def random_board():
    B = np.zeros((8,8))
    for r in range(8):
        B[r, random.randint(0,7)] = 1
    return B

def swap(board):
    B = board.copy()
    ones = np.argwhere(B==1)
    zeros = np.argwhere(B==0)
    r1,c1 = random.choice(ones)
    r2,c2 = random.choice(zeros)
    B[r1,c1], B[r2,c2] = 0, 1
    return B

if __name__ == "__main__":
    B = random_board()
    E = energy(B)

    for _ in range(2000):
        newB = swap(B)
        newE = energy(newB)
        if newE < E:
            B, E = newB, newE
        if E == 0:
            break

    print("Final board:")
    print(B)

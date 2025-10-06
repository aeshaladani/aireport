import cv2
import numpy as np
import random
import math
from copy import deepcopy

#PARAMETERS
ROWS, COLS = 3,3
T0 = 1000        # Initial temperature
ALPHA = 0.995    # Cooling rate
MAX_ITER = 10000 # Maximum iterations
STOP_T = 1e-3    # Minimum temperature

#HELPER FUNCTIONS

def split_image(image, rows, cols):
    """Split the image into equal tiles."""
    h, w = image.shape[:2]
    tile_h, tile_w = h // rows, w // cols
    pieces = []
    for i in range(rows):
        for j in range(cols):
            piece = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            pieces.append(piece)
    return pieces

def reconstruct_image(pieces, order, rows, cols):
    """Rebuild image from ordered pieces."""
    h, w = pieces[0].shape[:2]
    new_img = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
    for idx, pos in enumerate(order):
        r, c = divmod(idx, cols)
        new_img[r*h:(r+1)*h, c*w:(c+1)*w] = pieces[pos]
    return new_img

def edge_diff(p1, p2, side):
    """Compute edge difference between two adjacent pieces."""
    if side == 'right':
        return np.sum((p1[:, -1, :] - p2[:, 0, :])**2)
    elif side == 'left':
        return np.sum((p1[:, 0, :] - p2[:, -1, :])**2)
    elif side == 'bottom':
        return np.sum((p1[-1, :, :] - p2[0, :, :])**2)
    elif side == 'top':
        return np.sum((p1[0, :, :] - p2[-1, :, :])**2)
    return 0

def energy(pieces, order, rows, cols):
    """Compute total mismatch energy for a given arrangement."""
    total = 0
    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
            current = pieces[order[idx]]
            if j < cols - 1:  # right neighbor
                right = pieces[order[idx + 1]]
                total += edge_diff(current, right, 'right')
            if i < rows - 1:  # bottom neighbor
                bottom = pieces[order[idx + cols]]
                total += edge_diff(current, bottom, 'bottom')
    return total

def neighbor(order):
    """Generate a neighbor by swapping two random tiles."""
    new_order = deepcopy(order)
    i, j = random.sample(range(len(order)), 2)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order

#SIMULATED ANNEALING
def simulated_annealing(pieces, rows, cols):
    order = list(range(len(pieces)))
    random.shuffle(order)
    best = deepcopy(order)
    E_best = energy(pieces, best, rows, cols)
    s = deepcopy(order)
    E_s = E_best
    T = T0

    for iteration in range(MAX_ITER):
        s_new = neighbor(s)
        E_new = energy(pieces, s_new, rows, cols)
        delta = E_new - E_s

        if delta < 0 or random.random() < math.exp(-delta / T):
            s = deepcopy(s_new)
            E_s = E_new

            if E_new < E_best:
                best = deepcopy(s_new)
                E_best = E_new

        T *= ALPHA
        if T < STOP_T:
            break

        if iteration % 500 == 0:
            print(f"Iteration {iteration}, Energy = {E_best:.2f}")

    print("Final energy:", E_best)
    return best

#EXECUTION

if __name__ == "__main__":
    image = cv2.imread("scrambled.png")
    pieces = split_image(image, ROWS, COLS)
    best_order = simulated_annealing(pieces, ROWS, COLS)
    result = reconstruct_image(pieces, best_order, ROWS, COLS)

    cv2.imwrite("solved_puzzle.png", result)
    print("Solved puzzle saved as solved_puzzle.png")

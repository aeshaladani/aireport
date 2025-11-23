import numpy as np

N = 10
neurons = N*N

def generate_coords():
    return np.random.rand(N,2)

def compute_distances(C):
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            D[i,j] = np.linalg.norm(C[i] - C[j])
    return D

def hopfield_tsp_weights(D):
    W = np.zeros((neurons, neurons))

    def idx(i,p):
        return i*N + p

    A = B = 2
    C = 1

    for i in range(N):
        for p in range(N):
            for j in range(N):
                for q in range(N):
                    a = idx(i,p)
                    b = idx(j,q)
                    if a == b:
                        continue
                    if i == j and p != q:
                        W[a,b] -= A
                    if p == q and i != j:
                        W[a,b] -= B
                    if p == (q-1)%N:
                        W[a,b] -= C * D[i,j]
    return W

if __name__ == "__main__":
    coords = generate_coords()
    D = compute_distances(coords)
    W = hopfield_tsp_weights(D)
    print("Weights shape:", W.shape)
    print("Total weights:", (neurons*(neurons-1))//2)

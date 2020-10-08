from mat import *
import numpy as np

def ln(x):
    """Approximation of the natural logarithm,
    return a large negative number if x is zero or negative"""
    n_terms = 10
    if x <= 0:
        return -10**10
    else:
        terms = [(-1)**(k+1)*(x-1)**k/k for k in range(1,n_terms)]
    return sum(terms)

def viterbi(A, B, pi, obs):
    A = np.array(A)
    B = np.array(B)
    pi = np.array(pi)
    obs = np.array(obs)
    N = len(pi)
    T = len(obs)

    deltanew = [[0 for i in range(N)] for t in range(T)]
    indices = [[0 for i in range(N)] for t in range(T)]

    deltanew[0] = [ln(p)+ln(b) for (b, p) in zip(B[:,obs[0]], pi)]


    for t in range(1, T):
        for i in range(0,N):

            # First j=0 as candidate
            deltanew[t][i] = ln(A[0][i]) + deltanew[t - 1][0] + ln(B[i][obs[t]])
            indices[t][i] = 0

            for j in range(1,N):
                candidate = ln(A[j][i]) + deltanew[t - 1][j] + ln(B[i][obs[t]]) 

                if candidate > deltanew[t][i]:
                    deltanew[t][i] = candidate
                    indices[t][i] = j


    arr = [0 for _ in range(T)]

    state_prob = deltanew[T-1][0]
    arr[T-1] = 0

    # Find argmax_i for last timestep
    for i in range(N):
        if deltanew[-1][i] > state_prob:
            state_prob = deltanew[T-1][i]
            arr[T-1] = i

    # Backtracking
    for t in range(T-2, -1, -1):
        arr[t] = indices[t+1][arr[t+1]]


    return arr
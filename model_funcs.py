from math import inf, log, exp
from random import gauss

def randrow(n):
    res = [gauss(0, 15) for _ in range(n)]
    res = [-x if x < 0 else x for x in res]
    s = sum(res)
    return [x / s for x in res]

def random_model(states, emissions):
    A = [randrow(states) for _ in range(states)]
    B = [randrow(emissions) for _ in range(states)]
    pi = randrow(states)
    return A, B, pi

def mprint(m):
    def printrow(r):
        print('\t'.join([str(round(x, 3)) for x in r]))
    if type(m[0]) == float or type(m[0]) == int:
        printrow(m)
    else:
        for r in m: printrow(r)

def mprints(*ms):
    for m in ms: 
        mprint(m)
        print()

def alpha_pass(A, B, pi, obs):
    T = len(obs)
    N = len(pi)

    alpha0 = []
    for i in range(N):
        alpha0.append(B[i][obs[0]] * pi[i])

    c0 = 1. / sum(alpha0)

    alpha0 = [alph * c0 for alph in alpha0]

    alpha = [alpha0]
    c = [c0]
    for t in range(1, T):
        alphat = []
        for i in range(N):
            alphat.append(B[i][obs[t]] * sum([
                A[j][i] * alpha[t - 1][j]
                for j in range(N)
            ]))
        c_t = 1. / sum(alphat)
        c.append(c_t)
        alphat = [alph * c_t for alph in alphat]
        alpha.append(alphat)
    return alpha, c

def beta_pass(A, B, pi, obs, alpha, c):

    T = len(obs)
    N = len(pi)

    beta = [
        [c[t] for _ in range(N)]
        for t in range(T)
    ]

    for t in range(T-2, -1, -1):
        for i in range(N):
            new = 0
            for j in range(N):
                new += beta[t + 1][j] * B[j][obs[t + 1]] * A[i][j]
            beta[t][i] = new * c[t]
    return beta

def b_w_iter(A, B, pi, obs):

    T = len(obs)
    N = len(pi)
    K = len(B[0])

    alpha, c = alpha_pass(A, B, pi, obs)
    beta = beta_pass(A, B, pi, obs, alpha, c)
    alphasum = sum(alpha[-1])
    digamma = [
        [
            [
                (alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j])
                for j in range(N)
            ]
            for i in range(N)
        ]
        for t in range(T - 1)
    ]

    # Calculate gammas
    gamma = [
        [
            sum(digamma[t][i])
            for i in range(N)
        ]
        for t in range(T - 1)
    ]
    gamma.append(alpha[T - 1])

    # Reestimate
    An = [
        [
            sum([digamma[t][i][j] for t in range(T - 1)]) \
                / sum([gamma[t][i] for t in range(T - 1)])
            for j in range(N)
        ]
        for i in range(N)
    ]

    ind = lambda t, k: 1 if obs[t] == k else 0
    Bn = [
        [
            max(1e-8, sum([ind(t, k) * gamma[t][j] for t in range(T)]) \
                / sum([gamma[t][j] for t in range(T)]))
            for k in range(K)
        ]
        for j in range(N)
    ]

    pin = [gamma[0][i] for i in range(N)]
    
    logProb = - sum([log(c[i]) for i in range(T)])

    return An, Bn, pin, logProb

def b_w(A, B, pi, obs, maxIter = 50):

    T = len(obs)
    N = len(pi)
    K = len(B[0])
    oldLogProb = -inf

    i = 0
    while i <= maxIter:
        A, B, pi, logProb = b_w_iter(A, B, pi, obs)
        if logProb + 1e-5 > oldLogProb:
            oldLogProb = logProb
        else:
            break
        i += 1

    return A, B, pi, oldLogProb, i

def predict(A, B, pi, obs):
    alpha, c = alpha_pass(A, B, pi, obs)
    return exp(- sum([log(c_) for c_ in c]))


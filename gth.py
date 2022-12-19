'''
script to calculate stationary probabilities of an irreducible finite Markov chain by state reduction (numerically stable procedure)
Daniel Joshua Sharpe
Dec 2022
'''

import numpy as np

# Grassmann-Taksar-Heyman (GTH) algorithm
# input: irreducible row-stochastic matrix P
# output: stationary distribution vector pi
def gth_algo(P):
    N = np.shape(P)[1] # number of states
    # elimination phase
    for n in range(N-1,0,-1):
        Sn = np.sum([Pnj for j, Pnj in enumerate(P[n,:]) if j<n])
        P[:n,n] *= 1./Sn
        for i in range(n):
            for j in range(n):
                P[i,j] += P[i,n]*P[n,j]
    # trivial solution for one-node system
    pi = np.zeros(N,dtype=float)
    pi[0] = 1.
    # recursive phase
    mu = 1.
    for n in range(1,N):
        pi[n] = P[0,n] + np.sum([pi[j]*P[j,n] for j in range(1,n)])
        mu += pi[n]
    pi *= 1./mu # normalization
    return pi

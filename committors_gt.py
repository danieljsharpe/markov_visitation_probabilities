'''
script to calculate committor probabilities of a finite Markov chain by state reduction (numerically stable procedure)
DJ Sharpe & DJ Wales, Phys Rev E (2021), 104, 015301
Daniel Joshua Sharpe
Dec 2022
'''

import numpy as np
import copy

# calculate forward committor probabilities qf, allowing qp /= 0 for nodes at the boundary of the initial state
# ie, qf[j] is the probability that the j-th node visits state A before hitting state B
# inputs:
#   P:    row-stochastic matrix
#   in_A: flag indicating nodes belonging to final (ie absorbing) state
#   in_B: flag indicating nodes belonging to initial state
# outputs:
#   qf:   forward committor probabilities
def calc_committors(P,in_A,in_B):
    N = np.shape(P)[1] # number of states
    states = np.array([i for i in range(1,N+1)])
    in_AorB = in_A + in_B
    nodes_I = states[np.bitwise_not(np.array(in_AorB,dtype=bool))] # nodes in set: I = (A \cup B)^c
    for n in nodes_I: # eliminate nodes of state I, retaining transitions *from* nodes n \in I
        # trick to keep numerical stability (fac is equivalent to 1-P_nn)
        fac = 0.
        for j in states:
            if j==n: continue
            fac += P[n-1,j-1]
        # renormalization of transition probabilities, retaining probabilities FROM node n
        for i in states:
            for j in states:
                if i==n or j==n: continue # renormalize transition from n last
                P[i-1,j-1] += P[i-1,n-1]*P[n-1,j-1]/fac
        # renormalize transitions from n
        tnn = P[n-1,n-1]
        for j in states:
            P[n-1,j-1] += tnn*P[n-1,j-1]/fac
        P[:,n-1] = np.zeros(N) # transitions TO node n are eliminated
    # extract transition probabilities from *to* B, and calculate committors
    # NB at this point, only nonzero P[i,j] are for j \in (A \cup B)
    mask = np.array(in_A,dtype=bool)
    idcs = states[mask]-1
    P_B = copy.deepcopy(P[:,idcs])
    qf = np.zeros(N)
    for j in states:
        qf[j-1] = np.sum(P_B[j-1,:])
    return qf

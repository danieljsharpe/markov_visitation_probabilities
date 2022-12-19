'''
script to calculate node visitation probabilities along reactive and first passage
paths in a finite Markov chain, by simulation and by renormalization (stable method)
Daniel Joshua Sharpe
Nov 2022
'''

import gth
import committors_gt
import numpy as np
import copy



## INPUTS

# transition probability matrix
#              1    2    3    4    5    6    7    8    9    10   11   12
P = np.array([[0.70,0.20,0.00,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00], # 1
              [0.10,0.75,0.05,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00], # 2
              [0.00,0.10,0.75,0.10,0.05,0.00,0.00,0.00,0.00,0.00,0.00,0.00], # 3
              [0.05,0.05,0.00,0.80,0.00,0.05,0.05,0.00,0.00,0.00,0.00,0.00], # 4
              [0.00,0.00,0.10,0.00,0.65,0.05,0.10,0.05,0.05,0.00,0.00,0.00], # 5
              [0.00,0.00,0.00,0.10,0.05,0.65,0.00,0.10,0.05,0.05,0.00,0.00], # 6
              [0.00,0.10,0.00,0.05,0.05,0.00,0.75,0.05,0.00,0.00,0.00,0.00], # 7
              [0.00,0.00,0.00,0.00,0.10,0.15,0.05,0.60,0.10,0.00,0.00,0.00], # 8
              [0.00,0.00,0.00,0.00,0.05,0.10,0.00,0.10,0.60,0.05,0.05,0.05], # 9
              [0.00,0.00,0.00,0.00,0.00,0.05,0.00,0.00,0.10,0.80,0.05,0.00], # 10
              [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.05,0.90,0.00], # 11
              [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.00,0.00,0.95]])# 12


#p0 = np.array([0.5,0.3,0.2,0.,0.,0.,0.,0.,0.,0.,0.,0.]) # initial distribution
p0 = np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) # initial distribution

in_A = np.array([0,0,0,0,0,0,0,0,0,1,1,1],dtype=int) # node in final set (Y/N)
in_B = np.array([1,0,0,0,0,0,0,0,0,0,0,0],dtype=int) # node in initial set (Y/N)

in_Z = np.array([0,0,0,0,1,0,0,0,0,0,0,0],dtype=int) # states for which to calculate visitation probabilities

n_traj = 4000 # no. of trajectories to simulate

## END INPUTS




N = np.shape(P)[1] # number of states
states = np.array([i for i in range(1,N+1)])

nodes_A = states[np.array(in_A,dtype=bool)]
nodes_B = states[np.array(in_B,dtype=bool)]
nodes_Z = states[np.array(in_Z,dtype=bool)]


# check that transition matrix is row-stochastic
for i in range(N): assert(abs((np.sum(P[i,:])-1.)<1.E-8))
# check that initial probability vector is valid (and contained within B)
assert(abs(np.sum(p0*np.array(in_B,dtype=float))-1.)<1.E-8)
# check that nodes in Z are not in endpoint set (A \cup B)
for i in range(N): assert(not (in_Z[i-1] and (in_B[i-1] or in_A[i-1])))

pi = gth.gth_algo(copy.deepcopy(P)) # calculate stationary distribution by state reduction (numerically stable)
qf = committors_gt.calc_committors(copy.deepcopy(P),in_A,in_B) # calculate forward committor probabilities by state reduction (numerically stable)

# calculate initial distribution of reactive trajectories at steady-state
mu0 = np.zeros(N)
for j in states:
    mu0[j-1] = pi[j-1]*np.dot(P[j-1,:],qf)*float(in_B[j-1])
mu0 *= 1./np.sum(mu0) # normalize

# calculate initial distribution of reactive trajectories at steady state

print("\nnumber of nodes: ",N,
      "\ninitial state nodes (B):\n",nodes_B,
      "\nabsorbing state nodes (A):\n",nodes_A,
      "\nvisited state nodes (Z):\n",nodes_Z)
print("\nstationary distribution:\n",pi)
print("\nforward committor probabilities:\n",qf)
print("\ninitial probability distribution:\n",p0)
print("\ninitial distribution of steady-state reactive paths:\n",mu0)

## PERFORM SIMULATION - NONEQUILIBRIUM PATHS

v_fp = np.zeros(N)                        # flag: node visited on first passage path (Y/N)
v_tp = np.zeros(N)                        # flag: node visited on transition path (starting from particular initial node) (Y/N)
r_fp = np.zeros(N)                        # visitation probabilities for nodes on first passage paths
r_tp = dict.fromkeys(nodes_B,np.zeros(N)) # visitation probabilities for nodes on transition paths (starting from particular initial node)
n_tp = dict.fromkeys(nodes_B,0)           # number of recorded transition paths starting from particular initial nodes
v_fp_Z = 0                         # flag: macrostate Z visited on first passage path (Y/N)
v_tp_Z = 0                         # flag: macrostate Z visited on reactive path (Y/N)
r_fp_Z = 0.                        # visitation probability for macrostate Z on first passage paths
r_tp_Z = dict.fromkeys(nodes_B,0)  # visitation probability for macrostate Z on transition paths (starting from particular initial node)

print("\ndoing simulation of nonequilibrium paths... (",n_traj," trajectories )\n")
for i in range(n_traj):
    n0 = np.random.choice(states,1,p=p0)[0] # draw initial state from p0
    b0 = n0 # b0 is the initial node of a reactive trajectory segment
    v_fp[n0-1] = 1
    v_tp[n0-1] = 1
    while (not in_A[n0-1]):
        n0 = np.random.choice(states,1,p=P[n0-1,:])[0] # draw next state from current row of P
        v_fp[n0-1] = 1
        if (in_Z[n0-1]): v_fp_Z = 1
        if (in_B[n0-1]): # gone back to initial state; reset reactive visit flags
            v_tp = np.zeros(N)
            v_tp_Z = 0
            b0 = n0
        v_tp[n0-1] = 1
        if (in_Z[n0-1]): v_tp_Z = 1
    # accumulate flag counts
    r_fp += v_fp
    r_tp[b0] += v_tp
    r_fp_Z += v_fp_Z
    r_tp_Z[b0] += v_tp_Z
    n_tp[b0] += 1
    # reset visit flags
    v_fp = np.zeros(N)
    v_tp = np.zeros(N)
    v_fp_Z = 0
    v_tp_Z = 0

r_fp *= 1./float(n_traj)
r_fp_Z *= 1./float(n_traj)
for b in nodes_B:
    r_tp[b] *= 1./float(n_tp[b])
    r_tp_Z[b] *= 1./float(n_tp[b])

print("probability that Z is visited on nonequilibrium first passage paths (simulation):\n",r_fp_Z,"\n")
print("probability that Z is visited on nonequilibrium transition paths (simulation) [for specific initial nodes]:\n",r_tp_Z,"\n")


## PERFORM RENORMALIZATION CALCULATIONS (DENSE MATRIX FORMULATION)

notin_N = in_A + in_B + in_Z # mask for states to not eliminate in first part of algorithm

Pc = copy.deepcopy(P) # censored (renormalized) transition matrix

for n in states:
    if (notin_N[n-1]): continue
    # trick to keep numerical stability
    Pc[n-1,n-1] = 0.
    fac = np.sum(Pc[n-1,:]) # equivalent to 1-P_nn
    # renormalization
    for i in states:
        for j in states: 
            Pc[i-1,j-1] += Pc[i-1,n-1]*Pc[n-1,j-1]/fac
    Pc[n-1,:] = np.zeros(N)
    Pc[:,n-1] = np.zeros(N)



###  calculate first passage visitation probability by state reduction

Pc_fp = copy.deepcopy(Pc)
for b in nodes_B:
    # trick to keep numerical stability (fac is equivalent to 1-P_bb)
    fac = 0.
    for j in states:
        if j==b: continue
        fac += Pc_fp[b-1,j-1]
    # renormalization of transition probabilities, retaining transitions FROM node b ONLY
    for i in states:
        for j in states:
            if i==b or j==b: continue # renormalize transition from b last
            Pc_fp[i-1,j-1] += Pc_fp[i-1,b-1]*Pc_fp[b-1,j-1]/fac
    # renormalize transitions from b
    tbb = Pc_fp[b-1,b-1]
    for j in states:
        Pc_fp[b-1,j-1] += tbb*Pc_fp[b-1,j-1]/fac
    Pc_fp[:,b-1] = np.zeros(N) # transitions TO node b are eliminated

r_fp_gt = 0
for z in nodes_Z:
    r_fp_gt += np.sum(Pc_fp[:,z-1]*p0)

print("probability that Z is visited on first passage paths (state reduction):\n",r_fp_gt,"\n")

## PERFORM SIMULATION - EQUILIBRIUM (STEADY STATE) PATHS

v_tp_ss = np.zeros(N) # flag: node visited on steady-state transition path (Y/N)
r_tp_ss = np.zeros(N) # visitation probabilities for nodes on steady-state transition paths
v_tp_Z_ss = 0         # flag: macrostate Z visited on steady-state reactive path (Y/N)
r_tp_Z_ss = 0.        # visitation probability for macrostate Z on steady-state transition paths
from_A = 0            # flag: =0 if trajectory last occupied B before A; =1 if vice versa 

print("\ndoing simulation of steady-state paths... (",n_traj," trajectories )\n")
i = 0
# draw initial state from p0 (formal steady-state path ensemble comprises a single infinitely long trajectory, so this choice is arbitrary if n_traj -> Inf)
n0 = np.random.choice(states,1,p=p0)[0] # note that, for implementation purposes, we are forcing start state to be in set B
v_tp_ss[n0-1] = 1
while (i<n_traj):
    n0 = np.random.choice(states,1,p=P[n0-1,:])[0] # draw next state from current row of P
    if (in_B[n0-1]): # gone back to initial state; reset reactive visit flags
        from_A = 0
        v_tp_ss = np.zeros(N)
        v_tp_Z_ss = 0
    if (not from_A): # only flag visitation if trajectory last visited B before A
        v_tp_ss[n0-1] = 1
        if (in_Z[n0-1]): v_tp_Z_ss = 1
    if (in_A[n0-1] and (not from_A)): # trajectory has first hit A, after last visiting B, before A: accumulate counts and reset reactive visit flags
        from_A = 1
        r_tp_ss += v_tp_ss
        r_tp_Z_ss += v_tp_Z_ss
        v_tp_ss = np.zeros(N)
        v_tp_Z_ss = 0
        i += 1

r_tp_ss *= 1./float(n_traj)
r_tp_Z_ss *= 1./float(n_traj)

print("probability that Z is visited on steady-state transition paths (simulation):\n",r_tp_Z_ss,"\n")


### calculate reactive visitation probability by state reduction

r_rc_gt = dict.fromkeys(nodes_B,0.) # probability of visiting Z on nonequilibrium transition paths (from specific initial nodes)
r_rc_gt_ss = 0.
for b in nodes_B:
    Pc_rc = copy.deepcopy(Pc)
    # eliminate all nodes in B \ b
    for e in nodes_B:
        if e==b: continue
        # trick to keep numerical stability
        Pc_rc[e-1,e-1] = 0.
        fac = np.sum(Pc_rc[e-1,:]) # equivalent to 1-P_ee
        # renormalization
        for i in states:
            for j in states: 
                Pc_rc[i-1,j-1] += Pc_rc[i-1,e-1]*Pc_rc[e-1,j-1]/fac
        Pc_rc[e-1,:] = np.zeros(n)
        Pc_rc[:,e-1] = np.zeros(n)
#    break
    p_bA_Z = 0. # probability of transitions from b to A via Z
    Pc_rc_b = copy.deepcopy(Pc_rc) # will become renormalized chain comprising only nodes in A \cup b
    # eliminate all nodes in Z
    for z in nodes_Z:
        # trick to keep numerical stability
        Pc_rc_b[z-1,z-1] = 0.
        fac = np.sum(Pc_rc_b[z-1,:]) # equivalent to 1-P_zz
        # renormalization
        for i in states:
            for j in states:
                tij = Pc_rc_b[i-1,z-1]*Pc_rc_b[z-1,j-1]/fac
                if i==b and in_A[j-1]:
                    p_bA_Z += tij
                else:
                    Pc_rc_b[i-1,j-1] += tij
        Pc_rc_b[z-1,:] = np.zeros(n)
        Pc_rc_b[:,z-1] = np.zeros(n)
    # calculate reactive visitation probability for transitions from node b
    r_rc_gt[b] = p_bA_Z/(1.-Pc_rc_b[b-1,b-1])
    r_rc_gt_ss += r_rc_gt[b]*mu0[b-1]


print("probability that Z is visited on steady-state transition paths (state reduction):\n",r_rc_gt_ss,"\n")
print("probability that Z is visited on nonequilibrium transition paths (state reduction) [for specific initial nodes]:\n",r_rc_gt,"\n")

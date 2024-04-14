'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    P = np.zeros((model.M, model.N, 4, model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            if model.TS[r, c]:
                continue
            idx_up = max(r - 1, 0) if not model.W[max(r - 1, 0), c] else r
            idx_down = min(r + 1, model.M - 1) if not model.W[min(r + 1, model.M - 1), c] else r
            idx_left = max(c - 1, 0) if not model.W[r, max(c - 1, 0)] else c
            idx_right = min(c + 1, model.N - 1) if not model.W[r, min(c + 1, model.N - 1)] else c
            for a in range(4):
                if a == 0:      # left
                    P[r, c, a, r, idx_left] += model.D[r, c, 0]
                    P[r, c, a, idx_down, c] += model.D[r, c, 1]  # counter-clockwise
                    P[r, c, a, idx_up, c] += model.D[r, c, 2]    # clockwise
                elif a == 1:    # up
                    P[r, c, a, idx_up, c] += model.D[r, c, 0]
                    P[r, c, a, r, idx_left] += model.D[r, c, 1]
                    P[r, c, a, r, idx_right] += model.D[r, c, 2]
                elif a == 2:    # right
                    P[r, c, a, r, idx_right] += model.D[r, c, 0]
                    P[r, c, a, idx_up, c] += model.D[r, c, 1]
                    P[r, c, a, idx_down, c] += model.D[r, c, 2]
                elif a == 3:    # down
                    P[r, c, a, idx_down, c] += model.D[r, c, 0]
                    P[r, c, a, r, idx_right] += model.D[r, c, 1]
                    P[r, c, a, r, idx_left] += model.D[r, c, 2]
                    
    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros((model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            max_utility = -np.inf
            for a in range(4):
                utility = np.dot(P[r, c, a].flatten(), U_current.flatten())
                max_utility = max(max_utility, utility)
            U_next[r, c] = model.R[r, c] + model.gamma * max_utility
    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    U_current = np.zeros((model.M, model.N))
    P = compute_transition(model)
    for _ in range(100):
        U_next = compute_utility(model, U_current, P)
        if np.max(np.abs(U_next - U_current)) < epsilon:
            break
        U_current = U_next
    return U_next

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    U_current = np.zeros((model.M, model.N))
    U_next    = np.zeros((model.M, model.N))
    for _ in range(10000):
        for r in range(model.M):
            for c in range(model.N):
                utility = model.R[r, c] + model.gamma * np.dot(model.FP[r, c].flatten(), U_current.flatten())
                U_next[r, c] = utility
        if np.max(np.abs(U_next - U_current)) < epsilon:
            break
        U_current = U_next.copy()
    return U_next

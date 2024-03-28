'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def sig2(x):
    '''Calculate the vector p = [1-sigmoid(x), sigmoid(x)] for scalar x'''
    sigmoid = 1 / (1 + np.exp(-x))
    return np.array([1-sigmoid, sigmoid])

def dsig2(p):
    '''Assume p=sig2(x).  Calculate the vector v such that v[i] is the derivative of p[i] with respect to x.'''
    return p[0]*p[1]*np.array([-1,1])

def Hsig2(p):
    '''Assume p=sig2(x).  Calculate the vector v such that v[i] is the second derivative of p[i] with respect to x.'''
    return p[0]*p[1]*(p[0]-p[1])*np.array([-1,1])

def symplectic_correction(partials, hessian):
    '''Calculate the symplectic correction matrix from Balduzzi et al., "The Mechanics of n-player Games," 2018.'''
    A = 0.5*(hessian-hessian.T)
    # Balduzzi et al. use sign opposite next line b/c they minimize loss instead of maximizing utility
    sgn = -np.sign(0.25*np.dot(partials,hessian.T@partials)*np.dot(A.T@partials,hessian.T@partials)+0.1)
    return sgn * A.T
    

def utility_partials(R, x):
    '''
    Calculate vector of partial derivatives of utilities with respect to logits. 
    If u[i] = sig2(x[0])@R[i,:,:]@sig2(x[1]),
    then partial[i] is the derivative of u[i] with respect to x[i].

    @param:
    R (2,2,2) - R[i,a,b] is reward to player i if player 0 plays a, player 1 plays b
    x (2) - player i plays move j with probability softmax([0,x[i]])[j]

    @return:
    partial (2) - partial[i] is the derivative of u[i] with respect to x[i].

    HINT: You may find the functions sig2 and dsig2 to be useful.
    '''
    partial = np.zeros(2)
    partial[0] = dsig2(sig2(x[0])) @ R[0,:,:] @ sig2(x[1])      # partial derivative of u[0] with respect to x[0]
    partial[1] = sig2(x[0]) @ R[1,:,:] @ dsig2(sig2(x[1]))      # partial derivative of u[1] with respect to x[1]
    
    # raise RuntimeError("You need to write this!")
    return partial

def episodic_game_gradient_ascent(init, rewards, nsteps, learningrate):
    '''
    nsteps of a 2-player, 2-action episodic game, strategies adapted using gradient ascent.

    @param:
    init (2) - intial logits for the two players
    rewards (2,2,2) - player i receives rewards[i,a,b] if player 0 plays a and player 1 plays b
    nsteps (scalar) - number of steps of gradient descent to perform
    learningrate (scalar) - learning rate

    @return:
    logits (nsteps,2) - logits of two players in each iteration of gradient descent
    utilities (nsteps,2) - utilities[t,i] is utility to player i of logits[t,:]

    Initialize: logits[0,:] = init. 
    
    Iterate: In iteration t, player 0's actions have probabilities sig2(logits[t,0]),
    and player 1's actions have probabilities sig2(logits[t,1]).

    The utility (expected reward) for player i is sig2(logits[t,0])@rewards[i,:,:]@sig2(logits[t,1]),
    and the next logits are logits[t+1,i] = logits[t,i] + learningrate * utility_partials(rewards, logits[t,:]).
    '''
    logits = np.zeros((nsteps,2))
    utilities = np.zeros((nsteps,2))
    logits[0,:] = init
    for t in range(nsteps-1):
        logits[t+1,:] = logits[t,:] + learningrate * utility_partials(rewards, logits[t,:])
        utilities[t,:] = sig2(logits[t,0]) @ rewards[0,:,:] @ sig2(logits[t,1]), sig2(logits[t,0]) @ rewards[1,:,:] @ sig2(logits[t,1])
    utilities[nsteps-1,:] = sig2(logits[nsteps-1,0]) @ rewards[0,:,:] @ sig2(logits[nsteps-1,1]), sig2(logits[nsteps-1,0]) @ rewards[1,:,:] @ sig2(logits[nsteps-1,1])
    
    # raise RuntimeError("You need to write this!")            
    return logits, utilities
    
def utility_hessian(R, x):
    '''
    Calculate matrix of partial second derivatives of utilities with respect to logits. 
    Define u[i] = sig2(x[0])@R[i,:,:]@sig2(x[1]),
    then hessian[i,j] is the second derivative of u[j] with respect to x[i] and x[j].

    @param:
    R (2,2,2) - R[i,a,b] is reward to player i if player 0 plays a, player 1 plays b
    x (2) - player i plays move j with probability softmax([0,x[i]])[j]

    @return:
    hessian (2) - hessian[i,j] is the second derivative of u[i] with respect to x[i] and x[j].

    HINT: You may find the functions sig2, dsig2, and Hsig2 to be useful.
    '''
    # raise RuntimeError("You need to write this!")   
    hessian = np.zeros((2,2))
    hessian[0,0] = Hsig2(sig2(x[0])) @ R[0,:,:] @ sig2(x[1])      # second partial derivative of u[0] with respect to x[0] and x[0]
    hessian[0,1] = dsig2(sig2(x[0])) @ R[0,:,:] @ dsig2(sig2(x[1]))      # second partial derivative of u[0] with respect to x[0] and x[1]
    hessian[1,0] = dsig2(sig2(x[0])) @ R[1,:,:] @ dsig2(sig2(x[1]))      # second partial derivative of u[1] with respect to x[1] and x[0]
    hessian[1,1] = sig2(x[0]) @ R[1,:,:] @ Hsig2(sig2(x[1]))      # second partial derivative of u[1] with respect to x[1] and x[1]
             
    return hessian
    
def episodic_game_corrected_ascent(init, rewards, nsteps, learningrate):
    '''
    nsteps of a 2-player, 2-action episodic game, strategies adapted using corrected ascent.

    @params:
    init (2) - intial logits for the two players
    rewards (2,2,2) - player i receives rewards[i,a,b] if player 0 plays a and player 1 plays b
    nsteps (scalar) - number of steps of gradient descent to perform
    learningrate (scalar) - learning rate

    @return:
    logits (nsteps,2) - logits of two players in each iteration of gradient descent
    utilities (nsteps,2) - utilities[t,i] is utility to player i of logits[t,:]

    Initialize: logits[0,:] = init.  

    Iterate: In iteration t, player 0's actions have probabilities sig2(logits[t,0]),
    and player 1's actions have probabilities sig2(logits[t,1]).

    The utility (expected reward) for player i is sig2(logits[t,0])@rewards[i,:,:]@sig2(logits[t,1]),
    its vector of partial derivatives is partials = utility_partials(rewards, logits[t,:]),
    its matrix of second partial derivatives is hessian = utility_hessian(rewards, logits[t,:]),
    and if t+1 is less than nsteps, the logits are updated as
    logits[t+1,i] = logits[t,i] + learningrate * (I + symplectic_correction(partials, hessian))@partials
    '''
    # raise RuntimeError("You need to write this!")     
    
    logits = np.zeros((nsteps,2))
    utilities = np.zeros((nsteps,2))
    logits[0,:] = init
    for t in range(nsteps-1):
        partials = utility_partials(rewards, logits[t,:])
        hessian = utility_hessian(rewards, logits[t,:])
        logits[t+1,:] = logits[t,:] + learningrate * (np.eye(2) + symplectic_correction(partials, hessian)) @ partials
        utilities[t,:] = sig2(logits[t,0]) @ rewards[0,:,:] @ sig2(logits[t,1]), sig2(logits[t,0]) @ rewards[1,:,:] @ sig2(logits[t,1])
    utilities[nsteps-1,:] = sig2(logits[nsteps-1,0]) @ rewards[0,:,:] @ sig2(logits[nsteps-1,1]), sig2(logits[nsteps-1,0]) @ rewards[1,:,:] @ sig2(logits[nsteps-1,1])
           
    return logits, utilities


'''
Extra Credit: define the strategy for a sequential game.

sequential_strategy[a,b] is the probability that your player will perform action 1
on the next round of play if, during the previous round of play, 
the other player performed action a, and your player performed action b.

Examples:
* If you want to always act uniformly at random, return [[0.5,0.5],[0.5,0.5]]
* If you want to always perform action 1, return [[1,1],[1,1]].
* If you want to return the other player's action (tit-for-tat), return [[0,0],[1,1]].
* If you want to repeat your own previous move, return [[0,1],[0,1]].
* If you want to repeat your last move with probability 0.8, and the other player's last move 
with probability 0.2, return [[0.0, 0.8],[0.2, 1.0]].
'''
sequential_strategy = np.array([[0.0, 0.5],
                                [0.5, 1.0]])



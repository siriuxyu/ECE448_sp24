'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    # Pmarginal = np.zeros(1)
    counts = list(range(len(texts)))
    num_count = list(range(len(counts)))
    for i in range(len(texts)):
        counts[i] = texts[i].count(word0)
    for i in range(len(counts)):
        num_count[i] = counts.count(i)

    for i in range(len(num_count)):
        if all(x == 0 for x in num_count[i:]):
            num_count = num_count[:i]
            break
    # sorted_num_count = sorted(filtered_num_count)
    Pmarginal = np.array(num_count)/sum(num_count)
    
    # raise RuntimeError("You need to write this part!")
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    
    word_count_0 = list(range(len(texts)))
    word_count_1 = list(range(len(texts)))
    num_count_0 = np.zeros(len(word_count_0))
    num_count_1 = np.zeros(len(word_count_1))
    Pcond_raw = np.zeros((len(word_count_0), len(word_count_1)))
    # computer Pcond_raw, the num of events that X0=x0 and X1=x1
    for i in range(len(texts)):
        word_count_0[i] = texts[i].count(word0)
        word_count_1[i] = texts[i].count(word1)
        Pcond_raw[word_count_0[i]][word_count_1[i]] += 1
    
    # filter out the 0s
    row_max = 0
    for i in range(len(word_count_0)):
        if ~((Pcond_raw[i,:] == 0).all()):
            if i > row_max:
                row_max = i
    Pcond_raw = Pcond_raw[:row_max+1,:]
    
    col_max = 0
    for i in range(Pcond_raw.shape[0]):
        for j in range(Pcond_raw.shape[1]):
            if Pcond_raw[i][j] != 0:
                if j > col_max:
                    col_max = j
    Pcond_raw = Pcond_raw[:,:col_max+1]
        
    Pcond = np.zeros((Pcond_raw.shape[0], Pcond_raw.shape[1]))
    for i_0 in range(Pcond_raw.shape[0]):
        row_sum = sum(Pcond_raw[i_0])
        for i_1 in range(Pcond_raw.shape[1]):
            if row_sum != 0:
                Pcond[i_0][i_1] = Pcond_raw[i_0][i_1]/sum(Pcond_raw[i_0])
            else:
                Pcond[i_0][i_1] = np.nan
            
    # raise RuntimeError("You need to write this part!")
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    Pjoint = np.zeros((Pcond.shape[0], Pcond.shape[1]))
    for i in range(Pcond.shape[0]):
        for j in range(Pcond.shape[1]):
            if Pmarginal[i] == 0:
                Pjoint[i][j] = 0
            else:
                Pjoint[i][j] = Pmarginal[i] * Pcond[i][j]  
    # raise RuntimeError("You need to write this part!")
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    mu = np.zeros(2)
    for i in range(Pjoint.shape[0]):
        for j in range(Pjoint.shape[1]):
            mu[0] += i * Pjoint[i][j]
            mu[1] += j * Pjoint[i][j]
    # raise RuntimeError("You need to write this part!")
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    Sigma = np.zeros((2,2))
    for i in range(Pjoint.shape[0]):
        for j in range(Pjoint.shape[1]):
            Sigma[0][0] += (i - mu[0])**2 * Pjoint[i][j]
            Sigma[0][1] += (i - mu[0]) * (j - mu[1]) * Pjoint[i][j]
            Sigma[1][0] = Sigma[0][1]
            Sigma[1][1] += (j - mu[1])**2 * Pjoint[i][j]
    
    # raise RuntimeError("You need to write this part!")
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    Pfunc = Counter()
    for i in range(Pjoint.shape[0]):
        for j in range(Pjoint.shape[1]):
            z = f(i, j)
            Pfunc[z] += Pjoint[i][j]
    
    # raise RuntimeError("You need to write this part!")
    return Pfunc
    

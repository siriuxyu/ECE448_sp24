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
    for i in range(len(texts)):
        word_count_0[i] = texts[i].count(word0)
        word_count_1[i] = texts[i].count(word1)
    for i in range(len(texts)):
        num_count_0[i] = word_count_0.count(i)
        num_count_1[i] = word_count_1.count(i)
    for i in range(len(num_count_0)):
        if all(x == 0 for x in num_count_0[i:]):
            num_count_0 = num_count_0[:i]
            break
    for i in range(len(num_count_1)):
        if all(x == 0 for x in num_count_1[i:]):
            num_count_1 = num_count_1[:i]
            break
    Pcond = np.zeros((len(num_count_0), len(num_count_1)))
    for i in range(len(num_count_0)):
        for j in range(len(num_count_1)):
            if num_count_0[i] == 0:
                Pcond[i][j] = np.nan
            else:
                Pcond[i][j] = num_count_1[j]/num_count_0[i]

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
    raise RuntimeError("You need to write this part!")
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    raise RuntimeError("You need to write this part!")
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
    raise RuntimeError("You need to write this part!")
    return Pfunc
    

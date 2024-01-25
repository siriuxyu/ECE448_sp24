'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    freq_list = {}
    frequency = {}
    for i in train.keys():
        freq_list[i] = []
        for j in range(len(train[i])):
            train[i][j][0] = train[i][j][0].lower()
            for k in range(len(train[i][j]) - 1):
                train[i][j][k + 1] = train[i][j][k + 1].lower()
                freq_list[i].append(train[i][j][k] + '*-*-*-*' + train[i][j][k + 1])
        frequency[i] = Counter(freq_list[i])

    return frequency
    # raise RuntimeError("You need to write this part!")

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    nonstop = {}
    for i in frequency.keys():
        nonstop[i] = Counter()
        for j in frequency[i].keys():
            if j.split('*-*-*-*')[0] not in stopwords or j.split('*-*-*-*')[1] not in stopwords:
                nonstop[i][j] = frequency[i][j]
                
    return nonstop
    # raise RuntimeError("You need to write this part!")


def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    likelihood = {}
    for i in nonstop.keys():
        likelihood[i] = {}
        # likelihood[i]['OOV'] = 0
        sum_bigram = sum(nonstop[i].values())
        for j in nonstop[i].keys():
            likelihood[i][j] = (nonstop[i][j] + smoothness) / (sum_bigram + smoothness * (len(nonstop[i].keys()) + 1))
        likelihood[i]['OOV'] = smoothness / (sum_bigram + smoothness * (len(nonstop[i].keys()) + 1))
        
    return likelihood
    
    # raise RuntimeError("You need to write this part!")

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = list(range(len(texts)))
    for i in range(len(texts)):
        likelihood_pos = np.log(prior)
        likelihood_neg = np.log(1 - prior)
        texts[i][0] = texts[i][0].lower()
        for j in range(len(texts[i]) - 1):
            texts[i][j + 1] = texts[i][j + 1].lower()
            if (texts[i][j] in stopwords) and (texts[i][j + 1] in stopwords):
                continue
            temp_key = texts[i][j] + '*-*-*-*' + texts[i][j + 1]
            if temp_key not in likelihood['pos'].keys():
                likelihood_pos += np.log(likelihood['pos']['OOV'])
            else:
                likelihood_pos += np.log(likelihood['pos'][temp_key])
            if temp_key not in likelihood['neg'].keys():
                likelihood_neg += np.log(likelihood['neg']['OOV'])
            else:
                likelihood_neg += np.log(likelihood['neg'][temp_key])
        if likelihood_pos > likelihood_neg:
            hypotheses[i] = ('pos')
        elif likelihood_pos < likelihood_neg:
            hypotheses[i] = ('neg')
        else:
            hypotheses[i] = ('undecided')
    
    return hypotheses
    # raise RuntimeError("You need to write this part!")



def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for i in range(len(priors)):
        for j in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[j])
            hypotheses = naive_bayes(texts, likelihood, priors[i])
            for k in range(len(hypotheses)):
                if hypotheses[k] == labels[k]:
                    accuracies[i,j] += 1
            accuracies[i,j] /= len(hypotheses)
    return accuracies
    # raise RuntimeError("You need to write this part!")
                          
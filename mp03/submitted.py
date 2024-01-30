'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    import utils
    word_tag_pairs = Counter()
    for i in range(len(train)):
        for j in range(len(train[i])):
            word_tag_pairs[train[i][j]] += 1
    # assign most common tag to each word
    sorted_word_tag_pairs = word_tag_pairs.most_common()
    most_common_pairs = {}
    for i in range(len(sorted_word_tag_pairs)):
        if sorted_word_tag_pairs[i][0][0] not in most_common_pairs.keys():
            most_common_pairs[sorted_word_tag_pairs[i][0][0]] = sorted_word_tag_pairs[i][0][1]
    
    # find out the most common tag
    tags = Counter()
    for i in word_tag_pairs.keys():
        tags[i[1]] += word_tag_pairs[i]
    most_common_tag = tags.most_common(1)[0][0]
    
    # assign most common tag to test data
    for i in range(len(test)):
        for j in range(len(test[i])):
            if test[i][j] in most_common_pairs.keys():
                test[i][j] = (test[i][j], most_common_pairs[test[i][j]])
            else:
                test[i][j] = (test[i][j], most_common_tag)
                
    return test
    # raise NotImplementedError("You need to write this part!")


def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tags = []
    initial_tags = []
    for i in range(len(train)):
        for j in range(len(train[i])):
            tag = train[i][j][1]
            if tag not in tags:
                tags.append(train[i][j][1])
                initial_tags.append(0)
        initial_tag = train[i][0][1]
        initial_tags[tags.index(initial_tag)] += 1
    initial_probability = np.array(initial_tags) / sum(initial_tags)
    
    # compute transition probability
    transition_probability = np.zeros((len(tags), len(tags)))
    for i in range(len(train)):
        for j in range(len(train[i]) - 1):
            transition_probability[tags.index(train[i][j][1])][tags.index(train[i][j + 1][1])] += 1

    for i in range(len(transition_probability)):
        transition_probability[i] = transition_probability[i] / sum(transition_probability[i])
    
    # compute emission probability
    smoothness = 0.0001
    emission_count = []
    emission_probability = []
    for i in range(len(tags)):
        emission_count.append(Counter())
        emission_probability.append(Counter())
    for i in range(len(train)):
        for j in range(len(train[i])):
            emission_count[tags.index(train[i][j][1])][train[i][j][0]] += 1
    for i in range(len(emission_count)):
        tag_sum = sum(emission_count[i].values())
        for j in emission_count[i].keys():
            emission_probability[i][j] = (emission_count[i][j] + smoothness) / (tag_sum + smoothness * (len(emission_count[i].keys()) + 1))
        emission_probability[i]['OOV'] = smoothness / (tag_sum + smoothness * (len(emission_count[i].keys()) + 1))
            
    # viterbi algorithm
    for i in range(len(test)):
        viterbi_matrix = np.zeros((len(tags), len(test[i])))
        backpointer = np.zeros((len(tags), len(test[i])))
        for j in range(len(test[i])):
            if j == 0:
                for k in range(len(tags)):
                    if test[i][j] in emission_probability[k].keys():
                        viterbi_matrix[k][j] = np.log(initial_probability[k]) + np.log(emission_probability[k][test[i][j]])
                    else:
                        viterbi_matrix[k][j] = np.log(initial_probability[k]) + np.log(emission_probability[k]['OOV'])
            for k in range(len(tags)):
                if test[i][j] in emission_probability[k].keys():
                    temp = np.log(viterbi_matrix[:, j - 1]) + np.log(transition_probability[:, k]) + np.log(emission_probability[k][test[i][j]])
                else:
                    temp = np.log(viterbi_matrix[:, j - 1]) + np.log(transition_probability[:, k]) + np.log(emission_probability[k]['OOV'])
                viterbi_matrix[k][j] = max(temp)
                backpointer[k][j] = np.argmax(temp)
    result = []
    for i in range(len(test)):
        temp = []
        for j in range(len(test[i])):
            temp.append((test[i][j], tags[int(backpointer[:, j][-1])]))
        result.append(temp)
    return result
    # raise NotImplementedError("You need to write this part!")


def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")




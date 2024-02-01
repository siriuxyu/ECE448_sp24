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
    smoothness = 0.0001     # laplace smoothness
    tags = []               # list of tags
    # initial_tags = []
    for i in range(len(train)):
        for j in range(len(train[i])):
            tag = train[i][j][1]        # collect all tags (no duplicates)
            if tag not in tags:
                tags.append(train[i][j][1])
                # initial_tags.append(0)
        initial_tag = train[i][0][1]
        # initial_tags[tags.index(initial_tag)] += 1
    # initial_probability = np.array(initial_tags) / sum(initial_tags)
    
    # compute transition probability
    transition_count = np.zeros((len(tags), len(tags)))
    transition_probability = np.zeros((len(tags), len(tags)))
    for i in range(len(train)):
        for j in range(len(train[i]) - 1):
            transition_count[tags.index(train[i][j][1])][tags.index(train[i][j + 1][1])] += 1

    for i in range(len(transition_probability)):    # Laplace smoothing
        tag_sum = sum(transition_count[i])
        for j in range(len(transition_probability[i])):
            transition_probability[i][j] = (transition_count[i][j] + smoothness) / (tag_sum + smoothness * len(tags))
    
    # compute emission probability
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
    result = []
    for i in range(len(test)):
        temp_result = []
        temp_col = np.zeros(len(tags))
        temp_result.append(('END', 'END'))
        viterbi_matrix = np.zeros((len(tags), len(test[i])))
        backpointer = np.zeros((len(tags), len(test[i])))
        for k in range(len(tags)):
            if tags[k] == 'START':
                viterbi_matrix[k][0] = np.log((1+smoothness)/(1+smoothness*len(tags)))
            else:
                viterbi_matrix[k][0] = np.log(smoothness/(1+smoothness*len(tags)))
        for j in range(1,len(test[i])-1):
            # initial probability
            for k in range(len(tags)):
                if test[i][j] in emission_probability[k].keys():
                    temp_col = viterbi_matrix[:, j - 1] + np.log(transition_probability[:, k]) + np.log(emission_probability[k][test[i][j]])
                else:
                    temp_col = viterbi_matrix[:, j - 1] + np.log(transition_probability[:, k]) + np.log(emission_probability[k]['OOV'])
                viterbi_matrix[k][j] = max(temp_col)
                backpointer[k][j] = np.argmax(temp_col)
        # find the most possible tag for the last word, then trace back
        
        temp_result.append((test[i][len(test[i])-2], tags[np.argmax(viterbi_matrix[:, len(test[i])-2])]))
        temp_tag_index = backpointer[np.argmax(viterbi_matrix[:, len(test[i])-2])][len(test[i])-2]
        for j in range(len(test[i])-3, 0,-1):
            temp_result.append((test[i][j], tags[int(temp_tag_index)]))
            temp_tag_index = backpointer[int(temp_tag_index)][j]
        temp_result.append((test[i][0], 'START'))
        temp_result.reverse()
        result.append(temp_result)
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
    smoothness = 0.00001     # laplace smoothness
    tags = []               # list of tags
    words_count = Counter()
    hapax_tags = {}
    # initial_tags = []
    for i in range(len(train)):
        for j in range(len(train[i])):
            tag = train[i][j][1]        # collect all tags (no duplicates)
            # collect all hapax tags
            words_count[train[i][j][0]] += 1
            if words_count[train[i][j][0]] == 1:
                hapax_tags[train[i][j][0]] = tag
            else:       # remove hapax tags
                if train[i][j][0] in hapax_tags.keys():
                    del hapax_tags[train[i][j][0]]
            if tag not in tags:
                tags.append(train[i][j][1])

    # compute P(T|hapax) for each hapax tag
    count_T_hapax = [0] * len(tags)
    for i in hapax_tags.values():
        if i in tags:
            count_T_hapax[tags.index(i)] += 1
    # laplace smoothing
    P_T_hapax = (np.array(count_T_hapax) + smoothness) / (sum(count_T_hapax) + smoothness * len(tags))
    
    # compute transition probability
    transition_count = np.zeros((len(tags), len(tags)))
    transition_probability = np.zeros((len(tags), len(tags)))
    for i in range(len(train)):
        for j in range(len(train[i]) - 1):
            transition_count[tags.index(train[i][j][1])][tags.index(train[i][j + 1][1])] += 1

    for i in range(len(transition_probability)):    # Laplace smoothing
        tag_sum = sum(transition_count[i])
        for j in range(len(transition_probability[i])):
            transition_probability[i][j] = (transition_count[i][j] + smoothness) / (tag_sum + smoothness * len(tags))
    
    # compute emission probability
    emission_count = []
    emission_probability = []
    for i in range(len(tags)):
        emission_count.append(Counter())
        emission_probability.append(Counter())
    for i in range(len(train)):
        for j in range(len(train[i])):
            emission_count[tags.index(train[i][j][1])][train[i][j][0]] += 1
    for i in range(len(emission_count)):
        temp_smoothness = smoothness * P_T_hapax[i]
        tag_sum = sum(emission_count[i].values())
        for j in emission_count[i].keys():
            emission_probability[i][j] = (emission_count[i][j] + temp_smoothness) / (tag_sum + temp_smoothness * (len(emission_count[i].keys()) + 1))
        emission_probability[i]['OOV'] = temp_smoothness / (tag_sum + temp_smoothness * (len(emission_count[i].keys()) + 1))
            
    # viterbi algorithm
    result = []
    for i in range(len(test)):
        temp_result = []
        temp_col = np.zeros(len(tags))
        temp_result.append(('END', 'END'))
        viterbi_matrix = np.zeros((len(tags), len(test[i])))
        backpointer = np.zeros((len(tags), len(test[i])))
        for k in range(len(tags)):
            if tags[k] == 'START':
                viterbi_matrix[k][0] = np.log((1+smoothness)/(1+smoothness*len(tags)))
            else:
                viterbi_matrix[k][0] = np.log(smoothness/(1+smoothness*len(tags)))
        for j in range(1,len(test[i])-1):
            # initial probability
            for k in range(len(tags)):
                if test[i][j] in emission_probability[k].keys():
                    temp_col = viterbi_matrix[:, j - 1] + np.log(transition_probability[:, k]) + np.log(emission_probability[k][test[i][j]])
                else:
                    temp_col = viterbi_matrix[:, j - 1] + np.log(transition_probability[:, k]) + np.log(emission_probability[k]['OOV'])
                viterbi_matrix[k][j] = max(temp_col)
                backpointer[k][j] = np.argmax(temp_col)
        # find the most possible tag for the last word, then trace back
        
        temp_result.append((test[i][len(test[i])-2], tags[np.argmax(viterbi_matrix[:, len(test[i])-2])]))
        temp_tag_index = backpointer[np.argmax(viterbi_matrix[:, len(test[i])-2])][len(test[i])-2]
        for j in range(len(test[i])-3, 0,-1):
            temp_result.append((test[i][j], tags[int(temp_tag_index)]))
            temp_tag_index = backpointer[int(temp_tag_index)][j]
        temp_result.append((test[i][0], 'START'))
        temp_result.reverse()
        result.append(temp_result)
    return result

    raise NotImplementedError("You need to write this part!")




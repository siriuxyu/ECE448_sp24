# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    learning_rate = 0.001
    W = np.zeros(train_set.shape[1])
    b = 0
    for i in range(max_iter):
        for j in range(train_set.shape[0]):
            # train_labels is 0 or 1
            if train_labels[j] == 1 and np.dot(W, train_set[j]) + b <= 0:
                W += learning_rate * train_set[j]
                b += learning_rate
            elif train_labels[j] == 0 and np.dot(W, train_set[j]) + b > 0:
                W -= learning_rate * train_set[j]
                b -= learning_rate

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    W, b = trainPerceptron(train_set, train_labels, max_iter)
    result = []
    for i in range(dev_set.shape[0]):
        if np.dot(W, dev_set[i]) + b > 0:
            result.append(1)
        else:
            result.append(0)
    return result




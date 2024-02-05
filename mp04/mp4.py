# MP5.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020
# Modified by Weilin Zhang (weilinz2@illinois.edu) on 10/18/2020
import sys
import argparse
import configparser
import copy
import time
import numpy as np

import reader
import perceptron as c

"""
This file contains the main application that is run for this MP.
"""
def checkIfInMargin(x,start,end,margin_size):
    ts = np.arange(0,101) * 0.01
    points = [t * np.array([start[0],start[1]]) + (1-t) * np.array([end[0],end[1]]) for t in ts]
    points = np.array(points)
    c = x
    r = margin_size
    v = end - start
    a = np.sum(v*v)
    b = 2*v.dot(start - c)
    c = start.dot(start) + c.dot(c) - 2 * start.dot(c) - r**2
    disc = b**2 - 4*a*c
    if disc < 0:
        return False,points
    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)
    if 0 <= t1 <= 1 or 0 <= t2 <= 1:
        return True,points
    return False, points


def classification_problem(batch_size):
    X = []
    Y = []
    np.random.seed(1)
    margin_size = 3
    p = np.array([0,0])
    support_vector1 = np.array([5,5])
    support_vector2 = np.array([-5,-5])
    X = np.random.uniform(-5,5,size=[batch_size, 2])
    Y = np.zeros([batch_size])
    for t in range(len(X)):
        in_margin,_ = checkIfInMargin(X[t],support_vector1, support_vector2,margin_size)
        while in_margin:
            X[t] = np.random.uniform(-5,5,size=[2])
            in_margin,_ = checkIfInMargin(X[t],support_vector1, support_vector2,margin_size)
        if X[t][1] > X[t][0]:
            Y[t] = 1
        else:
            Y[t] = 0
    _,decision_curve = checkIfInMargin(X[0], support_vector1, support_vector2,margin_size)
    return X,Y,decision_curve


def separable_test(affine=False):
    # use affine = True for bias
    X,y,decision_curve = classification_problem(batch_size=100)
    if affine:
        translation = np.array([3,-3])
        X += translation
    train_size = int(len(X) / 2)
    train_set = X[:train_size]
    test_set =   X[train_size:]
    train_labels = y[:train_size]
    test_labels =  y[train_size:]
    """
    import matplotlib.pyplot as plt
    # uncomment to see data
    pos_data = [X[i] for i in range(len(X)) if y[i] == 1]
    neg_data = [X[i] for i in range(len(X)) if y[i] == 0]
    pos_data = np.array(pos_data)
    neg_data = np.array(neg_data)
    plt.scatter(pos_data[:,0], pos_data[:,1])
    plt.scatter(neg_data[:,0], neg_data[:,1])
    plt.show()
    plt.scatter(train_set[:,0], train_set[:,1])
    plt.show()
    """
    lrate = 1
    max_iter = 10
    predicted_labels = c.classify(train_set,train_labels, test_set,lrate, max_iter)
    accuracy,f1,precision,recall = compute_accuracies(predicted_labels,test_labels)

    #print("ACCURACY:",accuracy,affine)
    thresholds = [.70,.80,.90,1.]
    score = 0
    for t in thresholds:
        if accuracy >= t:
            score += 2
    return score, accuracy

def compute_accuracies(predicted_labels,dev_labels):
    if not isinstance(predicted_labels, list):
        print('Predict labels must be list')
        assert False
    yhats = predicted_labels
    if len(yhats) != len(dev_labels):
        print('Predict labels must have the same length as actual labels!')
        assert False
    if not all([(y == 0 or y == 1) for y in yhats]):
        print('Predicted labels must only contain 0s or 1s')
        assert False
    accuracy = np.mean(yhats == dev_labels)
    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    precision = tp / np.sum([yhats[i]==1 for i in range(len(yhats))])
    recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))]) + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)

    return accuracy,f1,precision,recall

def main(args):
    # Test the perceptron on linearly separable data
    print("Testing your perceptron model on linearly separable data")
    if args.method != 'perceptron':
        print("Method must be `perceptron`")
    X, y, decision_curve = classification_problem(batch_size=100)
    affine = True  # test bias term
    if affine:
        translation = np.array([3, -3])
        X += translation
    train_size = int(len(X) / 2)
    train_set = X[:train_size]
    test_set = X[train_size:]
    train_labels = y[:train_size]
    test_labels = y[train_size:]
    lrate = 1
    max_iter = 10

    predicted_labels = c.classifyPerceptron(train_set, train_labels, test_set, max_iter)

    accuracy, f1, precision, recall = compute_accuracies(predicted_labels, test_labels)

    if accuracy < 0.99:
        print('Your accuracy: %.2f.' % (accuracy))
        print(
            'your perceptron model did not achieve perfect accuracy on linearly separable data. Please rectify your code.')
        print(f'To obtain full credit, you must achieve an accuracy of at least 0.99')
    else:
        print('Test passed!')
    print('Your accuracy: %.2f.' % (accuracy))

    # Test on original data, and calculate the running time
    print("Testing your perceptron model on the actual data")


    if args.method == 'perceptron':
        train_set, train_labels, dev_set,dev_labels = reader.load_dataset_perceptron(args.dataset_file)
        start_time = time.time()
        pred_p = c.classifyPerceptron(train_set, train_labels, dev_set, args.max_iter)
        end_time = time.time()
        print("Perceptron")
        accuracy,f1,precision,recall = compute_accuracies(pred_p, dev_labels)
    else:
        print("Method must be `perceptron`")


    print("A correct implementation of perceptron should achieve accuracy of around 0.78 to 0.80 on dev. data.")
    runtimes = {'perceptron': 1.8}

    print(f'Time taken for {args.method} to execute training and classification: {end_time - start_time}')
    print(f'Time taken for our model implementation is approximately {runtimes[args.method]}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP4 Perceptron')

    parser.add_argument('--dataset', dest='dataset_file', type=str, default = 'mp4_data',
                        help='the directory of the training data')
    parser.add_argument('--method',default="perceptron",
                        help="classification method, ['perceptron']")
    parser.add_argument('--max_iter',dest="max_iter", type=int, default = 10,
                        help='Maximum iterations - default 10')

    args = parser.parse_args()
    main(args)

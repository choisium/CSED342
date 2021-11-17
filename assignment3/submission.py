# ID: 20160169 NAME: Choi Soomin
######################################################################################

#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################


def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)

    reviews = [
        {"occurrence": {"pretty": 1, "good": 1}, "label": 1},
        {"occurrence": {"bad": 1, "plot": 1}, "label": -1},
        {"occurrence": {"not": 1, "good": 1}, "label": -1},
        {"occurrence": {"pretty": 1, "scenery": 1}, "label": 1}
    ]

    weight = {"pretty": 0, "good": 0, "bad": 0, "plot": 0, "not": 0, "scenery": 0}

    def iterate(review, weight):
        sum = 0

        occurrence = review["occurrence"]
        label = review["label"]

        new_weight = dict()
        for (key, w) in weight.items():
            if key in occurrence:
                sum += w * occurrence[key]
                new_weight[key] = w + label * occurrence[key]
            else:
                new_weight[key] = w

        loss = max(0, 1 - sum * label)
        if loss == 0:
            new_weight = weight
        
        return (loss, new_weight)
    
    converge = False
    
    while not converge:
        for review in reviews:
            (loss, new_weight) = iterate(review, weight)
            if new_weight == weight:
                converge = True
            else:
                weight = new_weight

    # print("final weight:", weight)
    # for review in reviews:
    #     (loss, new_weight) = iterate(review, weight)
    #     print("\treview:", review, "loss:", loss)

    return weight

    # END_YOUR_ANSWER


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    
    return dict(Counter(x.split()))

    # END_YOUR_ANSWER


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)

    # Loss = -log(p)
    # grad(Loss(x, y, w)) = (dLoss/dp) * (dp/dw)
    # dLoss/dp = -1/p
    # dp/dw = pi * dsigmoid(w * pi) if y = 1, -pi * dsigmoid(w * pi) if y = -1
    # dsigmoid = e^(-z)/(1+e^(-z))^2 = sigmoid * (1 - sigmoid)

    def get_p(weights, feature, label):
        s = sigmoid(dotProduct(weights, feature))
        return s if label == 1 else 1 - s
    
    def get_loss(weights, feature, label):
        return - math.log(get_p(weights, feature, label))

    def d_sigmoid(n):
        s = sigmoid(n)
        return s * (1 - s)

    def get_gradient(weights, feature, label):
        p = get_p(weights, feature, label)
        wf = dotProduct(weights, feature)
        dloss_dp = -1/p if label > 0 else 1/p
        return {word: dloss_dp * occur * d_sigmoid(wf) for (word, occur) in feature.items()}

    for i in range(numIters):
        for (data, label) in trainExamples:
            feature = featureExtractor(data)
            loss = get_loss(weights, feature, label)
            gradient = get_gradient(weights, feature, label)
            for (word, grad) in gradient.items():
                weights[word] = weights.get(word, 0) - eta * grad

            # trainError = evaluatePredictor(
            #     trainExamples,
            #     lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
            # )
            # testError = evaluatePredictor(
            #     testExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)
            # )
            # print("loss:", loss, "train error:", trainError, "test error:", testError)

    # END_YOUR_ANSWER
    return weights


############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)

    phi = dict()
    splited_x = x.split()
    last_word = "<s>"
    for word in splited_x:
        phi[word] = phi.get(word, 0) + 1
        bigram = (last_word, word)
        phi[(last_word, word)] = phi.get(bigram, 0) + 1
        last_word = word
        if (word == splited_x[-1]):
            phi[(word, "</s>")] = 1

    # END_YOUR_ANSWER
    return phi

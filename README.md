# NaiveBayesClassifier
    A simple naive bayes classifier for spam emails detection.
    
1. General idea

    The Naïve Bayes classification is based on the Bayes formula.
    In a classification problem with two classes w1 and w2. We decide according to the maximum posterior probability.
    Naïve Bayes classifier assumes the elements of feature vector x are independent.
    
    In our spam/ham classification problem, the probability of an email being a spam depends on all word in this email and the prior probability of spam emails.

2. Issues

    1. underflow
    When a variable has value under2.5e^-320 will induce a data underflow, which means the value will be 0. To avoid value underflow, we use logarithm to compute likelihood.
    
    2. Zero-probabilities
    When encounter a word that does not exist in the word bank, it will produce a probability of zero, which will result in class likelihood becoming 0. Since we calculate the likelihood of email being in one class by compute all product of the word likelihood in that class.
    To resolve this issue, we pay special attention to the words that are not present in the word bank by applying Laplace smoothing.

Dataset: https://plg.uwaterloo.ca/~gvcormac/treccorpus06 /


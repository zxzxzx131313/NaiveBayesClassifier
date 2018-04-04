import numpy as np
import math
import time
from collections import Counter
import random


class NBClassifier(object):
    def __init__(self, training_percent, common_factor):

        self.training_percent = training_percent
        self.common_factor = common_factor
        self.vocab_size = 0
        self.p_spam = 0.0
        self.p_ham = 0.0
        self.spam = Counter()
        self.ham = Counter()
        
        self.data = {}
        self.diff = []

    def collect_sets(self, lpath):
        test_list = []
        true_value = []
        all_path = []
        with open(lpath) as dire:
            for idx, line in enumerate(dire):
                all_path.append(line[:-1])

        # split the files into training set and test set
        training_size = int(len(all_path)/2 * self.training_percent)
        training_list = all_path[:len(all_path)/2]
        random.shuffle(training_list)
        training_list = training_list[:training_size]

        for line in all_path[len(all_path)/2:]:
            tokens = line.split(' ')
            test_list.append(tokens[1])
            true_value.append(tokens[0] == 'spam')

        return training_list, test_list, true_value

    def build_data(self, training_list):
        spam_lines = []
        ham_lines = []
        c_spam = 0.0
        c_ham = 0.0
        for idx, line in enumerate(training_list):
            tokens = line.split(' ')
            if tokens[0] == "spam":
                c_spam += 1.0
            else:
                c_ham += 1.0
            path = tokens[1]
            with open(path) as f:
                for idx, line in enumerate(f):
                    l = line.split(' ')
                    for word in l:
                        if word not in self.data:
                            ind = len(self.data)
                            self.data[word] = ind
                    if tokens[0] == "spam":
                        spam_lines.extend(l)
                    else:
                        ham_lines.extend(l)
        self.vocab_size = len(self.data)
        # Construct the counters for each word in spam and ham emails.
        self.spam = Counter(spam_lines)
        self.ham = Counter(ham_lines)

        # Build the most-frequent feature list.
        spam_common = self.spam.most_common(self.common_factor)
        spam_common_list = []
        self.diff = []
        for i in spam_common:
            spam_common_list.append(i[0])
        ham_common = self.ham.most_common(self.common_factor)
        ham_common_list = []
        for i in ham_common:
            ham_common_list.append(i[0])
        for i in range(self.common_factor):
            if spam_common_list[i] not in ham_common_list:
                self.diff.append(spam_common_list[i])

        self.diff = self.diff

        return c_spam, c_ham

    def cal_priors(self, c_spam, c_ham):
        total = c_spam + c_ham
        self.p_spam = float(c_spam)/total
        self.p_ham = float(c_ham) / total

    def test(self, test_sets, true_value):
        inference = []
        spam_word_size = sum(self.spam.values())
        ham_word_size = sum(self.ham.values())
        for path in test_sets:
            with open(path) as test_file:
                spam_likelihood = self.p_spam
                ham_likelihood = self.p_ham

                factor = 0
                for idx, line in enumerate(test_file):
                    words = line.split(' ')
                    for word in words:
                        if word in self.diff:
                            factor += 1
                            if factor > 10:
                                spam_likelihood = 1
                                ham_likelihood = 0
                                break
                        spam_count_w = self.spam[word]
                        #spam_likelihood += math.log10(float(spam_count_w+1.0)/(spam_word_size+self.vocab_size))
                        # with out laplace smoothing
                        if spam_count_w != 0:
                            spam_likelihood += math.log10(float(spam_count_w) /spam_word_size)

                        ham_count_w = self.ham[word]
                        #ham_likelihood += math.log10(float(ham_count_w+1.0)/(ham_word_size+self.vocab_size))
                        # with out laplace smoothing
                        if ham_count_w != 0:
                            ham_likelihood += math.log10(float(ham_count_w)/ham_word_size)

                inference.append(spam_likelihood > ham_likelihood)
        accuracy = 0.0
        for i in range(len(inference)):
            if inference[i] == true_value[i]:
                accuracy += 1

        accuracy /= len(test_sets)
        print accuracy
        return accuracy

if __name__ == '__main__':

    repeats = 5
    accuracy_list = []
    training_percent = 0.05
    # Set the number of features in pre-check list
    common_factor = 100

    for n in range(repeats):
        start = time.time()
        classifier = NBClassifier(training_percent, common_factor)

        label_path = "./trec06c-utf8/label/index-datacut"

        training_list, test_list, true_value = classifier.collect_sets(label_path)

        c_spam, c_ham = classifier.build_data(training_list)
        classifier.cal_priors(c_spam, c_ham)

        accuracy = classifier.test(test_list, true_value)
        accuracy_list.append(accuracy)

        # Create a file that stores all accuracies and means.
        with open("./results.txt", 'a') as result:
            result.write(str(accuracy)+'  ')
            result.write("\ntime: "+str(time.time()-start)+'\n')

    with open("./results.txt", 'a') as result:
        result.write('mean: \n')
        result.write(str(np.mean(accuracy_list)) + '\n')



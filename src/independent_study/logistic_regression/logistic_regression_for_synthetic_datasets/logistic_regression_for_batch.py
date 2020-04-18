import numpy as np
from src.processing.parse_data import process_data
import os
from src.independent_study.dataset_generation.synthetic_dataset import SyntheticDatasetGeneration
import math
import json

'''
    This file is for normal logistic regression
    There is no compression of gradients
    All elements considered while next step during stochastic gradient descent
 '''


class LogisticRegressionBatch(object):
    def __init__(self, examples, features, batch_size, sparsity, dataset_files_path):
        self.learning_rate = 0.5
        self.features = features
        self.examples = examples
        self.sparsity = sparsity
        self.batch_size = batch_size
        self.correctly_classified = 0
        self.samples = np.loadtxt(dataset_files_path['examples_path'], delimiter=',')
        self.weight = np.loadtxt(dataset_files_path['weights_path'], delimiter=',')
        self.true_labels = np.loadtxt(dataset_files_path['true_label_path'], delimiter=',')
        self.noisy_labels = np.loadtxt(dataset_files_path['noisy_label_path'], delimiter=',')
        self.recovered_weight = np.zeros(self.features, )
        self.non_zero_indexes = np.nonzero(self.weight)
        print("non zero indexes of weights {}".format(self.non_zero_indexes))
        self.non_zero_weights = []
        for index in self.non_zero_indexes:
            self.non_zero_weights.append(self.weight[index])
        print("non zero weights {}".format(self.non_zero_weights))

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def loss(self, y, p):
        # print(y,p)
        if p == 1:
            p = 0.999999
        if p == 0:
            p = 0.000001
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def reset_weight_sparsity(self):
        indexes = sorted(range(len(self.recovered_weight)), key=lambda i: abs(self.recovered_weight[i]), reverse=True)[
                  :self.sparsity]
        indexes = set(indexes)
        for i in range(len(self.recovered_weight)):
            if i not in indexes:
                self.recovered_weight[i] = 0
        print("Sparsity achieved", np.count_nonzero(self.recovered_weight))

    def predict(self, sample):
        logit = 0
        for i in range(len(sample)):
            logit += self.recovered_weight[i] * sample[i]
        a = self.sigmoid(logit)
        if a >= 0.5:
            return 1
        else:
            return 0

    def accuracy_on_test(self):
        print("Dataset Testing Started")
        test_labels, test_features = self.true_labels, self.samples
        for i in range(len(test_features)):
            test_example = test_features[i]
            pred_label = self.predict(test_example)
            if pred_label == test_labels[i]:
                self.correctly_classified += 1
        # print("correctly classified test examples {}".format(self.correctly_classified))
        print("Dataset Testing Done")

    def dump_top_K(self, filename):
        grad_list = []
        for i, gradient in enumerate(self.recovered_weight):
            grad_list.append((i, gradient))
        topk = sorted(grad_list, key=lambda x: abs(x[1]), reverse=True)
        with open(filename + str(datetime.datetime.now()), 'w') as f:
            f.write(json.dumps(topk[:8001]))

    def get_recovery_mse(self):
        self.weight -= np.mean(self.weight)
        self.weight /= np.std(self.weight)
        self.recovered_weight -= np.mean(self.recovered_weight)
        self.recovered_weight /= np.std(self.recovered_weight)
        zip_object = zip(self.weight, self.recovered_weight)
        difference = []
        for list1_i, list2_i in zip_object:
            difference.append(list1_i - list2_i)
        return np.mean(np.square(difference))

    def number_of_position_recovered(self):
        topk_recovered = []
        for i, item in enumerate(self.recovered_weight):
            if item != 0:
                topk_recovered.append(i)
        recovered = np.intersect1d(topk_recovered, self.non_zero_indexes[0])
        print("recovered {}".format(recovered))
        return len(recovered)

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def train_dataset(self, epochs, iterations):
        print("Dataset Training Started")
        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for iteration in self.batch(range(0, iterations), self.batch_size):
                self.train_batch(iteration)
        print("Dataset Training Done")

    def train_batch(self, iteration):
        gradient = 0
        loss = 0
        feature_values = [0] * self.features
        for example_index in iteration:
            sample, label = self.samples[example_index], self.true_labels[example_index]
            val = 0
            for i in range(len(sample)):
                val += self.recovered_weight[i] * sample[i]
                feature_values[i] += sample[i]
            sigm_val = self.sigmoid(val)
            loss += self.loss(y=label, p=sigm_val)
            gradient += (label - sigm_val)
        feature_values = [item / len(iteration) for item in feature_values]
        for i in range(self.features):
            self.recovered_weight[i] += (self.learning_rate * gradient * feature_values[i])
        return loss


if __name__ == '__main__':
    examples = 10000
    features = 10000
    sparsity = 50
    dataset_sparsity = 100
    # dataset = SyntheticDatasetGeneration(examples, features, sparsity, "../../dataset_generation/dataset/", 0)
    # read data from file
    dataset_files_path = {
        "examples_path": "../../dataset_generation/dataset/data_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity),
        "true_label_path": "../../dataset_generation/dataset/true_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity),
        "noisy_label_path": "../../dataset_generation/dataset/noisy_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity),
        "weights_path": "../../dataset_generation/dataset/weights_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity)
    }
    lgr = LogisticRegressionBatch(examples, features, 20, sparsity, dataset_files_path)
    lgr.train_dataset(1, examples)
    lgr.accuracy_on_test()
    print("recovered positions {}".format(lgr.number_of_position_recovered()))
    print("recovery mse {}".format(lgr.get_recovery_mse()))
    print("correctly classified {}".format(lgr.correctly_classified))

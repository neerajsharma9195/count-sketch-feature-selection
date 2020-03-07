import numpy as np
from numpy import random


class synthetic_dataset:
    def __init__(self, examples, features, sparsity):
        self.features = features
        self.examples = examples
        self.sparsity = sparsity
        self.samples = [[0 for i in range(features)] for j in range(examples)]
        self.weight = [0] * features
        self.true_labels = [0] * features
        self.noisy_labels = [0] * features
        self.create_dataset()
        self.create_true_labels()
        self.create_noisy_true_labels()

    def create_dataset(self):
        self.samples = np.random.random((self.examples, self.features))
        self.weight = np.random.random(self.examples)
        for i in range(self.sparsity):
            number = np.random.randint(self.features)
            self.weight[number] = 0
        # print(samples,weight)
        # Due to collisions in random number generation, we might not achieve exact sparsity
        self.sparsity = self.features - np.count_nonzero(self.weight)

    def create_noisy_true_labels(self):
        for i, example in enumerate(self.samples):
            self.noisy_labels[i] = self.adding_noise(np.dot(example, self.weight))
        #print(self.noisy_labels)

    def create_true_labels(self):
        for i, example in enumerate(self.samples):
            self.true_labels[i] = self.without_noise(np.dot(example, self.weight))
        #print(self.true_labels)

    def adding_noise(self,x):
        sigmoid_value = self.sigmoid(x)
        #print(sigmoid_value)
        return np.random.choice([0, 1], p=[1 - sigmoid_value, sigmoid_value])

    def without_noise(self,x):
        sigmoid_value = self.sigmoid(x)

        if sigmoid_value >= 0.5:
            return 1
        else:
            return 0

    def sigmoid(self,x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))


if __name__ == '__main__':
    dataset=synthetic_dataset(5,5,3)
    print(dataset.samples)
    print(dataset.true_labels)
    print(dataset.noisy_labels)
    print(dataset.weight)
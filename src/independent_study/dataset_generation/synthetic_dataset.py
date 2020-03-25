import numpy as np
from numpy import random
import json


class SyntheticDatasetGeneration:
    def __init__(self, examples, features, sparsity, data_path, dataset_sparsity):
        np.random.seed(42)
        print("examples {} features {} sparsity {}".format(examples, features, sparsity))
        self.features = features
        self.examples = examples
        self.sparsity = sparsity
        self.dataset_sparsity = dataset_sparsity
        self.samples = np.zeros((self.examples, self.features))
        self.weight = np.zeros(self.features, )
        self.true_labels = np.zeros(self.examples, )
        self.noisy_labels = np.zeros(self.examples, )
        self.create_dataset()
        self.create_true_labels()
        self.create_noisy_true_labels()
        self.data_path = data_path
        self.save_dataset()

    def create_dataset(self):
        if self.dataset_sparsity == 0:
            self.samples = np.random.randn(self.examples, self.features)
        else:
            for i in range(len(self.samples)):
                for j in range(self.dataset_sparsity):
                    number = np.random.randint(self.features)
                    self.samples[i][number]=np.random.randn()
        for i in range(self.sparsity):
            number = np.random.randint(self.features)
            self.weight[number] = np.random.randn()
        # Due to collisions in random number generation, we might not achieve exact sparsity
        self.sparsity = np.count_nonzero(self.weight)
        # print(self.samples)

    def create_noisy_true_labels(self):
        self.noisy_labels = [self.adding_noise(np.dot(example, self.weight)) for example in self.samples]

    def create_true_labels(self):
        self.true_labels = [self.without_noise(np.dot(example, self.weight)) for example in self.samples]

    def adding_noise(self, x):
        sigmoid_value = self.sigmoid(x)
        return np.random.choice([0, 1], p=[1 - sigmoid_value, sigmoid_value])

    def without_noise(self, x):
        sigmoid_value = self.sigmoid(x)
        if sigmoid_value >= 0.5:
            return 1
        else:
            return 0

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def save_dataset(self):
        print("dimensions {} {} sparsity {}".format(self.examples, self.features, self.sparsity))
        true_labels_path = self.data_path + "true_labels_dim_{}_{}_sparsity_{}.csv".format(self.examples,
                                                                                           self.features,
                                                                                           self.sparsity)
        noisy_labels_path = self.data_path + "noisy_labels_dim_{}_{}_sparsity_{}.csv".format(self.examples,
                                                                                             self.features,
                                                                                             self.sparsity)
        samples_path = self.data_path + "data_dim_{}_{}_sparsity_{}.csv".format(self.examples,
                                                                                self.features,
                                                                                self.sparsity)
        weights_path = self.data_path + "weights_dim_{}_{}_sparsity_{}.csv".format(self.examples,
                                                                                   self.features,
                                                                                   self.sparsity)
        np.savetxt(true_labels_path, np.asarray(self.true_labels), delimiter=",")
        np.savetxt(noisy_labels_path, np.asarray(self.noisy_labels), delimiter=",")
        np.savetxt(weights_path, np.asarray(self.weight), delimiter=",")
        np.savetxt(samples_path, np.asarray(self.samples), delimiter=",")


if __name__ == '__main__':
    dataset = SyntheticDatasetGeneration(10000, 10000, 25, "dataset/", 0)
    test_data_path = dataset.data_path + "data_dim_{}_{}_sparsity_{}.csv".format(dataset.examples, dataset.features, dataset.sparsity)
    data = np.loadtxt(test_data_path, delimiter=',')
    print(data.shape)
    print(type(data[0][0]))

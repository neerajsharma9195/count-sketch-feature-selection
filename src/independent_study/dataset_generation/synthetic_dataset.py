import numpy as np
import json
import random


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
        self.power_law_distribution()
        self.create_dataset()
        self.create_true_labels()
        self.create_noisy_true_labels()
        self.data_path = data_path
        self.save_dataset()

    def power_law(self, k_min, k_max, y, gamma):
        return ((k_max ** (-gamma + 1) - k_min ** (-gamma + 1)) * y + k_min ** (-gamma + 1.0)) ** (1.0 / (-gamma + 1.0))

    def power_law_distribution(self):
        nodes = 100000
        power_law_distribution = np.zeros(nodes, float)
        k_min = 1.0
        k_max = self.features * k_min
        gamma = 2.0
        for n in range(nodes):
            power_law_distribution[n] = self.power_law(k_min, k_max, np.random.uniform(0, 1), gamma)
        self.power_law_features = [int(round(item)) for item in power_law_distribution]

    def create_dataset(self):
        if self.dataset_sparsity == 0:
            self.samples = np.random.randn(self.examples, self.features)
        else:
            for i in range(len(self.samples)):
                features_poses = random.sample(self.power_law_features, self.dataset_sparsity)
                for feature_pos in features_poses:
                    self.samples[i][feature_pos] = np.random.randn()
        unique_feature_poses = list(set(self.power_law_features))
        if len(unique_feature_poses) > self.sparsity:
            weight_poses = random.sample(unique_feature_poses, self.sparsity)
            for weight_pos in weight_poses:
                self.weight[weight_pos] = np.random.randn()
        else:
            print("number of unique feature poses are {}".format(len(unique_feature_poses)))
            for weight_pos in unique_feature_poses:
                self.weight[weight_pos] = np.random.randn()
            self.sparsity = np.count_nonzero(self.weight)

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
        true_labels_path = self.data_path + "true_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            self.examples,
            self.features,
            self.sparsity, self.dataset_sparsity)
        noisy_labels_path = self.data_path + "noisy_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            self.examples,
            self.features,
            self.sparsity, self.dataset_sparsity)
        samples_path = self.data_path + "data_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(self.examples,
                                                                                                    self.features,
                                                                                                    self.sparsity,
                                                                                                    self.dataset_sparsity)
        weights_path = self.data_path + "weights_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(self.examples,
                                                                                                       self.features,
                                                                                                       self.sparsity,
                                                                                                       self.dataset_sparsity)
        np.savetxt(true_labels_path, np.asarray(self.true_labels), delimiter=",")
        np.savetxt(noisy_labels_path, np.asarray(self.noisy_labels), delimiter=",")
        np.savetxt(weights_path, np.asarray(self.weight), delimiter=",")
        np.savetxt(samples_path, np.asarray(self.samples), delimiter=",")


if __name__ == '__main__':
    dataset = SyntheticDatasetGeneration(2000, 1000, 50, "dataset/", 100)
    path = "dataset/"
    test_data_path = path + "data_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(2000, 1000, 50, 100)
    print("test data path {}".format(test_data_path))
    data = np.loadtxt(test_data_path, delimiter=',')
    print(data.shape)
    print(type(data[0][0]))

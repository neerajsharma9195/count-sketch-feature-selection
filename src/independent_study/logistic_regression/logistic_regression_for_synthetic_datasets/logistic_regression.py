import numpy as np
import math
from src.independent_study.dataset_generation.synthetic_dataset import SyntheticDatasetGeneration
import json
import datetime


class LogisticRegression(object):
    def __init__(self, examples, features, sparsity, dataset_files_path):
        self.learning_rate = 0.5
        self.features = features
        self.examples = examples
        self.sparsity = sparsity
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

    def train(self, example, label):
        val = 0
        for i in range(len(example)):
            val += self.recovered_weight[i] * example[i]
        sigm_val = self.sigmoid(val)
        # print("label {} sigmoid {}".format(label, sigm_val))
        loss = self.loss(y=label, p=sigm_val)
        gradient = (label - sigm_val)
        if gradient != 0:
            for i in range(len(example)):
                # updating the change only on previous values
                updated_val = self.learning_rate * gradient * example[i]
                self.recovered_weight[i] += updated_val
        return loss

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

    def train_dataset(self, epochs):
        train_labels, train_features = self.true_labels, self.samples
        print("Dataset Training Started")
        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for i in range(self.examples):
                # print("i {}".format(i))
                label = train_labels[i]
                example = train_features[i]
                loss = self.train(example, label)
                # print("loss {}".format(loss))
        self.reset_weight_sparsity()
        print("Dataset Training Done")

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


if __name__ == '__main__':
    examples = 12000
    features = 10000
    sparsity = 10
    dataset = SyntheticDatasetGeneration(examples, features, sparsity, "../../dataset_generation/dataset/", 0)
    # read data from file
    dataset_files_path = {
        "examples_path": "../../dataset_generation/dataset/data_dim_{}_{}_sparsity_{}.csv".format(examples, features, sparsity),
        "true_label_path": "../../dataset_generation/dataset/true_labels_dim_{}_{}_sparsity_{}.csv".format(examples, features, sparsity),
        "noisy_label_path": "../../dataset_generation/dataset/noisy_labels_dim_{}_{}_sparsity_{}.csv".format(examples, features, sparsity),
        "weights_path": "../../dataset_generation/dataset/weights_dim_{}_{}_sparsity_{}.csv".format(examples, features, sparsity)
    }
    lgr = LogisticRegression(examples, features, sparsity, dataset_files_path)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print("recovered positions {}".format(lgr.number_of_position_recovered()))
    print("recovery mse {}".format(lgr.get_recovery_mse()))
    print("correctly classified {}".format(lgr.correctly_classified))
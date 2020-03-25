import numpy as np
from src.processing.parse_data import process_data
import os
import math
import json

'''
    This file is for normal logistic regression
    There is no compression of gradients
    All elements considered while next step during stochastic gradient descent
 '''


class LogisticRegression(object):
    def __init__(self, dimensions, train_file, test_file):
        self.learning_rate = 0.5
        self.gradients = [0] * (dimensions + 1)
        self.dimensions = dimensions
        self.gradient_updates_dict = {k: [] for k in range(1, dimensions + 1)}
        self.train_file = train_file
        self.test_file = test_file
        self.correctly_classified = 0

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def loss(self, y, p):
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def train(self, feature_pos, features, label):
        val = 0
        for i in range(len(feature_pos)):
            # print("top k at pos {} value {}".format(feature_pos[i], self.top_k.get_item(feature_pos[i])))
            val += self.gradients[feature_pos[i]] * features[i]
        sigm_val = self.sigmoid(val)
        print("label {} sigmoid {}".format(label, sigm_val))
        loss = self.loss(y=label, p=sigm_val)
        gradient = (label - sigm_val)
        if gradient != 0:
            for i in range(len(feature_pos)):
                # updating the change only on previous values
                updated_val = self.learning_rate * gradient * features[i]
                self.gradient_updates_dict[feature_pos[i]].append(updated_val)
                self.gradients[feature_pos[i]] += updated_val
        return loss

    def predict(self, feature_pos, feature_val):
        logit = 0
        for i in range(len(feature_pos)):
            logit += self.gradients[feature_pos[i]] * feature_val[i]
        a = self.sigmoid(logit)
        if a >= 0.5:
            return 1
        else:
            return 0

    def processing_dataset_train(self):
        current_directory = (os.path.dirname(__file__))
        data_directory_path = os.path.join(current_directory, '../../../', 'data')
        print("data directory :", data_directory_path)
        filePath = os.path.join(data_directory_path, self.test_file)
        print("Filepath :", filePath)
        train_labels, train_features = process_data(filePath)
        print("len of labels {}".format(len(train_labels)))
        return train_labels, train_features

    def processing_dataset_test(self):
        current_directory = (os.path.dirname(__file__))
        data_directory_path = os.path.join(current_directory, '../../../', 'data')
        print("data directory :", data_directory_path)
        filePath = os.path.join(data_directory_path, self.test_file)
        print("Filepath :", filePath)
        test_filePath = os.path.join(data_directory_path, self.test_file)
        test_labels, test_features = process_data(test_filePath)
        print("Dataset Processing Done")
        return test_labels, test_features

    def train_dataset(self, epochs, iterations):
        train_labels, train_features = self.processing_dataset_train()
        print("Dataset Training Started")
        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for iteration in range(iterations):
                print("i {}".format(iteration))
                random_index = np.random.randint(len(train_labels))
                label = train_labels[random_index]
                label = (1 + label) / 2
                example_features = train_features[random_index]
                feature_pos = [item[0] for item in example_features]
                feature_vals = [item[1] for item in example_features]
                loss = self.train(feature_pos, feature_vals, label)
                print("loss {}".format(loss))
        print("Dataset Training Done")

    def accuracy_on_test(self):
        print("Dataset Testing Started")
        test_labels, test_features = self.processing_dataset_test()
        for i in range(len(test_labels)):
            print("{} test example".format(i))
            true_label = int((test_labels[i] + 1) / 2)
            test_example = test_features[i]
            feature_pos = [item[0] for item in test_example]
            feature_vals = [item[1] for item in test_example]
            pred_label = self.predict(feature_pos, feature_vals)
            if pred_label == true_label:
                self.correctly_classified += 1
        print("correctly classified test examples {}".format(self.correctly_classified))
        print("Dataset Testing Done")

    def dump_top_K(self, filename):
        grad_list = []
        for i, gradient in enumerate(self.gradients):
            grad_list.append((i, gradient))
        topk = sorted(grad_list, key=lambda x: abs(x[1]), reverse=True)
        dict_topK = {}
        for i, num in enumerate(topk[:8001]):
            dict_topK[num[0]] = num[1]
        with open(filename + ".json", 'w') as f:
            f.write(json.dumps(dict_topK))

    def dump_gradient_updates(self, filename):
        with open(filename + ".json", 'w') as f:
            f.write(json.dumps(self.gradient_updates_dict))


if __name__ == '__main__':
    lgr = LogisticRegression(47326, "rcv1_train.binary", "rcv1_test.binary")
    lgr.train_dataset(1, 20000)
    lgr.accuracy_on_test()
    # lgr.dump_top_K('../../dumps/normal_logistic/topk_logistic_regression')
    # lgr.dump_gradient_updates("../../dumps/normal_logistic/logistic_regression_gradient_updates")

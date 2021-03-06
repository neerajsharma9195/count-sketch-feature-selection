import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.sketches.custom_count_sketch import CustomCountSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
from guppy import hpy
import time


class LogisticRegression(object):
    def __init__(self, num_features):
        self.D = num_features
        self.learning_rate = 0.5
        # self.cms = CustomCountSketch(3, int(np.log(self.D) ** 2 / 3))
        self.cms = CustomCountSketch(3, (1 << 18) - 1)
        self.top_k = TopK(1 << 14 - 1)

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def loss(self, y, p):
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def train_with_sketch(self, feature_pos, features, label):
        logit = 0
        min_logit = float("inf")
        max_logit = float("-inf")
        for i in range(len(feature_pos)):
            # print("top k at pos {} value {}".format(feature_pos[i], self.top_k.get_item(feature_pos[i])))
            val = self.top_k.get_value_for_key(feature_pos[i]) * features[i]
            if val > max_logit:
                max_logit = val
            if val < min_logit:
                min_logit = val
            logit += val
        if max_logit - min_logit == 0:
            max_logit = 1
            min_logit = 0
        normalized_weights = (logit - min_logit) / (max_logit - min_logit)
        print("normalized weights {}".format(normalized_weights))
        sigm_val = self.sigmoid(normalized_weights)
        if sigm_val == 1.0:
            sigm_val = sigm_val - (1e-5)
        print("label {} sigmoid {}".format(label, sigm_val))
        loss = self.loss(y=label, p=sigm_val)
        gradient = (label - sigm_val)
        if gradient != 0:
            for i in range(len(feature_pos)):
                # updating the change only on previous values
                updated_val = self.learning_rate * gradient * features[i]
                value = self.cms.update(feature_pos[i], updated_val)
                self.top_k.push(Node(feature_pos[i], value))
        return loss

    def predict(self, feature_pos, feature_val):
        logit = 0
        for i in range(len(feature_pos)):
            logit += self.top_k.get_value_for_key(feature_pos[i]) * feature_val[i]
        a = self.sigmoid(logit)
        if a > 0.5:
            return 1
        else:
            return 0


if __name__ == '__main__':
    h = hpy()
    current_directory = (os.path.dirname(__file__))
    data_directory_path = os.path.join(current_directory, '..', 'data')
    fileName = "rcv1_train.binary"
    filePath = os.path.join(data_directory_path, fileName)
    labels, features = process_data(filePath)
    D = 47236
    lgr = LogisticRegression(num_features=D)
    print("len of labels {}".format(len(labels)))
    start_time = time.time()
    for epoch in range(0, 1):
        print("epoch {}".format(epoch))
        for i in range(len(labels)):
            print("i {}".format(i))
            label = labels[i]
            label = (1 + label) / 2
            example_features = features[i]
            feature_pos = [item[0] for item in example_features]
            feature_vals = [item[1] for item in example_features]
            loss = lgr.train_with_sketch(feature_pos, feature_vals, label)
            print("loss {}".format(loss))
    end_time = time.time()
    correct = 0
    test_fileName = "rcv1_test.binary"
    test_filePath = os.path.join(data_directory_path, test_fileName)
    test_labels, test_features = process_data(test_filePath)
    print("test labels size {}".format(len(test_labels)))
    for i in range(len(test_labels)):
        print("{} test example".format(i))
        true_label = int((test_labels[i] + 1) / 2)
        test_example = test_features[i]
        feature_pos = [item[0] for item in test_example]
        feature_vals = [item[1] for item in test_example]
        pred_label = lgr.predict(feature_pos, feature_vals)
        if pred_label == true_label:
            correct += 1
    print("correctly classified test examples {}".format(correct))
    x = h.heap()
    print("total memory {}".format(x.size))
    print("total time taken {}".format(end_time - start_time))

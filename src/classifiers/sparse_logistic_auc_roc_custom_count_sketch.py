import numpy as np
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
from sklearn.metrics import roc_auc_score
from src.sketches.custom_count_sketch import CustomCountSketch
import json

np.random.seed(42)


class LogisticRegression(object):
    def __init__(self, num_features, top_k_size):
        self.D = num_features
        self.w = np.array([0] * self.D)
        self.b = 0
        self.learning_rate = 5e-1
        self.cms = CustomCountSketch(3, (1 << 18) - 1)
        # self.cms = CustomCountMinSketch(2, (1<<15) - 1)
        self.top_k = TopK(top_k_size)
        self.loss_val = 0

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def loss(self, y, p):
        return - (y * math.log(p) + (1 - y) * math.log(1 - p))

    def train_with_sketch(self, feature_pos, features, label):
        logit = 0
        min_logit = float("inf")
        max_logit = float("-inf")
        for i in range(len(feature_pos)):
            # print("top k at pos {} value {}".format(feature_pos[i], self.top_k.get_item(feature_pos[i])))
            # multiplying w[i] with x[i]
            val = self.top_k.get_value_for_key(feature_pos[i]) * features[i]
            if val > max_logit:
                max_logit = val
            if val < min_logit:
                min_logit = val
            # calculating wTx
            logit += val
        if max_logit - min_logit == 0:
            max_logit = 1
            min_logit = 0
        normalized_weights = (logit - min_logit) / (max_logit - min_logit)
        sigm_val = self.sigmoid(normalized_weights)
        if sigm_val == 1.0:
            sigm_val = sigm_val - (1e-5)
        # print("label {} sigmoid {}".format(label, sigm_val))
        gradient = (label - sigm_val)
        loss = self.loss(y=label, p=sigm_val)
        self.loss_val += loss
        if gradient != 0:
            for i in range(len(feature_pos)):
                updated_val = self.learning_rate * gradient * features[i]
                value = self.cms.update(feature_pos[i], updated_val)
                self.top_k.push(Node(feature_pos[i], value))
        return loss

    def negative_log_likelihood(self, y, x):
        return - y * x / (1 + math.exp(y))

    def predict(self, feature_pos, feature_val):
        logit = 0
        for i in range(len(feature_pos)):
            logit += self.top_k.get_value_for_key(feature_pos[i]) * feature_val[i]
        a = self.sigmoid(logit)
        return a


if __name__ == '__main__':
    current_directory = (os.path.dirname(__file__))
    data_directory_path = os.path.join(current_directory, '..', 'data')
    fileName = "rcv1_train.binary"
    filePath = os.path.join(data_directory_path, fileName)
    train_labels, train_features = process_data(filePath)
    test_fileName = "rcv1_test.binary"
    test_filePath = os.path.join(data_directory_path, test_fileName)
    test_labels, test_features = process_data(test_filePath)
    D = 47236
    feature_nums = [100*i for i in range(1, 201, 5)]
    #feature_nums = [10000]
    true_test_labels = [int((i + 1) / 2) for i in test_labels]
    roc_score_list = []
    for n_feature in feature_nums:
        print("top k size {}".format(n_feature))
        lgr = LogisticRegression(num_features=D, top_k_size=n_feature)
        test_predicted_label = []
        for epoch in range(len(train_labels)):
            print("i {}".format(epoch))
            label = train_labels[epoch]
            label = (1 + label) / 2
            example_features = train_features[epoch]
            feature_pos = [item[0] for item in example_features]
            feature_vals = [item[1] for item in example_features]
            loss = lgr.train_with_sketch(feature_pos, feature_vals, label)
            # print("loss {}".format(loss))
        for i in range(len(test_labels)):
            print("{} th test examples".format(i))
            test_example = test_features[i]
            test_feature_pos = [item[0] for item in test_example]
            test_feature_vals = [item[1] for item in test_example]
            predicted_label = lgr.predict(test_feature_pos, test_feature_vals)
            test_predicted_label.append(predicted_label)
        roc_score = roc_auc_score(true_test_labels, test_predicted_label)
        print("roc score {}".format(roc_score))
        roc_score_list.append(roc_score)
    with open("roc_score_custom_count_sketch_for_all_test_data.json", 'w') as f:
        f.write(json.dumps(roc_score_list))
    with open("features_list.json", 'w') as f:
        f.write(json.dumps(feature_nums))


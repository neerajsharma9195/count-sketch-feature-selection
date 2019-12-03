import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.sketches.count_sketch import CountSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
from sklearn.metrics import roc_auc_score
from src.sketches.custom_count_min_sketch import CustomCountMinSketch
import json

np.random.seed(42)

class LogisticRegression(object):
    def __init__(self, num_features, top_k_size):
        self.D = num_features
        self.w = np.array([0] * self.D)
        self.b = 0
        self.learning_rate = 5e-1
        #self.cms = CountSketch(3, int(np.log(self.D) ** 2 / 3))
        self.cms = CustomCountMinSketch(2, (1<<15) - 1)
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
        print("number of features {}".format(len([i for i in range(0, len(features)) if features[i] > 0])))
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
        print("normalized weights {}".format(normalized_weights))
        sigm_val = self.sigmoid(normalized_weights)
        if sigm_val == 1.0:
            sigm_val = sigm_val - (1e-5)
        print("label {} sigmoid {}".format(label, sigm_val))
        gradient = (label - sigm_val)
        loss = self.loss(y=label, p=sigm_val)
        self.loss_val += loss
        if gradient != 0:
            for i in range(len(feature_pos)):
                # updating the change only on previous values
                # if features[i] != 0 :
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
    labels, features = process_data(filePath)
    D = 47236
    feature_nums = [100*i for i in range(1, 201, 5)]
    train_example_indexes = [np.random.choice(20242) for i in range(20242)]
    train_example_labels = [labels[i] for i in train_example_indexes]
    test_examples_indexes = [i for i in range(15000)]
    test_labels = [labels[i] for i in test_examples_indexes]
    true_test_labels = [int((i + 1) / 2) for i in test_labels]
    roc_score_list = []
    for n_feature in feature_nums:
        lgr = LogisticRegression(num_features=D, top_k_size=n_feature)
        test_predicted_label = []
        for epoch in range(len(train_example_indexes)):
            print("i {}".format(epoch))
            label = train_example_labels[epoch]
            label = (1 + label) / 2
            example_features = features[train_example_indexes[epoch]]
            feature_pos = [item[0] for item in example_features]
            feature_vals = [item[1] for item in example_features]
            loss = lgr.train_with_sketch(feature_pos, feature_vals, label)
            print("loss {}".format(loss))
        for i in range(len(test_labels)):
            test_example = features[test_examples_indexes[i]]
            test_feature_pos = [item[0] for item in test_example]
            test_feature_vals = [item[1] for item in test_example]
            predicted_label = lgr.predict(test_feature_pos, test_feature_vals)
            test_predicted_label.append(predicted_label)
        roc_score = roc_auc_score(true_test_labels, test_predicted_label)
        roc_score_list.append(roc_score)
    with open("roc_score_custom_cms_2_hash_1_15.json", 'w') as f:
        f.write(json.dumps(roc_score_list))
    with open("features_list.json", 'w') as f:
        f.write(json.dumps(feature_nums))




    #     print("total loss after epoch {} is {}".format(i, lgr.loss_val))
    # # # test_fileName = "rcv1_test.binary"
    # # # test_filePath = os.path.join(data_directory_path, test_fileName)
    # # # test_labels, test_features = process_data(test_filePath)
    # # # print("test labels {}".format(test_labels))
    # print("printing heap")
    # lgr.top_k.print_heap()
    # with open("topk_results.txt", 'w') as f:
    #     for item in lgr.top_k.heap:
    #         f.write("{}:{}\n".format(item.key, item.value))
    # correct = 0
    # for i in range(1000):
    #     true_label = int((labels[i] + 1) / 2)
    #     test_example = features[i]
    #     feature_pos = [item[0] for item in test_example]
    #     feature_vals = [item[1] for item in test_example]
    #     pred_label = lgr.predict(feature_pos, feature_vals)
    #     if pred_label == true_label:
    #         correct += 1
    # print("correctly classified test examples {}".format(correct))


    # n, d = X.shape
    # logistic = LogisticRegression(n, d)
    # print(X.shape, y.shape)
    # for i in range(n):
    #     logistic.train(X[i], y[i])
    # output = [logistic.predict(X[i]) for i in range(n)]
    # print(sum(output == y) / n)
    # n_t, d_t = X_test.shape
    # output = [logistic.predict(X_test[i]) for i in range(n_t)]
    # print(sum(output == y_test) / n_t)

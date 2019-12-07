import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# from src.sketches.count_sketch import CountSketch
# from src.sketches.custom_count_sketch import CustomCountSketch
from src.sketches.custom_count_min_sketch import CustomCountMinSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
import json


class LogisticRegression(object):
    def __init__(self, num_features, top_k_size, learning_rate):
        self.D = num_features
        self.w = np.array([0] * self.D)
        self.b = 0
        self.learning_rate = learning_rate
        self.cms = CustomCountMinSketch(2, (1 << 15) - 1)
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
        # print("learning rate {}".format(self.learning_rate))
        # print("number of features {}".format(len([i for i in range(0, len(features)) if features[i] > 0])))
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
        # print("normalized weights {}".format(normalized_weights))
        sigm_val = self.sigmoid(normalized_weights)
        if sigm_val == 1.0:
            sigm_val = sigm_val - (1e-5)
        # print("label {} sigmoid {}".format(label, sigm_val))
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
        if a > 0.5:
            return 1
        else:
            return 0


if __name__ == '__main__':
    current_directory = (os.path.dirname(__file__))
    data_directory_path = os.path.join(current_directory, '..', 'data')
    fileName = "rcv1_train.binary"
    filePath = os.path.join(data_directory_path, fileName)
    labels, features = process_data(filePath)
    test_fileName = "rcv1_test.binary"
    test_filePath = os.path.join(data_directory_path, test_fileName)
    test_labels, test_features = process_data(test_filePath)
    validation_labels = test_labels[:20000]
    validation_features = test_features[:20000]
    test_test_labels = test_labels[20000:]
    test_test_features = test_features[20000:]
    D = 47236
    feature_nums = [100 * i for i in range(1, 201, 5)]
    # feature_nums = [100, 5000]
    roc_score_list = []
    lr = [1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
    #lr = [1e-1, 2e-1]
    output_dict = {}
    #random_seeds = [47, 95]
    for j, learning_rate in enumerate(lr):
        print("learning rate {}".format(learning_rate))
        output_dict[str(learning_rate)] = []
        # print("random seed at j {} is {}".format(j, random_seeds[j]))
        # np.random.seed(random_seeds[j])
        for n_feature in feature_nums:
            print("top k size {} lr {}".format(n_feature, learning_rate))
            lgr = LogisticRegression(num_features=D, top_k_size=n_feature, learning_rate=learning_rate)
            for epoch in range(0, 1):
                random_index = [np.random.randint(len(labels)) for i in range(len(labels))]
                for i in range(len(random_index)):
                    print("i th index {}".format(i))
                    label = labels[random_index[i]]
                    label = (1 + label) / 2
                    example_features = features[random_index[i]]
                    feature_pos = [item[0] for item in example_features]
                    feature_vals = [item[1] for item in example_features]
                    loss = lgr.train_with_sketch(feature_pos, feature_vals, label)
            correct = 0
            with open("../results/topk_custom_count_min_{}_{}".format(learning_rate, n_feature) + ".txt", 'w') as f:
                for item in lgr.top_k.heap:
                    key = lgr.top_k.keys[item.value]
                    value = lgr.top_k.features[key]
                    f.write("{}:{}\n".format(key, value))
            print("test run")
            for i in range(len(validation_labels)):
                # print("{}th validation".format(i))
                true_label = int((validation_labels[i] + 1) / 2)
                test_example = validation_features[i]
                feature_pos = [item[0] for item in test_example]
                feature_vals = [item[1] for item in test_example]
                pred_label = lgr.predict(feature_pos, feature_vals)
                if pred_label == true_label:
                    correct += 1
            print("correctly classified examples {}".format(correct))
            output_dict[str(learning_rate)].append((n_feature, correct))
            print("output dict {}".format(output_dict))
    print("final output dict {}".format(output_dict))
    with open("hyper_params_custom_count_min_sketch.json", 'w') as f:
        f.write(json.dumps(output_dict))

import numpy as np
from src.sketches.custom_count_sketch import CustomCountSketch
from src.processing.parse_data import process_line
import os
import math
from src.sketches.top_k import TopK, Node
from guppy import hpy
import time
import json


class LogisticRegression(object):
    def __init__(self, num_features):
        self.learning_rate = 5e-1
        self.cms = CustomCountSketch(3, (1 << 18) - 1)
        self.top_k = TopK(num_features)
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
        # if sigm_val == 1.0:
        #     sigm_val = sigm_val - (1e-5)
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
    fileName = "kdd12.tr"
    filePath = os.path.join(data_directory_path, fileName)
    D = 54686452
    top_k_size = [1000*i for i in range(10, 101, 5)]
    output_results = {}
    for top_k in top_k_size:
        h = hpy()
        train_i = 0
        lgr = LogisticRegression(num_features=top_k)
        start_time = time.time()
        with open(filePath, 'r') as f:
            for line in f:
                print("train i {}".format(train_i))
                label, feature = process_line(line)
                feature = [(int(item[0]), float(item[1])) for item in feature]
                feature_pos = [item[0] for item in feature]
                feature_vals = [item[1] for item in feature]
                loss = lgr.train_with_sketch(feature_pos, feature_vals, int(label))
                print("loss {}".format(loss))
                train_i += 1
        end_time = time.time()
        with open("topk_kdd_custom_count_sketch_size_{}.txt".format(top_k), 'w') as f:
            for item in lgr.top_k.heap:
                key = lgr.top_k.keys[item.value]
                value = lgr.top_k.features[key]
                f.write("{}:{}\n".format(key, value))
        test_fileName = "kdd12.val"
        test_filePath = os.path.join(data_directory_path, test_fileName)
        correct = 0
        print("now will do testing on validation set")
        i = 0
        with open(test_filePath, 'r') as f:
            for line in f:
                print("test i {}".format(i))
                test_label, test_feature = process_line(line)
                test_feature = [(int(item[0]), float(item[1])) for item in test_feature]
                test_feat_pos = [item[0] for item in test_feature]
                test_feat_vals = [item[1] for item in test_feature]
                pred_label = lgr.predict(test_feat_pos, test_feat_vals)
                if pred_label == int(test_label):
                    correct += 1
                i += 1
        x = h.heap()
        output_results[str(top_k)] = {"correct": correct, "time_taken": end_time-start_time, "memory_usage": x.size}
    with open("kdd_custom_count_sketch_results.json", 'w') as result_file:
        result_file.write(json.dumps(output_results))
    print("output results {}".format(output_results))


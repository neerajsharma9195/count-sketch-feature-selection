import numpy as np
from src.sketches.custom_count_min_sketch import CustomCountMinSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
import time
from src.independent_study.utils import get_top_k_positions
import json


class LogisticRegression(object):
    def __init__(self, count_sketch_size, top_k, feature_size):
        self.learning_rate = 0.5
        self.cms = CustomCountMinSketch(2, count_sketch_size)
        self.top_k = TopK(top_k)
        self.recovered_weight_vector = [0] * feature_size

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
        print("label {} sigmoid {}".format(label, sigm_val))
        loss = self.loss(y=label, p=sigm_val)
        diff_label = (label - sigm_val)  # difference in label
        if diff_label != 0:
            for i in range(len(feature_pos)):
                # updating the change only on previous values
                grad_update = self.learning_rate * diff_label * features[i]
                value = self.cms.update(feature_pos[i], grad_update)
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

    def sparse_recovery(self, feature_pos, feature_vals, label):
        for i in range(len(feature_pos)):
            cumulative_grad_val = self.cms.query(feature_pos[i])
            self.recovered_weight_vector[feature_pos[i]-1] += cumulative_grad_val / feature_vals[i]


if __name__ == '__main__':
    current_directory = (os.path.dirname(__file__))
    data_directory_path = os.path.join(current_directory, '..', 'data')
    fileName = "rcv1_train.binary"
    filePath = os.path.join(data_directory_path, fileName)
    labels, features = process_data(filePath)
    print("len of labels {}".format(len(labels)))
    test_fileName = "rcv1_test.binary"
    test_filePath = os.path.join(data_directory_path, test_fileName)
    test_labels, test_features = process_data(test_filePath)
    print("test labels size {}".format(len(test_labels)))
    count_sketch_size = 11000
    print("len of labels {}".format(len(labels)))
    correctly_classified_examples = []
    D = 47236
    time_taken = []
    # top_k_file_path = "/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/project_repo/src/independent_study/topk_features_2000.txt"
    # top_k_positions = get_top_k_positions(top_k_file_path)
    top_k_dict = {k: [] for k in range(D)}
    lgr = LogisticRegression(count_sketch_size=count_sketch_size, top_k=2000, feature_size=D)
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
    with open("positive_sketch.json", 'w') as f:
        f.write(json.dumps(lgr.cms.countSketchPos))
    with open("negative_sketch.json", 'w') as f:
        f.write(json.dumps(lgr.cms.countSketchNeg))
    with open('results/topk_features_for_recovery_size_2000.txt', 'w') as f:
        for item in lgr.top_k.heap:
            key = lgr.top_k.keys[item.value]
            value = lgr.top_k.features[key]
            f.write("{}:{}\n".format(key, value))
    for i in range(len(labels)):
        print("i {}".format(i))
        label = labels[i]
        label = (1 + label) / 2
        example_features = features[i]
        feature_pos = [item[0] for item in example_features]
        feature_vals = [item[1] for item in example_features]
        lgr.sparse_recovery(feature_pos, feature_vals, label)
    with open("recovered_weight_vectors.txt", 'w') as f:
        f.write(json.dumps(lgr.recovered_weight_vector))

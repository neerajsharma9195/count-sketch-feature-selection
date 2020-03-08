import numpy as np
from src.sketches.custom_count_min_sketch import CustomCountMinSketch
from src.sketches.conservative_count_min_sketch import ConservativeCountMinSketch
from src.sketches.count_sketch import CountSketch
from src.sketches.custom_count_sketch import CustomCountSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
import time
from src.independent_study.utils import get_top_k_positions
import json


class LogisticRegression(object):
    cms_dicts = {
        "complementary_cms": CustomCountMinSketch,
        "complementary_cms_conservative": ConservativeCountMinSketch,
        "mission_count_sketch": CountSketch,
        "conservative_count_sketch": CustomCountSketch
    }

    def __init__(self, cms_type, hash_func_counts, count_sketch_size, top_k, top_k_dict={}):
        self.learning_rate = 0.5
        self.cms = self.cms_dicts[cms_type](hash_func_counts, count_sketch_size)
        self.top_k = TopK(top_k)
        self.top_k_dict = top_k_dict

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def loss(self, y, p):
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def train_with_sketch(self, feature_pos, features, label):
        logit = 0
        for i in range(len(feature_pos)):
            # print("top k at pos {} value {}".format(feature_pos[i], self.top_k.get_item(feature_pos[i])))
            val = self.top_k.get_value_for_key(feature_pos[i]) * features[i]
            logit += val
        sigm_val = self.sigmoid(logit)
        loss = self.loss(y=label, p=sigm_val)
        diff_label = (label - sigm_val)  # difference in label
        if diff_label != 0:
            for i in range(len(feature_pos)):
                # updating the change only on previous values
                grad_update = self.learning_rate * diff_label * features[i]
                if feature_pos[i] in self.top_k_dict.keys():
                    self.top_k_dict[feature_pos[i]].append(grad_update)
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


if __name__ == '__main__':
    current_directory = (os.path.dirname(__file__))
    data_directory_path = os.path.join(current_directory, '../../', 'data')
    fileName = "rcv1_train.binary"
    filePath = os.path.join(data_directory_path, fileName)
    labels, features = process_data(filePath)
    print("len of labels {}".format(len(labels)))
    test_fileName = "rcv1_test.binary"
    test_filePath = os.path.join(data_directory_path, test_fileName)
    test_labels, test_features = process_data(test_filePath)
    print("test labels size {}".format(len(test_labels)))
    print("len of labels {}".format(len(labels)))
    correctly_classified_examples = []
    D = 47236
    time_taken = []
    # top_k_file_path = "/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/project_repo/src/independent_study/topk_features_2000.txt"
    # top_k_positions = get_top_k_positions(top_k_file_path)
    top_k_dict = {k: [] for k in range(D)}
    # model params
    top_k_size = 8000
    cms_type = "complementary_cms_conservative"
    num_hashes = 2
    count_sketch_size = 11000
    lgr = LogisticRegression(cms_type=cms_type,
                             hash_func_counts=num_hashes,
                             count_sketch_size=count_sketch_size,
                             top_k=top_k_size,
                             top_k_dict=top_k_dict)
    start_time = time.time()
    # training
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
    results_dir_path = os.path.join(current_directory, '../', 'results')
    topk_dict_filePath = os.path.join(results_dir_path,
                                      "topk_feature_gradients_{}_all_topk_{}.json".format(cms_type,
                                                                                          top_k_size))
    with open(topk_dict_filePath, 'w') as f:
        f.write(json.dumps(lgr.top_k_dict))
    end_time = time.time()
    correct = 0
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
    correctly_classified_examples.append(correct)
    print("correct examples {}".format(correctly_classified_examples))
    top_k_features_path = os.path.join(results_dir_path, "topk_features_{}_{}.txt".format(cms_type, top_k_size))
    with open(top_k_features_path, 'w') as f:
        for item in lgr.top_k.heap:
            key = lgr.top_k.keys[item.value]
            value = lgr.top_k.features[key]
            f.write("{}:{}\n".format(key, value))

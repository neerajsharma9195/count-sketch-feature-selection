import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.sketches.count_sketch import CountSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
from guppy import hpy
import time


class MissionQuadreticLoss(object):
    def __init__(self, top_k_size):
        self.learning_rate = 0.2
        self.cms = CountSketch(3, 1000)
        # self.cms = CountSketch(3, int(np.log(self.D) ** 2 / 3))
        self.top_k = TopK(top_k_size)
        self.loss_val = 0

    def train_with_sketch(self, feature_pos, features, label):
        logit = 0
        for i in range(len(feature_pos)):
            val = self.top_k.get_value_for_key(feature_pos[i]) * features[i]
            # calculating wTx
            logit += val
        # print("label {} wx {}".format(label, logit))
        gradient = (label - logit)
        print("loss {}".format(gradient))
        if gradient != 0:
            for i in range(len(feature_pos)):
                updated_val = 2 * self.learning_rate * gradient * features[i]
                value = self.cms.update(feature_pos[i], updated_val)
                self.top_k.push(Node(feature_pos[i], value))
        return gradient


if __name__ == '__main__':
    non_zero_count = 10
    n = 100
    d = 300
    A = np.random.rand(n, d)
    x = np.zeros((d, 1))
    lgr = MissionQuadreticLoss(non_zero_count)
    for j in range(non_zero_count):
        pos = np.random.randint(d)
        x[pos] = np.random.rand() * np.random.randint(10)
    true_labels = np.dot(A, x)
    start_time = time.time()
    for epoch in range(4):
        for i in range(A.shape[0]):
            data_row = A[i, :]
            true_label = true_labels[i]
            lgr.train_with_sketch([i for i in range(d)], data_row, true_label)
    non_zero_x, non_zero_y = np.where(x > 0)
    print("non zero x pos {}".format(non_zero_x))
    non_zero_values = x[x > 0]
    print("non zero values {}".format(non_zero_values))
    print("shape of true labels {}".format(true_labels.shape))
    lgr.top_k.print_heap()

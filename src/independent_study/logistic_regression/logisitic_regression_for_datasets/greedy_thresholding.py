import numpy as np
from src.processing.parse_data import process_data
from src.independent_study.logistic_regression.logisitic_regression_for_datasets.logistic_regression import LogisticRegression
import os
import math
import json
import datetime
import pandas as pd


class GreedyThresholding(LogisticRegression):
    def __init__(self, dimensions, train_file, test_file):
        super(GreedyThresholding, self).__init__(dimensions, train_file, test_file)

    def train(self, feature_pos, features, label):
        val = 0
        for i in range(len(feature_pos)):
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
            self.threshold()
        return loss

    def threshold(self):
        df = pd.DataFrame(data=self.gradients[0:])
        df = df.abs().values.argsort(0)[:-1]
        updated_gradients = [0] * (self.dimensions + 1)
        for i in range(1, 8001):
            updated_gradients[df[i][0]] = self.gradients[df[i][0]]
        self.gradients = updated_gradients



if __name__ == '__main__':
    lgr = GreedyThresholding(47326, "rcv1_train.binary", "rcv1_test.binary")
    lgr.train_dataset(1,20)
    lgr.accuracy_on_test()
    lgr.dump_top_K('../../dumps/top8000_greedy_thresholding')
    lgr.dump_gradient_updates("../../dumps/greedy_thresholding_gradient_updates")

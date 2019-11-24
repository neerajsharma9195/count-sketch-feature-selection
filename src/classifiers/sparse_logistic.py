import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.sketches.count_sketch import CountSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node
import heapq


class LogisticRegression(object):
    def __init__(self, num_features):
        self.D = num_features
        self.w = np.array([0] * self.D)
        self.b = 0
        self.learning_rate = 0.01
        self.cms = CountSketch(3, int(np.log(self.D) ** 2 / 3))
        self.top_k = TopK(1 << 8 - 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def loss(self, y, p):
        return y * math.log(p) + (1 - y) * math.log(1 - p)

    def train(self, X, y):
        y_hat = np.dot(X, self.w) + self.b
        loss = self.loss(y, self.sigmoid(y_hat))
        dw, db = self.gradient(self.w, y, X, self.b)
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def train_with_sketch(self, feature_pos, features, label):
        logit = 0
        for i in range(len(feature_pos)):
            logit += self.top_k.get_item(feature_pos[i]) * features[i]
        sigm_val = self.sigmoid(logit)
        loss = self.loss(y=label, p=sigm_val)
        gradient = (label - sigm_val)
        for i in range(len(feature_pos)):
            updated_val = features[i] - self.learning_rate * gradient * features[i]
            self.cms.update(feature_pos[i], updated_val)
            value = self.cms.query(feature_pos[i])
            self.top_k.push_item(Node(feature_pos[i], value))
        return loss

    def predict(self, X):
        a = self.sigmoid(np.dot(X, self.w) + self.b)
        if a > 0.5:
            return 1
        else:
            return 0

    def gradient_using_sketch(self, X):
        for i in range(self.D):
            self.cms.update(i, self.w[i])
        dw, db = self.gradient(self.w, y, X, self.b)
        for i in range(self.D):
            self.cms.update(i, dw[i])
        # todo: update in top K

    def fit(self, X, y):
        num_features = X.shape[1]
        initial_wcb = np.zeros(shape=(2 * X.shape[1] + 1,))
        params, min_val_obj, grads = fmin_l_bfgs_b(func=self.objective,
                                                   args=(X, y), x0=initial_wcb,
                                                   disp=10,
                                                   maxiter=500,
                                                   fprime=self.objective_grad)
        print("params {}".format(params))
        print("min val obj {}".format(min_val_obj))
        print("grads dict {}".format(grads))


if __name__ == '__main__':
    current_directory = (os.path.dirname(__file__))
    data_directory_path = os.path.join(current_directory, '..', 'data')
    fileName = "rcv1_train.binary"
    filePath = os.path.join(data_directory_path, fileName)
    labels, features = process_data(filePath)
    D = 47236
    lgr = LogisticRegression(num_features=D)
    print("len of labels {}".format(len(labels)))
    for i in range(len(labels)):
        label = labels[i]
        label = (1 + label) / 2
        example_features = features[i]
        print("label {}".format(label))
        feature_pos = [item[0] for item in example_features]
        feature_vals = [item[1] for item in example_features]
        loss = lgr.train_with_sketch(feature_pos, feature_vals, label)
        print("loss {}".format(loss))
    while len(lgr.top_k.heap) > 0:
        print("heap value {}".format(heapq.heappop(lgr.top_k.heap)))
    print(lgr.cms.countsketch)
    test_fileName = "rcv1_test.binary"
    test_filePath = os.path.join(data_directory_path, test_fileName)
    test_labels, test_features = process_data(test_filePath)

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

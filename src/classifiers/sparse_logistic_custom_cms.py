import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.sketches.custom_count_sketch import CustomCountSketch
from src.processing.parse_data import process_data
import os
import math
from src.sketches.top_k import TopK, Node


class LogisticRegression(object):
    def __init__(self, num_features):
        self.D = num_features
        self.w = np.array([0] * self.D)
        self.b = 0
        self.learning_rate = 0.5
        self.cms = CustomCountSketch(3, int(np.log(self.D) ** 2 / 3))
        self.top_k = TopK(1 << 14 - 1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

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
        min_logit = float("inf")
        max_logit = float("-inf")
        for i in range(len(feature_pos)):
            # print("top k at pos {} value {}".format(feature_pos[i], self.top_k.get_item(feature_pos[i])))
            val = self.top_k.get_item(feature_pos[i]) * features[i]
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
        gradient = (label - sigm_val)
        for i in range(len(feature_pos)):
            # updating the change only on previous values
            updated_val = self.learning_rate * gradient * features[i]
            value = self.cms.update(feature_pos[i], updated_val)
            self.top_k.push_item(Node(feature_pos[i], value))
        return loss

    def predict(self, feature_pos, feature_val):
        logit = 0
        for i in range(len(feature_pos)):
            logit += self.top_k.get_item(feature_pos[i]) * feature_val[i]
        a = self.sigmoid(logit)
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
    fileName = ""
    filePath = (r"C:\Users\Kanchi\Desktop\COMPSCI-689\MLProject\final_project\count-sketch-feature-selection\src\data\rcv1_train.binary")
    print(filePath)
    labels, features = process_data(filePath)
    D = 47236
    lgr = LogisticRegression(num_features=D)
    print("len of labels {}".format(len(labels)))
    for i in range(len(labels)):
        print("i {}".format(i))
        label = labels[i]
        label = (1 + label) / 2
        example_features = features[i]
        feature_pos = [item[0] for item in example_features]
        feature_vals = [item[1] for item in example_features]
        loss = lgr.train_with_sketch(feature_pos, feature_vals, label)
    # test_fileName = "rcv1_test.binary"
    # test_filePath = os.path.join(data_directory_path, test_fileName)
    # test_labels, test_features = process_data(test_filePath)
    # print("test labels {}".format(test_labels))
    correct = 0
    for i in range(850):
        true_label = int((labels[i] + 1) / 2)
        test_example = features[i]
        feature_pos = [item[0] for item in test_example]
        feature_vals = [item[1] for item in test_example]
        pred_label = lgr.predict(feature_pos, feature_vals)
        if pred_label == true_label:
            correct += 1
    print("correctly classified test examples {}".format(correct))

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

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.sketches.count_sketch import CountSketch


class LogisticRegression(object):
    def __init__(self, num_examples, num_features):
        self.N = num_examples
        self.D = num_features
        self.w = np.array([0] * self.D)
        self.b = 0
        self.learning_rate = 0.01
        self.cms = CountSketch(3, int(np.log(self.D) ** 2 / 3))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y, p):
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradient(self, w, y, x, b):
        dw = (-y * x) / (1 + np.exp(y * (np.dot(x, w) + b)))
        db = -y / (1 + np.exp(y * (np.dot(x, w) + b)))
        return dw, db

    def train(self, X, y):
        y_hat = np.dot(X, self.w) + self.b
        loss = self.loss(y, self.sigmoid(y_hat))
        dw, db = self.gradient(self.w, y, X, self.b)
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def train_with_sketch(self, X, y):
        y_hat = np.dot(X, self.w) + self.b
        loss = self.loss(y, self.sigmoid(y_hat))
        dw, db = self.gradient(self.w, y, X, self.b)
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        a = self.sigmoid(np.dot(X, self.w) + self.b)
        if a > 0.5:
            return 1
        else:
            return -1

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
    X = np.load('/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/final_project/src/data/q2_train_X.npy')
    y = np.load('/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/final_project/src/data/q2_train_y.npy')
    X_test = np.load('/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/final_project/src/data/q2_test_X.npy')
    y_test = np.load('/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/final_project/src/data/q2_test_y.npy')
    n, d = X.shape
    logistic = LogisticRegression(n, d)
    print(X.shape, y.shape)
    for i in range(n):
        logistic.train(X[i], y[i])
    output = [logistic.predict(X[i]) for i in range(n)]
    print(sum(output == y) / n)
    n_t, d_t = X_test.shape
    output = [logistic.predict(X_test[i]) for i in range(n_t)]
    print(sum(output == y_test) / n_t)

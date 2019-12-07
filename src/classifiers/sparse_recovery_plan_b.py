import numpy as np
from src.sketches.count_sketch import CountSketch
from src.classifiers.sparse_logistic_baseline import LogisticRegression


class SparseRecovery(object):
    def __init__(self, n, d, rand_count):
        self.A = np.random.randn(n, d)
        self.d = d
        # self.A = np.ones((n, d))
        self.rand_count = rand_count
        self.countsketch = CountSketch(3, 10000)

    def run(self):
        print("A {}".format(self.A))
        for i in range(1):
            self.x = np.zeros((self.d, 1))
            for j in range(self.rand_count):
                pos = np.random.randint(self.d)
                self.x[pos] = np.random.rand() * np.random.randint(10000)

            ground_truth_features = self.x
            lgr = LogisticRegression()
            for i in range(self.A.shape[0]):
                example = self.A[i,:]


            self.non_zero_values = self.x[self.x > 0]
            self.non_zero_x, self.non_zero_y = np.where(self.x > 0)
            for j in range(len(self.x)):
                values = self.A[:,j]*self.x[j]
                for k in range(0, len(values)):
                    self.countsketch.update(k, values[k])
                # self.top_k.push(Node(i, value))
            print("non zero values {}".format(self.non_zero_values))
            print("non zero x {}".format(self.non_zero_x))
            print("printing heap")
            approximate_values = []
            for k in range(len(self.x)):
                approximate_values.append(self.countsketch.query(k))
            approximate_values = np.array(approximate_values)
            print(approximate_values.argsort()[-self.rand_count:][::-1])
        # print("count sketch {}".format(self.countsketch.countsketch))


if __name__ == '__main__':
    sparse_recovery = SparseRecovery(n=100, d=2, rand_count=10)
    sparse_recovery.run()

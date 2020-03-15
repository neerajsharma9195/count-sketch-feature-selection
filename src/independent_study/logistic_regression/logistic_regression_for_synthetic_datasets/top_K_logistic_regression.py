from src.independent_study.logistic_regression.logistic_regression_for_synthetic_datasets.logistic_regression import \
    LogisticRegression
from src.sketches.top_k import TopK, Node


class TopKLogisticRegression(LogisticRegression):
    def __init__(self, examples, features, sparsity):
        super(TopKLogisticRegression, self).__init__(examples, features, sparsity)
        self.top_k = TopK(sparsity)

    def train(self, example, label):
        val = 0
        for i in range(len(example)):
            val += self.top_k.get_value_for_key(i) * example[i]
        sigm_val = self.sigmoid(val)
        # print("label {} sigmoid {}".format(label, sigm_val))
        loss = self.loss(y=label, p=sigm_val)
        diff_label = (label - sigm_val)  # difference in label
        if diff_label != 0:
            for i in range(len(example)):
                # updating the change only on previous values
                grad_update = self.learning_rate * diff_label * example[i]
                self.recovered_weight[i] += grad_update
                self.top_k.push(Node(i, self.recovered_weight[i]))
        return loss


if __name__ == '__main__':
    # lgr.dump_top_K('../../dumps/top8000_synthetic_logistic_regression.txt')
    # print(lgr.sparsity)
    # print(len(lgr.recovered_weight))
    # print(len(lgr.weight))
    # print(lgr.recovered_weight)
    # print(lgr.weight)
    lgr = TopKLogisticRegression(5000, 5000, 3)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 5)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 10)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 15)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 20)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 25)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 30)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 35)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 40)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 45)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)
    lgr = TopKLogisticRegression(5000, 5000, 50)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print(lgr.get_recovery_mse())
    print(lgr.correctly_classified)

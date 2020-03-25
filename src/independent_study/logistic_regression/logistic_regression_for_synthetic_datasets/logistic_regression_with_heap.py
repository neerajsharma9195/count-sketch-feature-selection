from src.independent_study.logistic_regression.logistic_regression_for_synthetic_datasets.logistic_regression import \
    LogisticRegression
from src.sketches.top_k import TopK, Node


class LogisticRegressionWithHeap(LogisticRegression):
    def __init__(self, examples, features, sparsity, dataset_files_path):
        super(LogisticRegressionWithHeap, self).__init__(examples, features, sparsity, dataset_files_path)
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
    examples = 12000
    features = 10000
    sparsity = 10
    dataset_files_path = {
        "examples_path": "../../dataset_generation/dataset/data_dim_{}_{}_sparsity_{}.csv".format(examples, features,
                                                                                                  sparsity),
        "true_label_path": "../../dataset_generation/dataset/true_labels_dim_{}_{}_sparsity_{}.csv".format(examples,
                                                                                                           features,
                                                                                                           sparsity),
        "noisy_label_path": "../../dataset_generation/dataset/noisy_labels_dim_{}_{}_sparsity_{}.csv".format(examples,
                                                                                                             features,
                                                                                                             sparsity),
        "weights_path": "../../dataset_generation/dataset/weights_dim_{}_{}_sparsity_{}.csv".format(examples, features,
                                                                                                    sparsity)
    }
    lgr = LogisticRegressionWithHeap(examples, features, sparsity, dataset_files_path)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print("number of data points recovered {}".format(lgr.number_of_position_recovered()))
    print("recovery mse {}".format(lgr.get_recovery_mse()))
    print("correctly classified {}".format(lgr.correctly_classified))

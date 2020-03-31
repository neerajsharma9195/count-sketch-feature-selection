from src.independent_study.logistic_regression.logistic_regression_for_synthetic_datasets.logistic_regression import \
    LogisticRegression
from src.sketches.top_k import TopK, Node
import pandas as pd


class GreedyThresholding(LogisticRegression):
    def __init__(self, examples, features, sparsity, dataset_files_path, threshold_limit):
        super(GreedyThresholding, self).__init__(examples, features, sparsity, dataset_files_path)
        self.threshold_limit = threshold_limit

    def train(self, example, label):
        val = 0
        for i in range(len(example)):
            val += self.recovered_weight[i] * example[i]
        sigm_val = self.sigmoid(val)
        loss = self.loss(y=label, p=sigm_val)
        gradient = (label - sigm_val)
        if gradient != 0:
            for i in range(len(example)):
                # updating the change only on previous values
                updated_val = self.learning_rate * gradient * example[i]
                self.recovered_weight[i] += updated_val
            self.threshold()
        return loss

    def threshold(self):
        df = pd.DataFrame(data=self.recovered_weight)
        df = df.abs().values.argsort(0)[::-1]
        updated_gradients = [0] * (self.features + 1)
        for i in range(1, self.threshold_limit):
            updated_gradients[df[i][0]] = self.recovered_weight[df[i][0]]
        self.recovered_weight = updated_gradients


if __name__ == '__main__':
    examples = 100
    features = 100
    sparsity = 5
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
    threshold = 20
    lgr = GreedyThresholding(examples, features, sparsity, dataset_files_path, threshold)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    print("number of data points recovered {}".format(lgr.number_of_position_recovered()))
    print("recovery mse {}".format(lgr.get_recovery_mse()))
    print("correctly classified {}".format(lgr.correctly_classified))

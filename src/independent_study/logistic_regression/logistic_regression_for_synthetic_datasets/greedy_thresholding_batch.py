from src.independent_study.logistic_regression.logistic_regression_for_synthetic_datasets.logistic_regression_for_batch import \
    LogisticRegressionBatch
from src.sketches.top_k import TopK, Node
import pandas as pd


class GreedyThresholdingBatch(LogisticRegressionBatch):
    def __init__(self, examples, features, sparsity, dataset_files_path, threshold_limit, batch_size):
        super(GreedyThresholdingBatch, self).__init__(examples, features, batch_size, sparsity, dataset_files_path)
        self.threshold_limit = threshold_limit

    def train_batch(self, iteration):
        gradient = 0
        loss = 0
        feature_values = [0] * self.features
        for example_index in iteration:
            sample, label = self.samples[example_index], self.true_labels[example_index]
            val = 0
            for i in range(len(sample)):
                val += self.recovered_weight[i] * sample[i]
                feature_values[i] += sample[i]
            sigm_val = self.sigmoid(val)
            loss += self.loss(y=label, p=sigm_val)
            gradient += (label - sigm_val)
        feature_values = [item / len(iteration) for item in feature_values]
        for i in range(self.features):
            self.recovered_weight[i] += (self.learning_rate * gradient * feature_values[i])
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
    examples = 10000
    features = 10000
    sparsity = 50
    dataset_sparsity = 100
    # examples = 100
    # features = 100
    # sparsity = 5
    dataset_files_path = {
        "examples_path": "../../dataset_generation/dataset/data_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity),
        "true_label_path": "../../dataset_generation/dataset/true_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity),
        "noisy_label_path": "../../dataset_generation/dataset/noisy_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity),
        "weights_path": "../../dataset_generation/dataset/weights_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features, sparsity, dataset_sparsity)
    }
    threshold = 20
    lgr = GreedyThresholdingBatch(examples, features, sparsity, dataset_files_path, threshold, batch_size=50)
    lgr.train_dataset(1, examples)
    lgr.accuracy_on_test()
    print("number of data points recovered {}".format(lgr.number_of_position_recovered()))
    print("recovery mse {}".format(lgr.get_recovery_mse()))
    print("correctly classified {}".format(lgr.correctly_classified))

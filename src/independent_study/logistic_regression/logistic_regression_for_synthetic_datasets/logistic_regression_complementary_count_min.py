import numpy as np
from src.sketches.custom_count_min_sketch import CustomCountMinSketch
from src.sketches.conservative_count_min_sketch import ConservativeCountMinSketch
from src.sketches.count_sketch import CountSketch
from src.sketches.custom_count_sketch import CustomCountSketch
import math
from src.sketches.top_k import TopK, Node
from src.independent_study.dataset_generation.synthetic_dataset import SyntheticDatasetGeneration


class LogisticRegression(object):
    cms_dicts = {
        "complementary_cms": CustomCountMinSketch,
        "complementary_cms_conservative": ConservativeCountMinSketch,
        "mission_count_sketch": CountSketch,
        "conservative_count_sketch": CustomCountSketch
    }

    def __init__(self, sparsity, cms_type, hash_func_counts, count_sketch_size, top_k, dataset_dict, top_k_dict={}):
        self.learning_rate = 0.5
        self.cms = self.cms_dicts[cms_type](hash_func_counts, count_sketch_size)
        self.top_k = TopK(top_k)
        self.top_k_dict = top_k_dict
        self.load_dataset(dataset_dict)
        self.sparsity = sparsity
        self.recovered_weight = np.zeros(self.features, )
        self.loss_val = 0
        self.correctly_classified = 0

    def load_dataset(self, dataset_dict):
        self.samples = np.loadtxt(dataset_dict['examples_path'], delimiter=',')
        self.num_data, self.features = self.samples.shape
        self.weight = np.loadtxt(dataset_dict['weights_path'], delimiter=',')
        self.true_labels = np.loadtxt(dataset_dict['true_label_path'], delimiter=',')
        self.noisy_labels = np.loadtxt(dataset_dict['noisy_label_path'], delimiter=',')

    def train_dataset(self, epochs):
        print("Dataset Training Started")
        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for i in range(self.num_data):
                print("i {}".format(i))
                label = self.true_labels[i]
                example = self.samples[i]
                loss = self.train_with_sketch(example, label)
                self.loss_val += loss
        self.reset_weight_sparsity()
        print("Dataset Training Done")

    def reset_weight_sparsity(self):
        indexes = sorted(range(len(self.recovered_weight)), key=lambda i: abs(self.recovered_weight[i]), reverse=True)[
                  :self.sparsity]
        indexes = set(indexes)
        for i in range(len(self.recovered_weight)):
            if i not in indexes:
                self.recovered_weight[i] = 0
        print("Sparsity achieved", np.count_nonzero(self.recovered_weight))

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def loss(self, y, p):
        if p == 1:
            p = 0.999999
        if p == 0:
            p = 0.000001
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def train_with_sketch(self, example, label):
        logit = 0
        for i in range(len(example)):
            val = self.top_k.get_value_for_key(i+1) * example[i]
            logit += val
        sigm_val = self.sigmoid(logit)
        loss = self.loss(y=label, p=sigm_val)
        diff_label = (label - sigm_val)  # difference in label
        if diff_label != 0:
            for i in range(len(example)):
                # updating the change only on previous values
                grad_update = self.learning_rate * diff_label * example[i]
                if i+1 in self.top_k_dict.keys():
                    self.top_k_dict[i+1].append(grad_update)
                value = self.cms.update(i, grad_update)
                # todo: Kanchi: Please check this line
                self.recovered_weight[i] = value
                self.top_k.push(Node(i+1, value))
        return loss

    def accuracy_on_test(self):
        print("Dataset Testing Started")
        test_labels, test_features = self.true_labels, self.samples
        for i in range(len(test_features)):
            test_example = test_features[i]
            pred_label = self.predict(test_example)
            if pred_label == test_labels[i]:
                self.correctly_classified += 1
        # print("correctly classified test examples {}".format(self.correctly_classified))
        print("Dataset Testing Done")

    def predict(self, example):
        logit = 0
        for i in range(len(example)):
            logit += self.top_k.get_value_for_key(i+1) * example[i]
        a = self.sigmoid(logit)
        if a > 0.5:
            return 1
        else:
            return 0


if __name__ == '__main__':
    examples = 10000
    features = 10000
    sparsity = 3
    dataset = SyntheticDatasetGeneration(examples, features, sparsity, "../../dataset_generation/dataset/")
    # read data from file
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
    cms_type = "complementary_cms"
    num_hashes = 2
    count_sketch_size = 6000
    top_k_size = 8000
    lgr = LogisticRegression(cms_type=cms_type,
                             sparsity=sparsity,
                             hash_func_counts=num_hashes,
                             count_sketch_size=count_sketch_size,
                             top_k=top_k_size,
                             dataset_dict=dataset_files_path)
    lgr.train_dataset(epochs=1)
    lgr.accuracy_on_test()
    print("corrrectly classified {}".format(lgr.correctly_classified))

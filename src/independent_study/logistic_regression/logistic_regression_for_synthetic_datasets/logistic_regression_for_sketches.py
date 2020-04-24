import numpy as np
from src.sketches.complementary_count_min_sketch import ComplementaryCountMinSketch
from src.sketches.conservative_complementary_count_min_sketch import ConservativeComplementaryCountMinSketch
from src.sketches.count_sketch import CountSketch
from src.sketches.conservative_count_sketch import ConservativeCountSketch
import math
from src.sketches.top_k import TopK, Node
import random


class LogisticRegression(object):
    cms_dicts = {
        "complementary_cms": ComplementaryCountMinSketch,
        "complementary_cms_conservative": ConservativeComplementaryCountMinSketch,
        "mission_count_sketch": CountSketch,
        "conservative_count_sketch": ConservativeCountSketch
    }

    def __init__(self, sparsity, cms_type, hash_func_counts, batch_size, count_sketch_size, top_k, dataset_dict,
                 top_k_dict={}):
        random.seed(42)
        self.learning_rate = 0.5
        self.cms = self.cms_dicts[cms_type](hash_func_counts, count_sketch_size)
        self.top_k = TopK(top_k)
        self.top_k_dict = top_k_dict
        self.load_dataset(dataset_dict)
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.recovered_weight = np.zeros(self.features, )
        self.non_zero_indexes = np.nonzero(self.weight)
        print("non zero indexes of weights {}".format(self.non_zero_indexes))
        self.non_zero_weights = []
        for index in self.non_zero_indexes:
            self.non_zero_weights.append(self.weight[index])
        print("non zero weights {}".format(self.non_zero_weights))
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
                if i % 500 == 0:
                    print("i {}".format(i))
                label = self.true_labels[i]
                example = self.samples[i]
                loss = self.train_with_sketch(example, label)
                self.loss_val += loss
        self.reset_weight_sparsity()
        print("Dataset Training Done")

    def reset_weight_sparsity(self):
        for item in self.top_k.heap:
            key = self.top_k.keys[item.value]
            value = lgr.top_k.features[key]
            print("heap position {} value {}".format(key, value))
            self.recovered_weight[key - 1] = value

    def sigmoid(self, x):
        if x >= 0:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def loss(self, y, p):
        if p == 1:
            p = 0.999999
        if p == 0:
            p = 0.000001
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def train_dataset_batch(self, epochs, total_examples):
        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for iteration in self.batch(range(0, total_examples), self.batch_size):
                self.train_batch(iteration)

    def train_batch(self, examples_batch):
        gradient = 0
        loss = 0
        feature_values = [0] * self.features
        for example_index in examples_batch:
            example, label = self.samples[example_index], self.true_labels[example_index]
            logit = 0
            for i in range(len(example)):
                val = self.top_k.get_value_for_key(i + 1) * example[i]
                feature_values[i] += example[i]
                logit += val
            sigm_val = self.sigmoid(logit)
            loss = self.loss(y=label, p=sigm_val)
            diff_label = (label - sigm_val)  # difference in label
            if diff_label != 0:
                for i in range(len(example)):
                    # updating the change only on previous values
                    if example[i] != 0:
                        grad_update = self.learning_rate * diff_label * example[i]
                        if i + 1 in self.top_k_dict.keys():
                            self.top_k_dict[i + 1].append(grad_update)
                        value = self.cms.update(i, grad_update)
                        # self.recovered_weight[i] = value
                        self.top_k.push(Node(i + 1, value))
            return loss


    def train_with_sketch(self, example, label):
        logit = 0
        for i in range(len(example)):
            val = self.top_k.get_value_for_key(i + 1) * example[i]
            logit += val
        sigm_val = self.sigmoid(logit)
        loss = self.loss(y=label, p=sigm_val)
        diff_label = (label - sigm_val)  # difference in label
        if diff_label != 0:
            for i in range(len(example)):
                # updating the change only on previous values
                if example[i] != 0:
                    grad_update = self.learning_rate * diff_label * example[i]
                    if i + 1 in self.top_k_dict.keys():
                        self.top_k_dict[i + 1].append(grad_update)
                    value = self.cms.update(i, grad_update)
                    # self.recovered_weight[i] = value
                    self.top_k.push(Node(i + 1, value))
        return loss

    def accuracy_on_test(self):
        print("Dataset Testing Started")
        test_labels, test_features = self.true_labels, self.samples
        for i in range(len(test_features)):
            if i % 500 == 0:
                print(i)
            test_example = test_features[i]
            pred_label = self.predict(test_example)
            test_label = int(test_labels[i])
            if pred_label == test_label:
                self.correctly_classified += 1
        # print("correctly classified test examples {}".format(self.correctly_classified))
        print("Dataset Testing Done")

    def predict(self, example):
        logit = 0
        for i in range(len(example)):
            logit += self.top_k.get_value_for_key(i + 1) * example[i]
        a = self.sigmoid(logit)
        if a >= 0.5:
            return 1
        else:
            return 0

    def get_recovery_mse(self):
        self.weight -= np.mean(self.weight)
        self.weight /= np.std(self.weight)
        self.recovered_weight -= np.mean(self.recovered_weight)
        self.recovered_weight /= np.std(self.recovered_weight)
        zip_object = zip(self.weight, self.recovered_weight)
        difference = []
        for list1_i, list2_i in zip_object:
            difference.append(list1_i - list2_i)
        return np.mean(np.square(difference))

    def store_in_topk(self, non_zero_index, non_zero_values):
        for i in range(len(non_zero_index)):
            self.top_k.push(Node(non_zero_index[i] + 1, non_zero_values[i]))

    def number_of_position_recovered(self):
        topk_recovered = []
        for item in self.top_k.heap:
            key = self.top_k.keys[item.value]
            topk_recovered.append(key - 1)
        recovered = np.intersect1d(topk_recovered, self.non_zero_indexes[0])
        print("recovered {}".format(recovered))
        return len(recovered)
        # for i in range(len(topk_recovered)):
        #     if topk_recovered[i] in self.non_zero_indexes[0]:
        #         count += 1
        # return count


if __name__ == '__main__':
    examples = 2000
    features = 1000
    sparsity = 50
    dataset_sparsity = 100
    # dataset = SyntheticDatasetGeneration(examples, features, sparsity, "../../dataset_generation/dataset/", 0)
    # read data from file
    dataset_files_path = {
        "examples_path": "../../dataset_generation/dataset/data_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features,
            sparsity, dataset_sparsity),
        "true_label_path": "../../dataset_generation/dataset/true_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples,
            features,
            sparsity, dataset_sparsity),
        "noisy_label_path": "../../dataset_generation/dataset/noisy_labels_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples,
            features,
            sparsity, dataset_sparsity),
        "weights_path": "../../dataset_generation/dataset/weights_dim_{}_{}_sparsity_{}_dataset_sparsity_{}.csv".format(
            examples, features,
            sparsity, dataset_sparsity)
    }
    cms_type = "complementary_cms"
    num_hashes = 2
    count_sketch_size = 100
    top_k_size = 100
    batch_size = 1
    lgr = LogisticRegression(cms_type=cms_type,
                             sparsity=sparsity,
                             batch_size=batch_size,
                             hash_func_counts=num_hashes,
                             count_sketch_size=count_sketch_size,
                             top_k=top_k_size,
                             dataset_dict=dataset_files_path)
    lgr.train_dataset(epochs=25)
    lgr.accuracy_on_test()
    print("method {}".format(cms_type))
    print("Positions recovered {}".format(lgr.number_of_position_recovered()))
    print("correctly classified {}".format(lgr.correctly_classified))
    print("recovery mse {}".format(lgr.get_recovery_mse()))

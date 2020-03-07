from src.independent_study.logistic_regression.logisitic_regression_for_datasets.logistic_regression import logisticRegression
from src.sketches.top_k import TopK, Node
import datetime
import json


class topKLogisticRegression(logisticRegression):

    def __init__(self, dimensions, train_file, test_file, size_topK):
        super(topKLogisticRegression,self).__init__(dimensions, train_file, test_file)
        self.top_k = TopK(size_topK)
        self.top_k_dict = {}

    def train(self, feature_pos, features, label):
        for i in range(len(feature_pos)):
            val = self.top_k.get_value_for_key(feature_pos[i]) * features[i]
        sigm_val = self.sigmoid(val)
        print("label {} sigmoid {}".format(label, sigm_val))
        loss = self.loss(y=label, p=sigm_val)
        diff_label = (label - sigm_val)  # difference in label
        if diff_label != 0:
            for i in range(len(feature_pos)):
                value=0
                # updating the change only on previous values
                if feature_pos[i] in self.top_k_dict.keys():
                    updated_val = self.learning_rate * diff_label * features[i]
                    self.gradient_updates_dict[feature_pos[i]].append(updated_val)
                    self.gradients[feature_pos[i]] += updated_val
                    value = self.gradients[feature_pos[i]]
                self.top_k.push(Node(feature_pos[i], value))
        return loss

    def dump_top_K(self, filename):
        with open(filename + str(datetime.datetime.now()), 'w') as f:
            f.write(json.dumps(self.top_k_dict))


if __name__ == '__main__':
    lgr = topKLogisticRegression(47326, "rcv1_train.binary", "rcv1_test.binary", 8000)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    lgr.dump_gradient_updates("../../dumps/topk_logistic_regression_gradient_updates.json")
    lgr.dump_top_K('../../dumps/top8000_topk_logistic_regression.txt')

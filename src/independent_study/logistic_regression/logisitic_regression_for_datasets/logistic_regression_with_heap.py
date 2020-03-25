from src.independent_study.logistic_regression.logisitic_regression_for_datasets.logistic_regression import LogisticRegression
from src.sketches.top_k import TopK, Node


class LogisticRegressionWithHeap(LogisticRegression):
    '''
    Logistic Regression with Top K Heap where in after each iteration we are maintaining
     top k features in heap.
    '''

    def __init__(self, dimensions, train_file, test_file, size_topK):
        super(LogisticRegressionWithHeap, self).__init__(dimensions, train_file, test_file)
        self.top_k = TopK(size_topK)
        self.top_k_dict = {}

    def train(self, feature_pos, features, label):
        val = 0
        for i in range(len(feature_pos)):
            val += self.top_k.get_value_for_key(feature_pos[i]) * features[i]
        sigm_val = self.sigmoid(val)
        print("label {} sigmoid {}".format(label, sigm_val))
        loss = self.loss(y=label, p=sigm_val)
        diff_label = (label - sigm_val)  # difference in label
        if diff_label != 0:
            for i in range(len(feature_pos)):
                # updating the change only on previous values
                grad_update = self.learning_rate * diff_label * features[i]
                self.gradient_updates_dict[feature_pos[i]].append(grad_update)
                self.gradients[feature_pos[i]] += grad_update
                self.top_k.push(Node(feature_pos[i],  self.gradients[feature_pos[i]]))
        return loss

    def dump_top_K(self, filename):
        with open(filename+".json", 'w') as f:
            for item in self.top_k.heap:
                key = self.top_k.keys[item.value]
                value = self.top_k.features[key]
                f.write("{}:{}\n".format(key, value))



if __name__ == '__main__':
    lgr = LogisticRegressionWithHeap(47326, "rcv1_train.binary", "rcv1_test.binary", 8000)
    lgr.train_dataset(1)
    lgr.accuracy_on_test()
    lgr.dump_top_K('../../dumps/heap_logistic/topk_heap_logistic_regression')
    lgr.dump_gradient_updates("../../dumps/heap_logistic/heap_logistic_regression_gradient_updates")


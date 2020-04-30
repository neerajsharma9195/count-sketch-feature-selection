import numpy as np
from src.utils.utils import isPrime
from src.utils.hash_generator import HashGeneration


class ComplementaryCountMinSketchVariant(object):
    '''
        hash function: ((a*number + b)%p)%w
    '''
    def __init__(self, h, w):
        np.random.seed(42)
        self.num_hash = h
        self.bucket_size = w
        self.countSketchPos = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.countSketchNeg = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.hash_func_obj = HashGeneration(self.num_hash, self.bucket_size)

    def update(self, number, value=1):
        '''
        If number is > 0: add it to countMinPositive else add it to countMinNegative
        :param number:
        :return:
        '''
        pos_hash_indexes = self.hash_func_obj.get_hash_value(number)
        neg_hash_indexes = self.hash_func_obj.get_hash_value(-number)
        if value > 0:
            for i in range(self.num_hash):
                self.countSketchPos[i][pos_hash_indexes[i]] += value
        else:
            for i in range(self.num_hash):
                self.countSketchNeg[i][neg_hash_indexes[i]] += abs(value)
        return self.query(number)


    def print_cms(self):
        print("positive cms")
        for i in range(self.num_hash):
            print(self.countSketchPos[i])
        print("negative cms")
        for i in range(self.num_hash):
            print(self.countSketchNeg[i])

    def insert_list(self, inputs):
        for val, item in inputs:
            self.update(val, item)

    def query(self, number):
        '''
        We find the number in both sketches and take min of positive sketch and max of negative sketch
        :param number:
        :return:
        '''
        poses = []
        negs = []
        hash_indexes_pos = self.hash_func_obj.get_hash_value(number)
        hash_indexes_neg = self.hash_func_obj.get_hash_value(-number)
        for i in range(self.num_hash):
            poses.append(self.countSketchPos[i][hash_indexes_pos[i]])
            negs.append(self.countSketchNeg[i][hash_indexes_neg[i]])
        resp = min(poses) - min(negs)
        return resp


if __name__ == '__main__':
    cms = ComplementaryCountMinSketchVariant(3, 10)
    cms.insert_list([(10, 1), (10, -1), (10, 1),(101, 1), (101, -1), (101, 1)])
    print(cms.query(10))
    print(cms.query(101))
    print(cms.query(7))
    cms.print_cms()

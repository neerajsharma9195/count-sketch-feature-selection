import numpy as np
from src.utils.utils import isPrime
from src.utils.hash_generator import HashGeneration


class CountMinSketchVariant(object):
    '''
        hash function: ((a*number + b)%p)%w
    '''
    def __init__(self, h, w):
        np.random.seed(42)
        self.num_hash = h
        self.bucket_size = w
        self.countSketch = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.hash_func_obj = HashGeneration(self.num_hash, self.bucket_size)

    def update(self, number, value=1):
        '''
        If number is > 0: add it to countMinPositive else add it to countMinNegative
        :param number:
        :return:
        '''
        if value > 0:
            hash_indexes = self.hash_func_obj.get_hash_value(number)
            for i in range(self.num_hash):
                self.countSketch[i][hash_indexes[i]] += value
        else:
            hash_indexes = self.hash_func_obj.get_hash_value(-number)
            for i in range(self.num_hash):
                self.countSketch[i][hash_indexes[i]] += abs(value)

        return self.query(number)


    def print_cms(self):
        print("positive cms")
        for i in range(self.num_hash):
            print(self.countSketch[i])


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
        pos_hash_indexes = self.hash_func_obj.get_hash_value(number)
        neg_hash_indexes = self.hash_func_obj.get_hash_value(-number)
        for i in range(self.num_hash):
            poses.append(self.countSketch[i][pos_hash_indexes[i]])
            negs.append(self.countSketch[i][neg_hash_indexes[i]])
        return min(poses) - min(negs)


if __name__ == '__main__':
    cms = CountMinSketchVariant(3, 10)
    cms.insert_list([(10, 1), (10, -1), (10, 1), (7, -1), (7, 1), (7, 1)])
    print(cms.query(10))
    print(cms.query(5))
    print(cms.query(7))
    cms.print_cms()

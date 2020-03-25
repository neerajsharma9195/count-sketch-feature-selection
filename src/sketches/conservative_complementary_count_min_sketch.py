import statistics as s
import random
from src.utils.utils import isPrime
import numpy as np


class ConservativeComplementaryCountMinSketch(object):
    '''
    Complementary Count Min Sketch with Conservative updates
    '''

    def __init__(self, h, w):
        np.random.seed(42)
        self.num_hash = h
        self.bucket_size = w
        self.countSketchPos = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.countSketchNeg = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.first_nums = [np.random.randint(1, 1000) for i in range(self.num_hash)]
        self.second_nums = [np.random.randint(1, 1000) for i in range(self.num_hash)]
        self.ps = []
        for i in range(self.num_hash):
            a = np.random.randint(self.bucket_size + 1, self.bucket_size + 3000)
            while not isPrime(a):
                a = np.random.randint(self.bucket_size + 1, self.bucket_size + 3000)
            self.ps.append(a)
        first_hash_function = lambda number: ((self.first_nums[0] * number + self.second_nums[0]) % self.ps[0]) % w
        second_hash_function = lambda number: ((self.first_nums[1] * number + self.second_nums[1]) % self.ps[1]) % w
        self.hashes = [first_hash_function, second_hash_function]

    def get_hash_values(self, number):
        return [(i, self.first_nums[i], self.second_nums[i], self.hashes[i](number)) for i in range(len(self.hashes))]

    def update(self, number, value):
        '''
        If number is > 0: add it to countMinPositive else add it to countMinNegative
        :param number:
        :return:
        '''
        if value > 0:
            current_values = [(i, hash_func(number), self.countSketchPos[i][hash_func(number)]) for i, hash_func in
                              enumerate(self.hashes)]
            min_val = min(current_values, key=lambda x: x[2])
            min_indexes = [current_values[i] for i in range(len(current_values)) if current_values[i][2] == min_val[2]]
            for i, hash_pos, current in min_indexes:
                self.countSketchPos[i][hash_pos] += value
        else:
            current_values = [(i, hash_func(number), self.countSketchNeg[i][hash_func(number)]) for i, hash_func in
                              enumerate(self.hashes)]
            min_val = min(current_values, key=lambda x: x[2])
            min_indexes = [current_values[i] for i in range(len(current_values)) if current_values[i][2] == min_val[2]]
            for i, hash_pos, current in min_indexes:
                self.countSketchNeg[i][hash_pos] += abs(value)
        return self.query(number)

    def print_cms(self):
        print("positive cms {}".format(self.countSketchPos))
        print("negative cms {}".format(self.countSketchNeg))

    def query(self, number):
        '''
        We find the number in both sketches and take min of positive sketch and max of negative sketch
        :param number:
        :return:
        '''
        indexes = [hash_func(number) for hash_func in self.hashes]
        poses = [self.countSketchPos[i][index] for i, index in enumerate(indexes)]
        negs = [self.countSketchNeg[i][index] for i, index in enumerate(indexes)]
        return min(poses) - min(negs)


if __name__ == '__main__':
    cms = ConservativeComplementaryCountMinSketch(3, 10)
    cms.update(8, 1)
    cms.update(8, 0.1)
    cms.update(8, - 0.1)
    # cms.update(5)
    cms.update(1, 1)
    cms.update(2, 1)
    print("query {}".format(cms.query(8)))
    print("query {}".format(cms.query(5)))
    print("query {}".format(cms.query(-8)))
    cms.print_cms()

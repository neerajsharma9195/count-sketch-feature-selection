from src.utils.utils import isPrime
from src.utils.hash_generator import HashGeneration
import numpy as np
import random


class ConservativeComplementaryCountMinSketch(object):
    '''
    Complementary Count Min Sketch with Conservative updates
    '''

    def __init__(self, h, w):
        random.seed(42)
        self.num_hash = h
        self.bucket_size = w
        self.countSketchPos = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.countSketchNeg = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.hash_func_obj = HashGeneration(self.num_hash, self.bucket_size)

    def get_hash_values(self, number):
        return [(i, self.first_nums[i], self.second_nums[i], self.hashes[i](number)) for i in range(len(self.hashes))]

    def update(self, number, value):
        '''
        If number is > 0: add it to countMinPositive else add it to countMinNegative
        :param number:
        :return:
        '''
        hash_indexes = self.hash_func_obj.get_hash_value(number)
        if value > 0:
            current_values = [(i, hash_indexes[i], self.countSketchPos[i][hash_indexes[i]]) for i in
                              range(self.num_hash)]
            min_val = min(current_values, key=lambda x: x[2])
            min_indexes = [current_values[i] for i in range(len(current_values)) if current_values[i][2] == min_val[2]]
            for i, hash_pos, current in min_indexes:
                self.countSketchPos[i][hash_pos] += value
        else:
            current_values = [(i, hash_indexes[i], self.countSketchNeg[i][hash_indexes[i]]) for i in
                              range(self.num_hash)]
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
        indexes = self.hash_func_obj.get_hash_value(number)
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

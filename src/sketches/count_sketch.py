import statistics as s
import random
from src.utils.utils import isPrime
import numpy as np
from src.utils.hash_generator import HashGeneration


class CountSketch(object):
    def __init__(self, h, w):
        np.random.seed(42)
        self.num_hash = h
        self.bucket_size = w
        self.countsketch = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.hash_func_obj = HashGeneration(self.num_hash, self.bucket_size)

    def update(self, number, value=1):
        hash_indexes, sign_funcs = self.hash_func_obj.get_hash_sign_and_value(number)
        for i in range(self.num_hash):
            sign = sign_funcs[i]
            self.countsketch[i][hash_indexes[i]] += sign * value

    def insert_list(self, inputs):
        for val, item in inputs:
            self.update(val, item)

    def print_cms(self):
        for i in range(self.num_hash):
            print(self.countsketch[i])

    def query(self, number):
        hash_indexes, sign_funcs = self.hash_func_obj.get_hash_sign_and_value(number)
        return s.median([self.countsketch[i][hash_indexes[i]] * sign_funcs[i] for i in range(self.num_hash)])


if __name__ == '__main__':
    cms = CountSketch(3, 10)
    cms.insert_list([(10, 1), (10, -1), (10, 1), (5, 1), (3, 1), (4, -1), (7, -1), (7, 1), (7, 1)])
    print(cms.query(10))
    print(cms.query(5))
    print(cms.query(7))
    cms.print_cms()

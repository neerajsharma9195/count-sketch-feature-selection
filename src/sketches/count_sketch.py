import statistics as s
import random
from src.utils.utils import isPrime
import numpy as np


class CountSketch(object):
    def __init__(self, h, w):
        np.random.seed(42)
        self.num_hash = h
        self.bucket_size = w
        self.countsketch = [[0 for i in range(self.bucket_size)] for j in range(h)]
        self.first_nums = [random.randint(1, 1000) for i in range(self.num_hash)]
        self.second_nums = [random.randint(1, 1000) for i in range(self.num_hash)]
        self.sign_firsts = [random.randint(1, 1000) for i in range(self.num_hash)]
        self.sign_seconds = [random.randint(1, 1000) for i in range(self.num_hash)]
        print("first nums {} second nums {} sign firsts {} sign seconds {}".format(self.first_nums, self.second_nums,
                                                                                   self.sign_firsts, self.sign_seconds))
        self.ps = []
        self.sign_ps = []
        for i in range(self.num_hash):
            a = random.randint(self.bucket_size + 1, self.bucket_size + 3000)
            while not isPrime(a):
                a = random.randint(self.bucket_size + 1, self.bucket_size + 3000)
            self.ps.append(a)
            b = random.randint(self.bucket_size + 1, self.bucket_size + 3000)
            while not isPrime(b):
                b = random.randint(self.bucket_size + 1, self.bucket_size + 3000)
            self.sign_ps.append(b)
        print("sign ps {} ps {}".format(self.sign_ps, self.ps))
        for a, b, p in zip(self.first_nums, self.second_nums, self.ps):
            print("zip results a b p {} {} {}".format(a, b, p))

        first_hash_function = lambda number: ((self.first_nums[0] * number + self.second_nums[0]) % self.ps[0]) % w
        second_hash_function = lambda number: ((self.first_nums[1] * number + self.second_nums[1]) % self.ps[1]) % w
        third_hash_function = lambda number: ((self.first_nums[2] * number + self.second_nums[2]) % self.ps[2]) % w
        self.hashes = [first_hash_function, second_hash_function, third_hash_function]
        first_sign_function = lambda number: 1 if ((self.sign_firsts[0] * number + self.sign_seconds[0]) % self.sign_ps[
            0] % 2) == 0 else -1
        second_sign_function = lambda number: 1 if ((self.sign_firsts[1] * number + self.sign_seconds[1]) %
                                                    self.sign_ps[1] % 2) == 0 else 1
        third_sign_function = lambda number: 1 if ((self.sign_firsts[2] * number + self.sign_seconds[2]) % self.sign_ps[
            2] % 2) == 0 else -1
        self.signhashes = [first_sign_function, second_sign_function, third_sign_function]

    def get_hash_values(self, number):
        return [(i, self.first_nums[i], self.second_nums[i], self.hashes[i](number)) for i in range(len(self.hashes))]

    def update(self, number, value):
        values = []
        for i in range(self.num_hash):
            sign = self.signhashes[i](number)
            self.countsketch[i][self.hashes[i](number)] += sign * value
            values.append(sign * self.countsketch[i][self.hashes[i](number)])
        return s.median(values)

    def conservative_update(self, number):
        median_value = self.query(number)
        for i in range(self.num_hash):
            if number > 0:
                if self.countsketch[i][self.hashes[i](number)] <= median_value:
                    self.countsketch[i][self.hashes[i](number)] += self.signhashes[i](number)
            else:
                if self.countsketch[i][self.hashes[i](number)] > median_value:
                    self.countsketch[i][self.hashes[i](number)] += self.signhashes[i](number)

    def print_cms(self):
        print(self.countsketch)

    def query(self, number):
        return s.median(
            [self.countsketch[i][self.hashes[i](number)] * self.signhashes[i](number) for i in range(self.num_hash)])


if __name__ == '__main__':
    cms = CountSketch(3, 10)
    print(cms.get_hash_values(9))
    cms.conservative_update(8)
    # cms.conservative_update(-8)
    cms.conservative_update(8)
    # cms.conservative_update(-8)
    cms.conservative_update(8)
    # cms.conservative_update(-5)
    cms.conservative_update(5)
    print("query {}".format(cms.query(8)))
    print("query {}".format(cms.query(5)))
    cms.print_cms()

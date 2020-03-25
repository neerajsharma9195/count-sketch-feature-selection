import numpy as np
from src.utils.utils import isPrime


class ComplementaryCountMinSketch(object):
    '''
        hash function: ((a*number + b)%p)%w
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
        print("ps {}".format(self.ps))
        first_hash_function = lambda number: ((self.first_nums[0] * number + self.second_nums[0]) % self.ps[0]) % w
        second_hash_function = lambda number: ((self.first_nums[1] * number + self.second_nums[1]) % self.ps[1]) % w
        # third_hash_function = lambda number: ((self.first_nums[2] * number + self.second_nums[2]) % self.ps[2]) % w
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
            for i, hash_func in enumerate(self.hashes):
                self.countSketchPos[i][hash_func(number)] += value
        else:
            for i, hash_func in enumerate(self.hashes):
                self.countSketchNeg[i][hash_func(number)] += abs(value)
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
        poses = []
        negs = []
        for i, hash_func in enumerate(self.hashes):
            poses.append(self.countSketchPos[i][hash_func(number)])
            negs.append(self.countSketchNeg[i][hash_func(number)])
        resp = min(poses) - max(negs)
        return resp


if __name__ == '__main__':
    cms = ComplementaryCountMinSketch(3, 10)
    cms.update(8, 1)
    cms.update(8, 0.1)
    cms.update(8, - 0.1)
    cms.update(1, 1)
    cms.update(2, 1)
    print("query {}".format(cms.query(8)))
    print("query {}".format(cms.query(5)))
    print("query {}".format(cms.query(-8)))
    cms.print_cms()

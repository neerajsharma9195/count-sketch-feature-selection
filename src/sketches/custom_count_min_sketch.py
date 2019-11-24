import statistics as s


class CustomCountMinSketch(object):
    def __init__(self, h, w):
        self.num_hash = h
        self.bucket_size = w
        self.countSketchPos = [[0 for i in range(w)] for j in range(h)]
        self.countSketchNeg = [[0 for i in range(w)] for j in range(h)]
        self.first_nums = [5, 11, 8]
        self.second_nums = [9, 19, 3]
        self.ps = [13, 17, 19]
        # todo: hash function: random generator (a, b, p: prime > w)
        '''
        hash function: ((a*number + b)%p)%w
        '''
        first_hash_function = lambda number: ((self.first_nums[0] * number + self.second_nums[0]) % self.ps[0]) % w
        second_hash_function = lambda number: ((self.first_nums[1] * number + self.second_nums[1]) % self.ps[1]) % w
        third_hash_function = lambda number: ((self.first_nums[2] * number + self.second_nums[2]) % self.ps[2]) % w
        self.hashes = [first_hash_function, second_hash_function, third_hash_function]
        # for i in range(0, self.num_hash):
        #     self.hashes.append(lambda number: (((self.first_nums[i] * number + self.second_nums[i]) % self.ps[i]) % w))
        # self.hashes = (lambda number: (((self.first_nums[i] * number + self.second_nums[i]) % self.ps[i]) % w) for i in range(self.num_hash))
        # self.signhashes = (lambda number: (1 if (((a * number + b) % p) % 2 == 0) else -1) for a, b, p in
        #                    zip(self.sign_firsts, self.sign_seconds, self.sign_ps))

    def get_hash_values(self, number):
        return [(i, self.first_nums[i], self.second_nums[i], self.hashes[i](number)) for i in range(len(self.hashes))]

    def update(self, number):
        '''
        If number is > 0: add it to countMinPositive else add it to countMinNegative
        :param number:
        :return:
        '''
        if number > 0:
            for i, hash_func in enumerate(self.hashes):
                self.countSketchPos[i][hash_func(number)] += 1
        else:
            for i, hash_func in enumerate(self.hashes):
                self.countSketchNeg[i][hash_func(abs(number))] += 1

    # def conservative_update(self, number):
    #     median_value = self.query(number)
    #     for i in range(self.num_hash):
    #         if number > 0:
    #             if self.countsketch[i][self.hashes[i](number)] < median_value:
    #                 self.countsketch[i][self.hashes[i](number)] += self.signhashes[i](number)
    #         else:
    #             if self.countsketch[i][self.hashes[i](number)] >= median_value:
    #                 self.countsketch[i][self.hashes[i](number)] += self.signhashes[i](number)

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
            negs.append(self.countSketchNeg[i][hash_func(abs(number))])
        print("query for number {} poses {} negs {}".format(number, poses, negs))
        return abs(min(poses) - max(negs))


if __name__ == '__main__':
    cms = CustomCountMinSketch(3, 10)
    cms.update(8)
    cms.update(8)
    cms.update(8)
    #cms.update(5)
    cms.update(-8)
    cms.update(-8)
    print("query {}".format(cms.query(8)))
    print("query {}".format(cms.query(5)))
    print("query {}".format(cms.query(-8)))
    cms.print_cms()

import statistics as s


class CountMinSketch(object):
    def __init__(self, h, w):
        self.num_hash = h
        self.bucket_size = w
        self.countsketch = [[0 for i in range(w)] for j in range(h)]
        self.first_nums = [5, 11, 8]
        self.second_nums = [9, 19, 3]
        self.ps = [13, 17, 19]
        for a, b, p in zip(self.first_nums, self.second_nums, self.ps):
            print("zip results a b p {} {} {}".format(a, b, p))

        first_hash_function = lambda number: ((self.first_nums[0] * number + self.second_nums[0]) % self.ps[0]) % w
        second_hash_function = lambda number: ((self.first_nums[1] * number + self.second_nums[1]) % self.ps[1]) % w
        third_hash_function = lambda number: ((self.first_nums[2] * number + self.second_nums[2]) % self.ps[2]) % w
        self.hashes = [first_hash_function, second_hash_function, third_hash_function]
        # for i in range(0, self.num_hash):
        #     self.hashes.append(lambda number: (((self.first_nums[i] * number + self.second_nums[i]) % self.ps[i]) % w))
        # self.hashes = [lambda number: (((self.first_nums[i] * number + self.second_nums[i]) % self.ps[i]) % w) for i in
        #                range(self.num_hash)]
        # self.signhashes = [lambda number: (1 if (((a * number + b) % p) % 2 == 0) else -1) for a, b, p in
        #                    zip(self.sign_firsts, self.sign_seconds, self.sign_ps)]

    def get_hash_values(self, number):
        return [(i, self.first_nums[i], self.second_nums[i], self.hashes[i](number)) for i in range(len(self.hashes))]

    def update(self, number):
        for i, hash_func in enumerate(self.hashes):
            print("at i {} hash value {}".format(i, hash_func(number)))
            print(self.countsketch[i][hash_func(number)])
            self.countsketch[i][hash_func(number)] += 1

    def conservative_update(self, number):
        median_value = self.query(number)
        for i, hash_func in enumerate(self.hashes):
            if number > 0:
                if self.countsketch[i][hash_func(number)] < median_value:
                    self.countsketch[i][hash_func(number)] += self.signhashes[i](number)
            else:
                if self.countsketch[i][self.hashes[i](number)] >= median_value:
                    self.countsketch[i][self.hashes[i](number)] += self.signhashes[i](number)

    def print_cms(self):
        print(self.countsketch)

    def query(self, number):
        output = min([self.countsketch[i][hash_func(number)] for i, hash_func in enumerate(self.hashes)])
        print(output)


if __name__ == '__main__':
    cms = CountMinSketch(3, 10)
    cms.update(8)
    cms.update(8)
    cms.update(8)
    cms.update(5)
    print("query {}".format(cms.query(8)))
    print("query {}".format(cms.query(5)))
    cms.print_cms()

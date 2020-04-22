from src.utils.hash_generator import HashGeneration


class CountMinSketch(object):
    def __init__(self, h, w):
        self.num_hash = h
        self.bucket_size = w
        self.countsketch = [[0 for i in range(w)] for j in range(h)]
        self.hash_func_obj = HashGeneration(self.num_hash, self.bucket_size)

    def update(self, number):
        hash_indexes = self.hash_func_obj.get_hash_value(number)
        for i in range(self.num_hash):
            self.countsketch[i][hash_indexes[i]] += 1

    def print_cms(self):
        for i in range(self.num_hash):
            print(self.countsketch[i])

    def query(self, number):
        hash_indexes = self.hash_func_obj.get_hash_value(number)
        output = min([self.countsketch[i][hash_indexes[i]] for i in range(self.num_hash)])
        return output

    def insert_list(self, inputs):
        for item in inputs:
            self.update(item)


if __name__ == '__main__':
    cms = CountMinSketch(3, 10)
    cms.insert_list([8, 8, 8, 5, 6, 7, 8, 2, -8, -8])
    print("query {}".format(cms.query(8)))
    print("query {}".format(cms.query(5)))
    print("query {}".format(cms.query(-8)))
    cms.print_cms()

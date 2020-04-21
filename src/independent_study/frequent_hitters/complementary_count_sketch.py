from src.independent_study.frequent_hitters.hash_generation import HashGeneration
from src.independent_study.frequent_hitters.count_min_sketch import CountMinSketch


class CompCountMinSketch():
    def __init__(self, repetitions, num_hash, size):
        self.pos = CountMinSketch(repetitions, num_hash, size)
        self.neg = CountMinSketch(repetitions, num_hash, size)
        self.num_hash = num_hash
        self.frequent_item = float('inf')
        self.frequent_count = 0

    def insert_list(self, numbers):
        for number in numbers:
            self.insert(number)

    def update_frequent_item(self, number):
        pos_positions = self.pos.hash.get_hash_value(number)
        neg_positions = self.neg.hash.get_hash_value(number)
        min_pos_freq = float('inf')
        min_neg_freq = float('inf')
        for i in range(self.num_hash):
            if self.pos.sketch[i][pos_positions[i]] < min_pos_freq:
                min_pos_freq = self.pos.sketch[i][pos_positions[i]]
        for i in range(self.num_hash):
            if self.neg.sketch[i][neg_positions[i]] < min_pos_freq:
                min_neg_freq = self.neg.sketch[i][neg_positions[i]]
        if min_pos_freq - min_neg_freq > self.frequent_count:
            self.frequent_count = min_pos_freq - min_neg_freq
            self.frequent_item = number

    def insert(self, number):
        if number < 0:
            self.neg.insert(-1 * number)
        else:
            self.pos.insert(number)
        self.update_frequent_item(number)

    def get_frequent_item(self):
        return self.frequent_item, self.frequent_count

    def get_item_frequency(self, number):
        pos_positions = self.pos.hash.get_hash_value(number)
        neg_positions = self.neg.hash.get_hash_value(number)
        min_pos_freq = float('inf')
        min_neg_freq = float('inf')
        for i in range(self.num_hash):
            if self.pos.sketch[i][pos_positions[i]] < min_pos_freq:
                min_pos_freq = self.pos.sketch[i][pos_positions[i]]
        for i in range(self.num_hash):
            if self.neg.sketch[i][neg_positions[i]] < min_pos_freq:
                min_neg_freq = self.neg.sketch[i][neg_positions[i]]
        return min_pos_freq - min_neg_freq

    def print_sketch(self):
        print("Positive Sketch")
        for i in range(len(self.pos.sketch)):
            print(self.pos.sketch[i])
        print("Negative Sketch")
        for i in range(len(self.neg.sketch)):
            print(self.neg.sketch[i])


if __name__ == '__main__':
    cms = CompCountMinSketch(1, 3, 10)
    cms.insert_list([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cms.insert_list([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    cms.insert_list([22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22])
    cms.insert_list([52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52])
    cms.insert_list([52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, -52, -52, -52, -52, -52, -52, -52])
    cms.print_sketch()
    print(cms.get_item_frequency(1))
    print(cms.get_frequent_item())

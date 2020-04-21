from src.independent_study.frequent_hitters.hash_generation import HashGeneration
import statistics

class CountSketch():
    def __init__(self, repetitions, num_hash, size):
        self.sketch = [[0 for i in range(size)] for j in range(num_hash)]
        self.num_hash = num_hash
        self.hash = HashGeneration(num_hash, size)
        self.frequent_item = float('inf')
        self.frequent_count = 0

    def insert_list(self, numbers):
        for number in numbers:
            self.insert(number)

    def insert(self, number):
        positions = self.hash.get_hash_value(number)
        values = []
        for i in range(self.num_hash):
            if positions[i]%2 == 0:
                self.sketch[i][positions[i]] += 1
            else:
                self.sketch[i][positions[i]] -= 1
            values.append(self.sketch[i][positions[i]])
        if self.frequent_count < statistics.median(values):
            self.frequent_count = statistics.median(values)
            self.frequent_item = number

    def get_frequent_item(self):
        return self.frequent_item, self.frequent_count

    def get_item_frequency(self, number):
        positions = self.hash.get_hash_value(number)
        values = []
        for i in range(self.num_hash):
            values.append(self.sketch[i][positions[i]])
        return statistics.median(values)

    def print_sketch(self):
        for i in range(len(self.sketch)):
            print(self.sketch[i])


if __name__ == '__main__':
    cs = CountSketch(1, 3, 10)
    cs.insert_list([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cs.insert_list([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2])
    cs.insert_list([22, 22, 22, 22, 22, 22, 22, -22,-22, -22, -22, -22, -22, -22, -22])
    cs.insert_list([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32])
    cs.insert_list([52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,-52,-52,-52,-52,-52,-52])
    cs.print_sketch()
    print(cs.get_item_frequency(32))
    print(cs.get_item_frequency(52))
    print(cs.get_item_frequency(22))
    print(cs.get_frequent_item())

from src.independent_study.frequent_hitters.hash_generation import HashGeneration


class CountMinSketch():
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
            self.sketch[i][positions[i]] += 1
            values.append(self.sketch[i][positions[i]])
        if self.frequent_count < min(values):
            self.frequent_count = min(values)
            self.frequent_item = number

    def get_frequent_item(self):
        return self.frequent_item, self.frequent_count

    def get_item_frequency(self, number):
        positions = self.hash.get_hash_value(number)
        minimum_frequency = float('inf')
        for i in range(self.num_hash):
            if self.sketch[i][positions[i]] < minimum_frequency:
                minimum_frequency = self.sketch[i][positions[i]]
        return minimum_frequency

    def print_sketch(self):
        for i in range(len(self.sketch)):
            print(self.sketch[i])


if __name__ == '__main__':
    cms = CountMinSketch(1, 3, 10)
    cms.insert_list([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cms.insert_list([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2])
    cms.insert_list([22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22])
    #cms.insert_list([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32])
    #cms.insert_list([52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, -52,-52,-52,-52,-52,-52,-52])
    cms.print_sketch()
    print(cms.get_item_frequency(1))
    print(cms.get_frequent_item())

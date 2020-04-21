import random
from src.utils.utils import isPrime


class HashGeneration():
    def __init__(self, num_hash, size):
        self.num_hash = num_hash
        self.size = size
        self.primes = []
        self.first = []
        self.second = []
        self.generate_hash_function()

    def generate_hash_function(self):
        primes = [i for i in range(self.size) if isPrime(i)]
        self.primes = random.sample(primes, self.num_hash)
        self.first = [random.randint(1, 1000) for i in range(self.num_hash)]
        self.second = [random.randint(1, 1000) for i in range(self.num_hash)]

    def get_hash_value(self, number):
        hash_values = [0] * self.num_hash
        for i in range(self.num_hash):
            hash_values[i] = (self.first[i] * number + self.second[i]) % self.primes[i]
        return hash_values

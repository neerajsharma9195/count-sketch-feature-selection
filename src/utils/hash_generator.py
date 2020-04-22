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
        # ((Ax + B) mod prime) % size_of_array
        primes = [i for i in range(self.size + 1, self.size + 3000) if isPrime(i)]
        self.primes = random.sample(primes, self.num_hash)
        self.first = [random.randint(1, 1000) for i in range(self.num_hash)]
        self.second = [random.randint(1, 1000) for i in range(self.num_hash)]

    def get_hash_sign_and_value(self, number):
        hash_values = [0] * self.num_hash
        sign_values = [0] * self.num_hash
        for i in range(self.num_hash):
            hash_val = ((self.first[i] * number + self.second[i]) % self.primes[i])
            hash_values[i] = hash_val % self.size
            sign_values[i] = 1 if hash_val % 2 == 0 else -1
        return hash_values, sign_values

    def get_hash_value(self, number):
        hash_values = [0] * self.num_hash
        for i in range(self.num_hash):
            hash_values[i] = ((self.first[i] * number + self.second[i]) % self.primes[i]) % self.size
        return hash_values

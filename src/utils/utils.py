from math import sqrt


def isPrime(n: int):
    if n % 2 == 0:
        return False
    for i in range(3, int(sqrt(n)), 2):
        if n % i == 0:
            return False
    return True


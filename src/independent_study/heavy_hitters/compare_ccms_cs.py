from collections import Counter
from src.sketches.count_sketch import CountSketch
from src.sketches.complementary_count_min_sketch import ComplementaryCountMinSketch
import numpy as np

import random

ccms_losses = []
cs_losses = []


def power_law(k_min, k_max, y, gamma):
    return ((k_max ** (-gamma + 1) - k_min ** (-gamma + 1)) * y + k_min ** (-gamma + 1.0)) ** (1.0 / (-gamma + 1.0))


nodes = 10000

power_law_distribution = np.zeros(nodes, float)
k_min = 1.0
k_max = 1000 * k_min
gamma = 3.0

for n in range(nodes):
    power_law_distribution[n] = power_law(k_min, k_max, np.random.uniform(0, 1), gamma)

round_values = [int(round(item)) for item in power_law_distribution]


def repetitions():
    for n in range(nodes):
        power_law_distribution[n] = power_law(k_min, k_max, np.random.uniform(0, 1), gamma)
    round_values = [int(round(item)) for item in power_law_distribution]
    pos_neg = [1, -1]
    random_numbers = [random.choice(pos_neg) * item for item in round_values]
    count_dict = Counter(random_numbers)
    actual_count_dict = count_dict
    count_dict = sorted(count_dict.items())
    ccms = ComplementaryCountMinSketch(4, 25)
    # top frequent items comparison
    cs = CountSketch(5, 50)
    for item in random_numbers:
        if item > 0:
            ccms.update(item)
            cs.update(item)
        else:
            ccms.update(abs(item), -1)
            cs.update(abs(item), -1)
    items = list(val[0] for val in count_dict)
    items = list(set(items))
    ccms_loss = 0
    cs_loss = 0
    for item in items:
        ccms_val = ccms.query(item)
        cs_val = cs.query(item)
        actual_count = actual_count_dict[item] - actual_count_dict[-item]
        ccms_loss += (actual_count - ccms_val) ** 2
        cs_loss += (actual_count - cs_val) ** 2
    ccms_losses.append(ccms_loss / len(items))
    cs_losses.append(cs_loss / len(items))


for i in range(100):
    repetitions()

print("ccms avergage loss {}".format(np.mean(ccms_losses)))
print("cs average loss {}".format(np.mean(cs_losses)))

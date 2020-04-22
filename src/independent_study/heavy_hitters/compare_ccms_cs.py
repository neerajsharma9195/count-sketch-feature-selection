import random
from collections import Counter
from src.sketches.count_sketch import CountSketch
from src.sketches.complementary_count_min_sketch import ComplementaryCountMinSketch
import numpy as np

ccms_losses = []
cs_losses = []


def repetitions():
    pos_neg = [1, -1]
    random_numbers = [random.choice(pos_neg) * random.randint(1, 100) for i in range(10000)]
    count_dict = Counter(random_numbers)
    actual_count_dict = count_dict
    count_dict = sorted(count_dict.items())
    ccms = ComplementaryCountMinSketch(4, 25)
    cs = CountSketch(5, 20)
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
    ccms_losses.append(ccms_loss/len(items))
    cs_losses.append(cs_loss/len(items))


for i in range(1000):
    repetitions()


print("ccms avergage loss {}".format(np.mean(ccms_losses)))
print("cs average loss {}".format(np.mean(cs_losses)))

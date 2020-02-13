items = []

with open("/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/project_repo/code/src/results/topk_results.txt", 'r') as f:
    for line in f:
        a = line.split(":")
        print("a {}".format(a))
        items.append(abs(float(a[1].strip())))

items = sorted(items, reverse=True)
import matplotlib.pyplot as plt
import numpy as np
x = [i for i in range(len(items))]
plt.plot(x, items)
plt.xlabel("features")
plt.ylabel("feature weights")
plt.title("power law distribution followed by selected features")
plt.savefig("/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/project_repo/code/src/results/power_law_distribution.eps", format='eps')
plt.show()



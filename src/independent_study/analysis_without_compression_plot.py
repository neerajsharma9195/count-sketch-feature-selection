import matplotlib.pyplot as plt
import numpy as np
import json

with open("results/topk_feature_gradients_all_topk_8000.json", 'r') as f:
    data = json.loads(f.read())

D = 47236

for feature_pos in range(0, D):
    feature_gradients = data[str(feature_pos)]
    plt.hist(feature_gradients, color='green', label="# Features VS roc")
    plt.xlabel('iterations')
    plt.ylabel('gradient updates feature {}'.format(feature_pos))
    plt.legend()
    plt.savefig('plots/hists/gradient_update_{}.jpg'.format(feature_pos))
    plt.clf()

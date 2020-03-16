import matplotlib.pyplot as plt
import numpy as np
import json

with open("dumps/topK_logistic/topk_logistic_regression_gradient_updates2020-03-08 15:52:11.832886.json", 'r') as f:
    data = json.loads(f.read())

D = 47236

for feature_pos in range(1, D + 1):
    if str(feature_pos) in data:
        feature_gradients = data[str(feature_pos)]
        plt.plot(feature_gradients, color='green', label="# Features VS roc")
        plt.xlabel('iterations')
        plt.ylabel('gradient updates feature {}'.format(feature_pos))
        plt.legend()
        plt.savefig('dumps/topk_plots_logistic_regression/gradient_update_{}.jpg'.format(feature_pos))
        plt.clf()

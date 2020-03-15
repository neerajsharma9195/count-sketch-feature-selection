import matplotlib.pyplot as plt
import numpy as np
import json

with open("dumps/logistic_regression_gradient_updates2020-03-08 16:19:19.285065.json", 'r') as f:
    data = json.loads(f.read())


D = 47236


for feature_pos in range(1, D+1):
    if str(feature_pos) in data:
        feature_gradients = data[str(feature_pos)]
        plt.plot(feature_gradients, color='green', label="# Features VS roc")
        plt.xlabel('iterations')
        plt.ylabel('gradient updates feature {}'.format(feature_pos))
        plt.legend()
        plt.savefig('dumps/plots_logistic/gradient_update_{}.jpg'.format(feature_pos))
        plt.clf()
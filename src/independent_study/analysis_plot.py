import matplotlib.pyplot as plt
import numpy as np
import json


with open("/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/project_repo/src/independent_study/results/topk_feature_gradients_all_topk_2000.json", 'r') as f:
    data = json.loads(f.read())

feature_pos = '11106'



feature_gradients = data[feature_pos]

plt.plot(feature_gradients, '--', color='green', label="# Features VS roc")
plt.xlabel('iterations')
plt.ylabel('gradient updates feature {}'.format(feature_pos))
plt.legend()
# plt.savefig('gradient_update_{}.jpg'.format(feature_pos))
plt.show()

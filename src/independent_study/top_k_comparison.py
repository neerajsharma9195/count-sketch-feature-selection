feature_poses = []
feature_vals = []
with open("results/topk_features_8000.txt", 'r') as f:
    for line in f:
        pos, val = line.split(":")
        feature_poses.append(float(pos))
        feature_vals.append(abs(float(val.strip())))


import numpy as np

min_pos = np.argmin(np.array(feature_vals))

print(min_pos, feature_vals[min_pos], feature_poses[min_pos])
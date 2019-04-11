from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = make_blobs(centers=2)
features = pd.DataFrame(dataset[0])
labels = dataset[1]

data = np.random.randint(-9, 9, size=(1, 2))


def k_nn(features, labels, data, clusters=2, k=3):
    diff = features - data
    eucrid_distance = [np.linalg.norm([row[0], row[1]], ord=2) for index, row in diff.iterrows()]
    features[2] = labels
    features[3] = eucrid_distance
    features = features.sort_values(by=3)
    features = features.reset_index()
    counts = np.zeros(clusters)

    for i in range(k):
        counts[features.loc[i, 2]] += 1
    pred = np.argmax(counts)
    return pred

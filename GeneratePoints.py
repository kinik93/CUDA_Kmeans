from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

flp = open('points.txt', 'w')
centers = [[-3, 4], [2 , 3], [-1, -4], [7, 12], [3, -5] ]

#make_blobs generate isotropic Gaussian blobs for clustering.
#See documentation at: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
X, y = make_blobs(n_samples=200000, centers=centers, cluster_std=0.7)

#plot the points generated
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolors='k')
plt.show()

#Write the points in the points_k3.txt file
for i in range(0, len(X[:, 0])):
    flp.write(np.array_str(X[i][0]) + "," + np.array_str(X[i][1]) + "\n")


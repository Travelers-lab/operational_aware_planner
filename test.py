import numpy as np

data = np.random.randn(58, 3, 2)

displacement_diffs = data[1:, 0, :] - data[:-1, 0, :]
displacement_diffs = np.concatenate([displacement_diffs[:, np.newaxis, :] , data[1:, 1:, :]], axis=1)
euclidean_distances = np.sqrt(np.sum(displacement_diffs ** 2, axis=2))
print(euclidean_distances.shape)
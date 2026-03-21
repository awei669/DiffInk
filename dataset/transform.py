import math
import numpy as np

class Transform:
    def __init__(self, data_fixed_length, prob=0.5):
        self.data_fixed_length = data_fixed_length
        self.prob = prob

    def random_scaling(self, data, scale_range=(0.9, 1.1)):
        scale = np.random.uniform(*scale_range)
        data[:, :2] *= scale
        return data

    def random_rotation(self, data, angle_range=(-math.pi / 36, math.pi / 36)):
        angle = np.random.uniform(*angle_range)
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])
        data[:, :2] = np.dot(data[:, :2], rotation_matrix)
        return data
    
    def point_dropping(self, data, drop_prob=0.02):
        if len(data) < 10:
            return data
        keep_mask = np.random.rand(len(data)) > drop_prob
        if keep_mask.sum() < 2:
            keep_mask[np.random.choice(len(data), 2, replace=False)] = True
        coords = data[:, :2]
        states = data[:, 2:]
        kept_idx = np.where(keep_mask)[0]
        x_interp = np.interp(np.arange(len(data)), kept_idx, coords[kept_idx][:, 0])
        y_interp = np.interp(np.arange(len(data)), kept_idx, coords[kept_idx][:, 1])
        coords_interp = np.stack([x_interp, y_interp], axis=1)
        return np.hstack((coords_interp, states))

    def augment_data(self, data):
        augmented_data = data.copy()
        methods = [
            self.random_scaling,    # global scaling
            self.random_rotation,   # geometric rotation
            # self.point_dropping,    # trajectory sparsification + interpolation
        ]

        for method in methods:
            if np.random.rand() < self.prob:
                augmented_data = method(augmented_data)
        return augmented_data

    def pad_or_truncate(self, data):
        if len(data) > self.data_fixed_length:
            return data[:self.data_fixed_length]
        else:
            padding = [[0] * 5] * (self.data_fixed_length - len(data))
            return np.concatenate([data, padding])

    def __call__(self, data):
        augmented_data = self.augment_data(data)
        return augmented_data
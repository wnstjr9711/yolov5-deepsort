import torch
import numpy as np


def xy2wh(point):
    """
    Convert [x1, y1, x2, y2] to [x_center, y_center, width, height]
    :param point: [x1, y1, x2, y2]
    :return: [x_center, y_center, width, height]
    """
    y = point.clone() if isinstance(point, torch.Tensor) else np.copy(point)
    y[:, 0] = (point[:, 0] + point[:, 2]) / 2  # x center
    y[:, 1] = (point[:, 1] + point[:, 3]) / 2  # y center
    y[:, 2] = point[:, 2] - point[:, 0]  # width
    y[:, 3] = point[:, 3] - point[:, 1]  # height
    return y


if __name__ == '__main__':
    print(xy2wh(torch.tensor([[1, 2, 3, 4]])))

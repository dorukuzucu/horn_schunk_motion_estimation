import cv2
import numpy as np
from src.image.utils import derivative_hs


# TODO add clipping
class HornSchunkFrame():
    def __init__(self, alpha, num_iter):
        self.alpha = alpha
        self.num_iter = num_iter

    def compute_derivatives(self, anchor, target):
        i_x = derivative_hs(img1=anchor, img2=target, direction='x')
        i_y = derivative_hs(img1=anchor, img2=target, direction='y')
        i_t = derivative_hs(img1=anchor, img2=target, direction='t')

        return i_x, i_y, i_t

    def weighted_average(self, frame):
        filter = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16
        return cv2.filter2D(src=frame, kernel=filter, ddepth=-1)

    def __call__(self, anchor, target):
        if type(anchor) is not np.ndarray:
            raise Exception("Please input a numpy nd array")
        if type(target) is not np.ndarray:
            raise Exception("Please input a numpy nd array")

        assert anchor.shape == target.shape, "Frame shapes do not match"
        assert len(anchor.shape) == 2, "Frames should be gray"

        u_k = np.zeros_like(anchor)
        v_k = np.zeros_like(anchor)

        i_x, i_y, i_t = self.compute_derivatives(anchor, target)

        for iteration in range(self.num_iter):
            u_avg = self.weighted_average(u_k)
            v_avg = self.weighted_average(v_k)

            numerator = i_x * u_avg + i_y * v_avg + i_t
            denominator = self.alpha ** 2 + i_x ** 2 + i_y ** 2

            u_k = u_avg - i_x * (numerator / denominator)
            v_k = v_avg - i_y * (numerator / denominator)

            u_k[np.isnan(u_k)] = 0
            v_k[np.isnan(v_k)] = 0

        return u_k, v_k


def reconstruct(anchor, u, v):
    assert u.shape == v.shape

    height, width = u.shape
    reconstructed_img = np.zeros_like(u)

    for h in range(height):
        for w in range(width):
            new_h = int(h + u[h, w])
            new_w = int(w + v[h, w])

            new_h = max(new_h, 0)
            new_h = min(new_h, height - 1)

            new_w = max(new_w, 0)
            new_w = min(new_w, width - 1)

            reconstructed_img[new_h, new_w] = anchor[h, w]
    return reconstructed_img

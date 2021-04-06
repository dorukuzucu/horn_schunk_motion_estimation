import cv2
import numpy as np


def derivative_hs(img1, img2, direction):
    assert img1.shape == img2.shape, "Frame shapes do not match"
    assert len(img1.shape) == 2, "Frames should be gray"
    if direction not in ('x', 'y', 't'):
        raise Exception("Please enter a valid direction: x, y")

    if direction == 'x':
        hs_filter = np.array([[-1, 1], [-1, 1]]) * 0.25
    elif direction == 'y':
        hs_filter = np.array([[-1, -1], [1, 1]]) * 0.25
    elif direction == 't':
        hs_filter = np.array([[1, 1], [1, 1]]) * 0.25

    if direction in ('x', 'y'):
        der_1 = cv2.filter2D(src=img1, kernel=hs_filter, ddepth=-1)
        der_2 = cv2.filter2D(src=img2, kernel=hs_filter, ddepth=-1)
    else:
        der_1 = cv2.filter2D(src=img1, kernel=-hs_filter, ddepth=-1)
        der_2 = cv2.filter2D(src=img2, kernel=hs_filter, ddepth=-1)

    return der_1 + der_2


def rgb2gray(img):
    """
    :param img: input 3 channel to be turned into gray
    :return: gray image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def blur_img(img):
    """
    :param img: input image to be blurred
    :return: blurred image
    """
    return cv2.GaussianBlur(img, (3, 3), 0)


class MotionVector:
    def __init__(self, start, end):
        self.start = tuple(start)
        self.end = tuple(end)

    def range(self):
        return np.sqrt(np.square(np.subtract(self.end, self.start)).sum())


def uv_2_motion_vector(u, v):
    assert u.shape == v.shape

    height, width = u.shape
    vectors = []

    for h in range(height):
        for w in range(width):
            vec = MotionVector([h, w], [h + u[h,w], w + v[h,w]])
            vectors.append(vec)
    return vectors


def draw_motion_vector(image, vector, color=(0, 0, 255), thickness=1):
    try:
        if vector.range == 0.0:
            image = cv2.circle(image, center=vector.start, radius=0, color=color, thickness=thickness)
        else:
            image = cv2.arrowedLine(image, pt1=vector.start, pt2=vector.end, color=color, thickness=thickness)
        return image
    except Exception:
        print("Make sure you input list of vectors!")
        return None


def draw_motion_vectors(image, u, v, color=(0, 0, 255), thickness=1):
    motion_vectors = uv_2_motion_vector(u=u, v=v)
    if isinstance(motion_vectors, list):
        for vector in motion_vectors:
            image = draw_motion_vector(image=image, vector=vector, color=color, thickness=thickness)
        return image
    else:
        return None

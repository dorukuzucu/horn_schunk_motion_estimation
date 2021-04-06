import cv2
import numpy as np

def img_mse(img1, img2):
    """
        :param img1: input image 1 as numpy array
        :param img2: input image 2 to be compared as numpy array
            ALL IMAGE PIXEL VALUES SHOULD BE BETWEEN 0-1. (IMG/255)
        :return: MSE value for input images
        """
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def img_psnr(img1, img2):
    """
    :param img1: input image 1 numpy array
    :param img2: input image 2 to be compared numpy array
        ALL IMAGE PIXEL VALUES SHOULD BE BETWEEN 0-1. (IMG/255)
    :return: PSNR value for input images
    """
    # Own implementation
    mse = img_mse(img1,img2)
    return 10 * np.log10(1. / mse)

from matplotlib import pyplot as plt
from skimage import data, exposure
import numpy as np
from scipy import stats
from PIL import Image
import cv2


# 中文显示工具函数
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False


set_ch()


def isValid(X, Y, point):
    """
    判断某个像素是否超出图像空间范围
    :param X:
    :param Y:
    :param point:
    :return:
    """
    if point[0] < 0 or point[0] >= X:
        return False
    if point[1] < 0 or point[1] >= Y:
        return False
    return True


def getNeighbors(X, Y, x, y, dist):
    """
    Find pixel neighbors according to various distances
    :param X:
    :param Y:
    :param x:
    :param y:
    :param dist:
    :return:
    """
    cn1 = (x + dist, y + dist)
    cn2 = (x + dist, y)
    cn3 = (x + dist, y - dist)
    cn4 = (x, y - dist)
    cn5 = (x - dist, y - dist)
    cn6 = (x - dist, y)
    cn7 = (x - dist, y + dist)
    cn8 = (x, y + dist)
    points = (cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8)
    Cn = []
    for i in points:
        if (isValid(X, Y, i)):
            Cn.append(i)
    return Cn


def corrlogram(image, dist):
    XX, YY, tt = image.shape
    cgram = np.zeros((XX, YY), dtype=np.int)
    for x in range(XX):
        for y in range(YY):
            for t in range(tt):
                color_i = image[x, y, t]
                neighbors_i = getNeighbors(XX, YY, x, y, dist)
                for j in neighbors_i:
                    j0 = j[0]
                    j1 = j[1]
                    color_j = image[j0, j1, t]
                    cgram[color_i, color_j] = cgram[color_i, color_j] + 1
    return cgram


class ColorFeature(object):
    def __init__(self, dish):
        self.dish = dish

    def __call__(self, img):
    
        #h,w = img_gray.shape
        # 计算灰度共生矩阵和 GLCM 特征
        dist = 4
        cgram = corrlogram(img, dist)
        # color_feature = color_moments(img)
        cgram = cv2.cvtColor(cgram, cv2.COLOR_GRAY2RGB)

        #img = img.convert('RGB')
        
        return  cgram

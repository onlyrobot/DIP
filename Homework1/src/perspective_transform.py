'''
Author: 彭瑶
Date: 2019/12/12
Description: 将图像数字透视转换
'''

import sys
import numpy as np 
import matplotlib.pyplot as plt


def transform(img, normal_vec, view_point):
    '''将图像透视转换

    除了灰度图像外还支持多通道图像
        
    Args:
        img: 输入灰度图像或多通道图像
        normal_vec: 裁剪平面的法向
        view_point: 相机位置
    Returns:
        返回转换后的图像
    '''
    trans = np.empty((img.shape[0], img.shape[1], 3), np.float)
    for x in range(trans.shape[0]):
        for y in range(trans.shape[1]):
            origin = np.array((x, y, 0), np.int16)
            k = 1 + 1 / np.dot(normal_vec, view_point - origin)
            trans[x][y] = k * (view_point - origin) + origin
    
    trans -= trans[0, 0]
    axisx = trans[-1, 0] / np.sqrt(trans[-1, 0].dot(trans[-1, 0]))
    axisy = trans[0, -1] / np.sqrt(trans[0, -1].dot(trans[0, -1]))
    trans[:, :, 0], trans[:, :, 1] = trans.dot(axisx), trans.dot(axisy)

    min_x, min_y, min_z = np.min(trans[(0, 0, -1, -1), (0, -1, 0, -1)], axis=0)
    max_x, max_y, _ = np.max(trans[(0, 0, -1, -1), (0, -1, 0, -1)], axis=0)
    ratio = img.shape[0] / (max_x - min_x)
    trans = np.array((trans - [min_x, min_y, min_z]) * ratio, np.int16)
    height = np.int16((max_x - min_x) * ratio)
    width = np.int16((max_y - min_y) * ratio)
    
    if len(img.shape) == 2:    # 灰度图像
        new_img = np.zeros((height + 1, width + 1), dtype=np.uint8)
    else:    # 多通道图像
        new_img = np.zeros((height + 1, width + 1, img.shape[2]), dtype=np.uint8)

    for x in range(trans.shape[0]):
        for y in range(trans.shape[1]):
            new_img[tuple(trans[x, y, (0, 1)])] = img[x, y]
    return new_img


def main():
    img = plt.imread(sys.path[0] + '/../res/1.jpg')
    # 透视变换，参数包含图像、裁剪平面的法向以及相机位置
    new_img = transform(img, (-1, -1, -5), (500, 500, 500))
    axes1 = plt.subplot(121)
    axes1.imshow(img, cmap='gray')
    axes2 = plt.subplot(122)
    axes2.imshow(new_img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
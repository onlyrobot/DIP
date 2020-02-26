'''
Author: 彭瑶
Date: 2019/12/12
Description: 基于K-means算法的图像分割
'''


import sys
import numpy as np 
import matplotlib.pyplot as plt 


def dist(x, y, img, means):
    '''获取图像img中坐标为x,y的像素点到均值means的距离

    Args:
        x: 像素点坐标x
        y: 像素点坐标y
        img: 输入图像
        means: 聚类的均值

    Returns:
        返回像素点到各个均值的距离
    '''
    dists = np.sqrt(np.sum(np.square(means - img[x, y]), axis=1))
    return dists


def seg(img, k):
    '''用k均值算法对图像进行分割

    Args:
        img: 待分割的图像
        k: K-means算法的k

    Returns:
        返回分割后的图像
    '''
    # clusters矩阵保存每个像素点所属的聚类
    clusters = np.random.randint(0, k, img.shape[: 2])
    means = np.empty((k, 1 if len(img.shape) == 2 else img.shape[2]))
    while True:
        for i in range(k):
            index = np.where(clusters == i)
            if index[0].size > 0:
                means[i] = np.mean(img[np.where(clusters == i)], axis=0)

        changed = False
        for x in range(clusters.shape[0]):
            for y in range(clusters.shape[1]):
                dists = dist(x, y, img, means)
                pos = np.argmin(dists)
                if pos != clusters[x, y]:
                    clusters[x, y] = pos
                    changed = True
        
        if not changed:
            break

    target = np.uint8(means)[clusters].reshape(img.shape)
    return target


def main():
    img = plt.imread(sys.path[0] + '/../res/lena.jpg')
    target1 = seg(img, 3)
    target2 = seg(img, 5)
    plt.subplot(131)
    plt.imshow(img, 'gray')
    plt.title('Origin')
    plt.subplot(132)
    plt.imshow(target1, 'gray')
    plt.title('Target K=3')
    plt.subplot(133)
    plt.imshow(target2, 'gray')
    plt.title('Target K=5')
    plt.show()


if __name__ == '__main__':
    main()
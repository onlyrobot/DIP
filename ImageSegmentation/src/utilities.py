'''
Author: 彭瑶
Date: 2019/1/20
Description: 图像分割的实用函数
'''


import sys
import numpy as np
import matplotlib.pyplot as plt


def guassian_blur(img, radius, sigma):
    '''高斯模糊函数

    Args:
        radius: 高斯核的半径
        sigma: 高斯分布的sigma
        img: 待模糊图像
    '''
    height, width, size = img.shape[0], img.shape[1], radius * 2 + 1
    if len(img.shape) < 3:
        filter, chanel = np.empty((size, size)), 1
    else:
        filter, chanel = np.empty((size, size, img.shape[2])), img.shape[2]
    for i in range(size):
        for j in range(size):
                x, y = i - radius, j - radius
                filter[i, j] = (1 / np.sqrt(2 * np.pi * sigma * sigma) * 
                np.exp(-(x * x + y * y) / (2 * sigma * sigma)))
    filter = filter / filter.sum() * chanel
    output = np.empty(img.shape) 
    
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            output[i, j] = (filter * img[i - radius: i + radius + 1, 
            j - radius: j + radius + 1]).sum(axis=(0, 1))
    return output


class DisjointSet:
    '''并查集

    Attributes:
        self.array: 并查集的关键数组
        self.sizes: 每个元素所属的集合元素个数
        self.values: 每个元素所属的集合的取值
    '''
    def __init__(self, n, value=None):
        self.array = [i for i in range(n)]
        self.sizes = [1 for i in range(n)]
        self.values = [value for i in range(n)]

    def find(self, a):
        '''从并查集中寻找元素所属的集合'''
        parent = self.array[a]
        while a != parent:
            self.array[a] = a = self.array[parent]
            parent = self.array[a]
        return a

    def merge(self, a, b, value=None):
        '''合并两个元素，并可选择指定合并之后的集合取值

        Args:
            a: 元素a
            b: 元素b
            value: 合并后他们所属集合的取值
        '''  
        parent_a, parent_b = self.find(a), self.find(b)
        if parent_a != parent_b:
            self.array[parent_b] = parent_a
            self.sizes[parent_a] += self.sizes[parent_b]
            self.values[parent_a] = value

    def get_size(self, a):
        '''得到元素a所属集合的大小'''
        parent = self.find(a)
        return self.sizes[parent]

    def __setitem__(self, key, value):
        '''设定元素key所属集合的取值'''
        parent = self.find(key)
        self.values[parent] = value 

    def __getitem__(self, key):
        '''得到元素key所属集合的取值'''
        parent = self.find(key)
        return self.values[parent]


def main():
    img = plt.imread(sys.path[0] + '/../res/img/origin.png')
    plt.gray()
    plt.subplot(121)
    plt.imshow(img)
    img = guassian_blur(img, 7, 5)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()
    pass 


if __name__ == '__main__':
    main()
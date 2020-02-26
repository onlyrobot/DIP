'''
Author: 彭瑶
Date: 2020/1/20
Description: 图相关的函数，用于图像分割
'''
 

import sys
import matplotlib.pyplot as plt 
import numpy as np
import utilities as utl 


class Edge:
    '''边
    表示相邻像素点的连接边

    Attributes:
        self.a: 像素点1的id
        self.b: 像素点2的id
        self.w: 该边的权重
    '''
    def __init__(self, a, b, w):
        self.a, self.b, self.w = a, b, w


def gen_edges(img, diff):
    '''将图片中相邻的像素点用边连接，得到一个联通图

    Args:
        img: 待处理图片
        diff: 用于衡量像素点之间的差别
    
    Returns:
        所有边的列表
    '''
    edges = []
    height, width = img.shape[: 2]
    get_id = lambda x, y: x * width + y
    for i in range(height):
        for j in range(width):
            if i > 0:
                weight = diff(img, (i, j), (i - 1, j))
                edges.append(Edge(get_id(i - 1, j), get_id(i, j), weight))
            if j > 0:
                weight = diff(img, (i, j), (i, j - 1))
                edges.append(Edge(get_id(i, j - 1), get_id(i, j), weight))
            # other case
    return edges 


def gen_img(ds, img):
    '''由像素点组成的并查集ds来还原分割后的图片

    Args:
        ds: 像素点组成的并查集
        img: 原图片

    Returns:
        返回还原好的图片
    '''
    # image = np.empty(img.shape)
    image = img.copy()
    height, width = img.shape[: 2]
    colors, counts = {}, {}
    for i in range(height):
        for j in range(width):
            parent = ds.find(i * width + j)
            if parent not in colors:
                colors[parent] = img[i, j]
                counts[parent] = 1
            else:
                colors[parent] = colors[parent] + img[i, j]
                counts[parent] += 1
    for parent in colors:
        colors[parent] = colors[parent] / counts[parent]
    for i in range(height):
        for j in range(width):
            parent = ds.find(i * width + j)
            image[i, j] = colors[parent]
    if image.max() > 1:
        image = np.array(image, dtype=np.uint8)
    return image


def seg_img(img, k, min_pixels, radius, sigma, diff=lambda img, xy, xxyy: 
    np.sqrt(np.sum((0. + img[xy] - img[xxyy]) ** 2)), 
    threshold=lambda k, size: k / size):
    '''图像分割的主调函数

    Args:
        img: 待分割的图像
        k: k值，配合内聚函数使用
        min_pixels: 最小的像素点聚类个数
        diff: 用于衡量像素点之间的差异程度
        threshold: 衡量像素点聚类的内聚度函数

    Returns:
        返回分割好的图像
    '''
    # 先进行高斯模糊处理
    img = utl.guassian_blur(img, radius, sigma)
    # 得到图像的所有边集
    edges = gen_edges(img, diff)
    height, width = img.shape[: 2]
    # 生成并查集
    ds = utl.DisjointSet(height * width, threshold(k, 1))
    # 将所有的边排好序
    edges.sort(key=lambda edge: edge.w)
    # 开始基于图的分割过程
    for edge in edges:
        pa, pb = ds.find(edge.a), ds.find(edge.b)
        if edge.w <= ds[pa] and edge.w <= ds[pb] and pa != pb:
            ds.merge(pa, pb)
            ds[pa] = edge.w + threshold(k, ds.get_size(pa))
    # 对小区域的聚类进行合并
    for edge in edges:
        pa, pb = ds.find(edge.a), ds.find(edge.b)
        if pa != pb and ds.get_size(pa) < min_pixels and ds.get_size(pb) < min_pixels:
            ds.merge(pa, pb)
    return gen_img(ds, img)


def main():
    # 显示原图
    img = plt.imread(sys.path[0] + '/../res/img/origin.png')
    plt.subplot(121)
    plt.imshow(img)
    # 显示分割好的图
    segged_img = seg_img(img, 3, 1000, 2, 1)
    plt.subplot(122)
    plt.imshow(segged_img)

    plt.show()


if __name__ == '__main__':
    main()
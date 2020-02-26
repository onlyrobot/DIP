import matplotlib.pyplot as plt
import numpy as np
import math
import sys


def guassian_blur(radius, sigma, img):
    '''高斯模糊函数

    Args:
        radius: 高斯核的半径
        sigma: 高斯分布的sigma
        img: 待模糊图像
    '''
    size = radius * 2 + 1
    filter = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            x, y = i - radius, j - radius
            filter[i, j] = (1 / math.sqrt(2 * math.pi * sigma * sigma) * 
            math.exp(-(x * x + y * y) / (2 * sigma * sigma)))
    filter /= filter.sum()
    width, height = img.shape 
    output = np.empty(img.shape) 
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            output[i, j] = (filter * img[i - radius: i + radius + 1, 
            j - radius: j + radius + 1]).sum()
    return output


def add_noise(img):
    '''为图像添加高斯白噪声'''
    return img + 5 * np.random.standard_normal(img.shape)


def wiener(blur_noise, img, blur, k):
    '''维纳滤波
    
    Args:
        blur_noise: 经过高斯模糊以及添加白噪声后的图像
        img: 原图像
        blur: 只经过高斯模糊的图像
        k: k值
    '''
    fft = np.fft.fft2(img)
    blur_fft = np.fft.fft2(blur)
    h = np.fft.ifft2(blur_fft / fft)
    h_fft = np.fft.fft2(h)
    H = np.abs(h_fft) ** 2

    blur_noise_fft = np.fft.fft2(blur_noise)
    
    output = np.conj(h_fft) / (H + k)
    output = np.abs(np.fft.ifft2(output * blur_noise_fft))
    return output


def main():
    plt.gray()
    # 原始图像
    img = plt.imread(sys.path[0] + '/../res/lena.jpg')
    plt.subplot(141)
    plt.title("Original Image")
    plt.imshow(img)
    # 高斯模糊后的图像
    blur = guassian_blur(4, 7, img)
    plt.subplot(142)
    plt.title('Gaussian Blur')
    plt.imshow(blur)
    # 高斯模糊和添加噪声的图像
    blur_noise = add_noise(blur)
    plt.subplot(143)
    plt.title('Gaussian & Noise')
    plt.imshow(blur_noise)
    # 维纳滤波后的图像
    output = wiener(blur_noise, img, blur, 0.05)
    plt.subplot(144)
    plt.title('Wiener Filter')
    plt.imshow(output)
    # 显示所有图像
    plt.show()


if __name__ == '__main__':
    main()
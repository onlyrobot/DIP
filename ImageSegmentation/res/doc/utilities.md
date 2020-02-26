Help on module utilities:

NAME
    utilities

DESCRIPTION
    Author: 彭瑶
    Date: 2019/1/20
    Description: 图像分割的实用函数

CLASSES
    builtins.object
        DisjointSet
    
    class DisjointSet(builtins.object)
     |  并查集
     |  
     |  Attributes:
     |      self.array: 并查集的关键数组
     |      self.sizes: 每个元素所属的集合元素个数
     |      self.values: 每个元素所属的集合的取值
     |  
     |  Methods defined here:
     |  
     |  __getitem__(self, key)
     |      得到元素key所属集合的取值
     |  
     |  __init__(self, n, value=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __setitem__(self, key, value)
     |      设定元素key所属集合的取值
     |  
     |  find(self, a)
     |      从并查集中寻找元素所属的集合
     |  
     |  get_size(self, a)
     |      得到元素a所属集合的大小
     |  
     |  merge(self, a, b, value=None)
     |      合并两个元素，并可选择指定合并之后的集合取值
     |      
     |      Args:
     |          a: 元素a
     |          b: 元素b
     |          value: 合并后他们所属集合的取值
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    guassian_blur(img, radius, sigma)
        高斯模糊函数
        
        Args:
            radius: 高斯核的半径
            sigma: 高斯分布的sigma
            img: 待模糊图像
    
    main()

FILE
    /home/onlyrobot/E/Project/DigitalImageProcessing/image_segmentation/src/utilities.py



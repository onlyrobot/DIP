Help on module graph:

NAME
    graph

DESCRIPTION
    Author: 彭瑶
    Date: 2020/1/20
    Description: 图相关的函数，用于图像分割

CLASSES
    builtins.object
        Edge
    
    class Edge(builtins.object)
     |  边
     |  表示相邻像素点的连接边
     |  
     |  Attributes:
     |      self.a: 像素点1的id
     |      self.b: 像素点2的id
     |      self.w: 该边的权重
     |  
     |  Methods defined here:
     |  
     |  __init__(self, a, b, w)
     |      Initialize self.  See help(type(self)) for accurate signature.
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
    gen_edges(img, diff)
        将图片中相邻的像素点用边连接，得到一个联通图
        
        Args:
            img: 待处理图片
            diff: 用于衡量像素点之间的差别
        
        Returns:
            所有边的列表
    
    gen_img(ds, img)
        由像素点组成的并查集ds来还原分割后的图片
        
        Args:
            ds: 像素点组成的并查集
            img: 原图片
        
        Returns:
            返回还原好的图片
    
    main()
    
    seg_img(img, k, min_pixels, radius, segma, diff=<function <lambda> at 0x7f3780ff3a60>, threshold=<function <lambda> at 0x7f3780ff3ae8>)
        图像分割的主调函数
        
        Args:
            img: 待分割的图像
            k: k值，配合内聚函数使用
            min_pixels: 最小的像素点聚类个数
            diff: 用于衡量像素点之间的差异程度
            threshold: 衡量像素点聚类的内聚度函数
        
        Returns:
            返回分割好的图像

FILE
    /home/onlyrobot/E/Project/DigitalImageProcessing/image_segmentation/src/graph.py



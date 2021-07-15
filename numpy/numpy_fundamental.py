#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-14
################################################################

import numpy as np


def main():
    # 创建矩阵&向量
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)

    # 矩阵乘法
    result = np.dot(a, b)
    print(result)

    # 对应位置元素相乘
    result = np.multiply(a, b)
    print(result)


if __name__ == '__main__':
    main()
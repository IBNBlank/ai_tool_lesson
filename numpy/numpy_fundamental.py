#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-14
################################################################

import numpy as np


def main():
    # 初始化
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)

    # 矩阵乘法
    np.dot(a, b)

    # 对应位置元素相乘
    np.multiply(a, b)


if __name__ == '__main__':
    main()
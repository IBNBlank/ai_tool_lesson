#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-14
################################################################

import torch


def main():
    # 创建矩阵
    a = torch.rand(2, 2)
    b = torch.rand(2, 2)

    # 矩阵乘法
    result = torch.matmul(a, b)
    print(result)

    # 对应位置元素相乘
    result = a * b
    print(result)

    # 梯度计算
    x = torch.tensor(-3., requires_grad=True)
    y = torch.tensor(5., requires_grad=True)
    z = torch.tensor(-2., requires_grad=True)
    q = x + y
    f = q * z
    f.backward()
    print(f"gradient of x:{x.grad}")
    print(f"gradient of y:{y.grad}")
    print(f"gradient of z:{z.grad}")


if __name__ == '__main__':
    main()

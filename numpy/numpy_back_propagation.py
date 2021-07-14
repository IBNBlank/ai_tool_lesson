#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-14
################################################################

import numpy as np


class NumpyBackPropagation(object):
    def __init__(self):
        # 模型参数
        self.__weight = np.array([1.0, 2.0])
        # 梯度下降参数
        self.__learning_rate = 0.01

    def back_propagation(self, input_data, label):
        # 误差计算
        predict = self.__forward_propagation(input_data)
        error = predict - label
        print(f"origin_error:{error}")

        # 梯度计算
        gradient = 2 * input_data.T * error

        # 参数计算
        self.__weight = self.__weight - self.__learning_rate * gradient

        # 检查新误差
        predict = self.__forward_propagation(input_data)
        error = predict - label
        print(f"new_error:{error}")

    def __forward_propagation(self, input_data):
        output_data = self.__weight.dot(input_data)
        return output_data

    def __activatino_function(self, input_data):
        output_data = input_data
        # tanh
        # output_data = np.tanh(input_data)
        # relu
        output_data = np.maximum(input_data, 0)
        return output_data


def main():
    # 输入
    input_data = np.array([[3], [4]])
    label = 6

    # 模型初始化
    numpy_back_propagation = NumpyBackPropagation()

    # 反向传播
    numpy_back_propagation.back_propagation(input_data, label)
    numpy_back_propagation.back_propagation(input_data, label)


if __name__ == '__main__':
    main()
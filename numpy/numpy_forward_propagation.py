#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-14
################################################################

import numpy as np


class NumpyForwardPropagation(object):
    def __init__(self):
        # 模型参数
        self.__weight = {
            'w1': np.array([[2., 4.], [4., -5.]]),
            'w2': np.array([2., 7.])
        }
        self.__bias = {'b1': np.array([[0.], [0.]]), 'b2': np.array([0.])}

    def forward_propagation(self, input_data):
        # input size [[x1], [x2]]
        hidden_layer_input = self.__weight['w1'].dot(
            input_data) + self.__bias['b1']
        hidden_layer_output = self.__activatino_function(hidden_layer_input)
        output_data = self.__weight['w2'].dot(
            hidden_layer_output) + self.__bias['b2']
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
    input_data = np.array([[1.], [3.]])

    # 模型初始化
    numpy_forward_propagation = NumpyForwardPropagation()

    # 前向传播
    print(numpy_forward_propagation.forward_propagation(input_data))


if __name__ == '__main__':
    main()
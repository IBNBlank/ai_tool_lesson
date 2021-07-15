#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-15
################################################################

import torch
from torch.autograd import grad


class TorchManualNetwork(object):
    def __init__(self):
        # 模型参数
        self.__weight = {
            'w1': torch.tensor([[1., 1.], [-1., 1.]], requires_grad=True),
            'w2': torch.tensor([2., -1.], requires_grad=True)
        }
        self.__bias = {
            'b1': torch.tensor([0., 0.], requires_grad=True),
            'b2': torch.tensor([0.], requires_grad=True)
        }

        # 梯度下降参数
        self.__learning_rate = 0.001

    def forward_propagation(self, input_data):
        hidden_layer_1_input = torch.matmul(self.__weight['w1'],
                                            input_data) + self.__bias['b1']
        hidden_layer_1_output = self.__activatino_function(
            hidden_layer_1_input)
        hidden_layer_2_input = torch.matmul(
            self.__weight['w2'], hidden_layer_1_output) + self.__bias['b2']
        hidden_layer_2_output = self.__activatino_function(
            hidden_layer_2_input)
        output = (hidden_layer_2_output - 13)**2
        return output

    def back_propagation(self, data, label):
        network = self.forward_propagation(data)
        network.backward()
        # weight
        self.__weight['w1'] = self.__weight[
            'w1'] - self.__learning_rate * self.__weight['w1'].grad
        self.__weight['w1'].retain_grad()
        self.__weight['w2'] = self.__weight[
            'w2'] - self.__learning_rate * self.__weight['w2'].grad
        self.__weight['w2'].retain_grad()
        # bias
        self.__bias['b1'] = self.__bias[
            'b1'] - self.__learning_rate * self.__bias['b1'].grad
        self.__bias['b1'].retain_grad()
        self.__bias['b2'] = self.__bias[
            'b2'] - self.__learning_rate * self.__bias['b2'].grad
        self.__bias['b2'].retain_grad()

    def __activatino_function(self, input_data):
        output_data = input_data
        # tanh
        # output_data = np.tanh(input_data)
        # relu
        relu = torch.nn.ReLU()
        output_data = relu(input_data)
        return output_data


def main():
    # 输入
    input_data = torch.tensor([2., 3.])
    label = 0

    # 模型初始化
    torch_manual_network = TorchManualNetwork()

    # 前向传播
    print(
        f"origin_result:{torch_manual_network.forward_propagation(input_data)}"
    )

    # 反向传播
    torch_manual_network.back_propagation(input_data, label)

    # 前向传播
    print(f"new_result:{torch_manual_network.forward_propagation(input_data)}")


if __name__ == '__main__':
    main()
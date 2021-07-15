#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-15
################################################################

import torch
import torch.nn as nn


class TorchNnNetwork(nn.Module):
    def __init__(self):
        super(TorchNnNetwork, self).__init__()
        self.__activation_function = nn.ReLU()
        self.__loss_function = nn.MSELoss(reduction='sum')
        self.__learning_rate = 0.0001
        self.__model = nn.Sequential(nn.Linear(10,
                                               20), self.__activation_function,
                                     nn.Linear(20, 20),
                                     self.__activation_function,
                                     nn.Linear(20, 4))

    def forward(self, input_data):
        output = self.__model(input_data)
        return output

    def backward(self, input_data, label):
        predict = self.forward(input_data)
        loss = self.__loss_function(predict, label)
        self.__model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.__model.parameters():
                param -= self.__learning_rate * param.grad


def main():
    # 输入
    input_data = torch.tensor([2., 3., 2., 3., 2., 3., 2., 3., 2., 3.])
    label = torch.tensor([1.,1.,1.,1.])

    # 模型初始化
    torch_nn_work = TorchNnNetwork()

    # 前向传播
    print(f"result:{torch_nn_work.forward(input_data)}")

    # 反向传播
    torch_nn_work.backward(input_data, label)

    # 前向传播
    print(f"new_result:{torch_nn_work.forward(input_data)}")


if __name__ == '__main__':
    main()
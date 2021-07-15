#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-15
################################################################
"""
torchvision 包含 数据集&预训练网络
"""
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

MODEL_PATH = 'model/torch_image_classifier.pth'
PARAMETER_PATH = 'model/torch_image_classifier_parameters.pth'


class TorchImageClassifier(nn.Module):
    def __init__(self, cuda_flag=False):
        super(TorchImageClassifier, self).__init__()
        self.__model = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 10),
        )
        self.__loss_function = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=3e-4)

        # 是否使用显卡
        self.__cuda_flag = cuda_flag
        if self.__cuda_flag:
            self.__model = self.__model.cuda()
            self.__loss_function = self.__loss_function.cuda()

    def forward(self, input_data):
        # 是否使用显卡
        if self.__cuda_flag:
            input_data = input_data.cuda()

        output = self.__model(input_data)
        return output.cpu()

    def backward(self, input_data, label):
        # 重置优化器
        self.__optimizer.zero_grad()

        # 计算loss
        predict = self.forward(input_data)

        # 是否使用显卡
        if self.__cuda_flag:
            predict = predict.cuda()
            label = label.cuda()
        loss = self.__loss_function(predict, label)

        # 反向传播
        loss.backward()

        # 梯度下降更新参数
        self.__optimizer.step()

    def save_all(self, path):
        torch.save(self.__model, path)

    def save_parameters(self, path):
        torch.save(self.__model.state_dict(), path)

    def load_all(self, path):
        self.__model = torch.load(path)

    def load_parameters(self, path):
        self.__model.load_state_dict(torch.load(path))


# 读取数据集
def get_dataloader():
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.48216, 0.44653),
                             (0.24703, 0.24349, 0.26159))
    ])
    train_data = torchvision.datasets.CIFAR10(root="data",
                                              train=True,
                                              download=True,
                                              transform=my_transform)
    test_data = torchvision.datasets.CIFAR10(root="data",
                                             train=False,
                                             download=True,
                                             transform=my_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=32,
                                              shuffle=False,
                                              num_workers=4)
    return train_loader, test_loader


# 模型训练
def train_model(classifier, train_loader):
    for epoch in range(10):
        print(f"\tepoch: {epoch}")
        for i, data in enumerate(train_loader, 0):
            input_data, label = data
            input_data = input_data.view(-1, 32 * 32 * 3)
            print(f"\t\tbackward: {i}")
            classifier.backward(input_data, label)


# 模型验证
def verify_model(classifier, test_loader):
    correct, total = 0, 0
    for _, data in enumerate(test_loader, 0):
        input_data, label = data
        input_data = input_data.view(-1, 32 * 32 * 3)
        output = classifier.forward(input_data)

        _, predict = torch.max(output, 1)
        total += label.size(0)
        correct += (predict == label).sum().item()

    print(
        f'The testing set accuracy of the network is: {100 * correct / total}%'
    )


def main():
    # 数据集读取
    train_loader, test_loader = get_dataloader()

    # 模型导入
    print("Create Model")
    torch_image_classifier = TorchImageClassifier(cuda_flag=True)

    # 模型训练
    print("Train Model")
    train_model(torch_image_classifier, train_loader)

    # 保存模型
    print("Save Model")
    torch_image_classifier.save_all(MODEL_PATH)
    torch_image_classifier.save_parameters(PARAMETER_PATH)

    # 模型验证
    print("Verify Model")
    verify_model(torch_image_classifier, test_loader)


if __name__ == '__main__':
    main()
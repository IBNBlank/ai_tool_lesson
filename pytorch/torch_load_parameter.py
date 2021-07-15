#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-15
################################################################

import torch_image_classifier


def main():
    # 数据集读取
    _, test_loader = torch_image_classifier.get_dataloader()

    # 模型导入
    print("Create Model")
    torch_load_classifier = torch_image_classifier.TorchImageClassifier(
        cuda_flag=True)

    # 模型加载
    print("Load Parameter")
    torch_load_classifier.load_parameters(
        torch_image_classifier.PARAMETER_PATH)

    # 模型验证
    print("Verify Model")
    torch_image_classifier.verify_model(torch_load_classifier, test_loader)


if __name__ == '__main__':
    main()
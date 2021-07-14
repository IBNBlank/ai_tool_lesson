#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-12
################################################################
"""
整个流程如下:
1.读取数据集
2.数据划分
3.数据预处理
4.模型定义
5.模型训练
6.模型评价&误差分析
"""
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SklearnFundamental(object):
    def __init__(self, data_train, label_train):
        # 获取数据
        self.__data_train, self.__label_train = data_train, label_train

        # 数据预处理
        # 归一化只能fit一次(用训练集fit), 否则两次fit的标准不统一
        self.__scaler = preprocessing.StandardScaler().fit(self.__data_train)
        self.__data_train = self.data_preprocessing(self.__data_train)

        # 模型定义
        self.__knn = neighbors.KNeighborsClassifier(n_neighbors=3)

        # 模型训练
        self.__knn.fit(self.__data_train, self.__label_train)

    # 数据预处理
    def data_preprocessing(self, data):
        return self.__scaler.transform(data)

    # 对测试集进行预测, 并显示分数
    def predict_and_show(self, test_data, test_label):
        test_predict = self.__knn.predict(test_data)
        score = accuracy_score(test_label, test_predict)
        print(score)


# 读取数据集
def get_dataset():
    iris = datasets.load_iris()
    return iris.data[:, :], iris.target


# 数据划分
def data_partitioning(data, label):
    # 留出法
    return train_test_split(data, label, random_state=33)


def main():
    # 读取数据集
    data, label = get_dataset()

    # 数据划分
    data_train, data_test, label_train, label_test = data_partitioning(
        data, label)

    # 模型操作
    sklearn_fundamental = SklearnFundamental(data_train, label_train)

    # 模型评价&误差分析
    data_test = sklearn_fundamental.data_preprocessing(data_test)
    sklearn_fundamental.predict_and_show(data_test, label_test)


if __name__ == '__main__':
    main()
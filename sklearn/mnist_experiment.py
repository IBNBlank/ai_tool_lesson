#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-13
################################################################

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class MnistExperiment(object):
    def __init__(self, data_train, label_train):
        # 获取数据
        self.__data_train, self.__label_train = data_train, label_train

        # 显示数据(optional)
        self.__display_data()

        # PCA(optional)
        self.__pca()

        # MLP
        print("Creat MLP")
        self.__mlp = self.__mlp_creat()

        # KNN
        print("Creat KNN")
        self.__knn = self.__knn_creat()

        # RidgeClassifier
        print("Creat RidgeClassifier")
        self.__ridge_classifier = self.__ridge_classifier_creat()

    # MLP创建
    def __mlp_creat(self):
        classifier = MLPClassifier(solver='lbfgs',
                                   hidden_layer_sizes=(1000, ),
                                   random_state=1)
        return classifier.fit(self.__data_train / 255.0, self.__label_train)

    # KNN创建
    def __knn_creat(self):
        knn = neighbors.KNeighborsClassifier(n_neighbors=3)
        return knn.fit(self.__data_train, self.__label_train)

    # RidgeClassifier创建
    def __ridge_classifier_creat(self):
        classifier = RidgeClassifier()
        return classifier.fit(self.__data_train, self.__label_train)

    # 显示混淆矩阵
    def __show_confusion_matrix(func):
        def inner(*args, **kwargs):
            label_predict, label_test = func(*args, **kwargs)
            model_confusion_matrix = confusion_matrix(label_test, label_predict)
            fig, ax = plt.subplots(figsize=(10, 10))

            sn.heatmap(model_confusion_matrix, annot=True, ax=ax, fmt='g', vmin=0)
            ax.set_ylabel("true label")
            ax.set_xlabel("predicted label")

            plt.show()

        return inner

    # MLP预测
    @__show_confusion_matrix
    def mlp_predict(self, data_test, label_test):
        label_predict = self.__mlp.predict(data_test)
        return label_predict, label_test

    # KNN预测
    @__show_confusion_matrix
    def knn_predict(self, data_test, label_test):
        label_predict = self.__knn.predict(data_test)
        return label_predict, label_test

    # RidgeClassifier预测
    @__show_confusion_matrix
    def ridge_classifier_predict(self, data_test, label_test):
        label_predict = self.__ridge_classifier.predict(data_test)
        return label_predict, label_test

    # 显示图像
    def __display_data(self):
        # set up array
        fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
        fig.suptitle("Display randomly images of the training data set")
        # loop over randomly drawn numbers
        for row in range(10):
            for column in range(10):
                index = np.random.randint(self.__data_train.shape[0])
                number = self.__data_train[index, :].reshape(28, 28)
                ax[row, column].set_title(f"Label:{self.__label_train[index]}")
                ax[row, column].imshow(number, cmap='gray_r')
                plt.setp(ax[row, column].get_xticklabels(), visible=False)
                plt.setp(ax[row, column].get_yticklabels(), visible=False)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show()

    # PCA 分解
    def __pca(self):
        pca = PCA(n_components=2)
        proj = pca.fit_transform(self.__data_train)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.set_title("PCA of dataset n_components=2")
        ax.scatter(proj[:, 0],
                   proj[:, 1],
                   c=self.__label_train,
                   label='number')
        ax.legend()
        plt.show()


# 读取数据集
def get_dataset():
    mnist = pd.read_csv('./mnist-train.csv.zip')
    label = mnist['label'].values.flatten()
    data = mnist.drop(['label'], axis=1).values
    return data, label


# 数据划分
def data_partitioning(data, label):
    # 留出法
    return train_test_split(data, label, random_state=42)


def main():
    # 读取数据集
    data, label = get_dataset()

    # 数据划分
    data_train, data_test, label_train, label_test = data_partitioning(
        data, label)

    # 模型操作
    mnist_experiment = MnistExperiment(data_train, label_train)

    # 模型评价&误差分析
    print("Predict MLP")
    mnist_experiment.mlp_predict(data_test, label_test)
    print("Predict KNN")
    mnist_experiment.knn_predict(data_test, label_test)
    print("Predict RidgeClassifier")
    mnist_experiment.ridge_classifier_predict(data_test, label_test)


if __name__ == '__main__':
    main()
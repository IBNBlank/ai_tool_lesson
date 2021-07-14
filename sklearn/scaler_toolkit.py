#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-13
################################################################
"""
数据缩放
"""
from sklearn import preprocessing
import numpy as np


class ScalerToolkit(object):
    def __init__(self, data_train):
        self.__data_train = data_train

    def __show_result(func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"结果:\n{result}")
            print(f"平均值:\n{result.mean(0)}")
            print(f"标准差:\n{result.std(0)}\n")

        return inner

    @__show_result
    def scale(self):
        # 归一化数据
        return preprocessing.scale(self.__data_train)

    @__show_result
    def standard_scale(self):
        # 计算归一化系数
        scaler = preprocessing.StandardScaler().fit(self.__data_train)
        # 用归一化系数归一化数据, 可归一化多组数据
        return scaler.transform(self.__data_train)

    @__show_result
    def min_max_scale(self):
        # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # X_scaled = X_std * (max - min) + min
        scaler = preprocessing.MinMaxScaler()
        return scaler.fit_transform(self.__data_train)

    @__show_result
    def max_abs_scale(self):
        scaler = preprocessing.MaxAbsScaler()
        return scaler.fit_transform(self.__data_train)

    @__show_result
    def robust_scale(self):
        # 对含有离群点的数据效果较好
        scaler = preprocessing.RobustScaler().fit(self.__data_train)
        return scaler.transform(self.__data_train)

    @__show_result
    def nomalize_scale(self):
        # 可用于向量单位化(行向量)
        # 此处fit没有实际意义
        scaler = preprocessing.Normalizer().fit(self.__data_train)
        return scaler.transform(self.__data_train)


def main():
    data_train = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
    print(f"原始数据:\n{data_train}")

    scaler_toolkit = ScalerToolkit(data_train)

    print("scaler_toolkit.scale()")
    scaler_toolkit.scale()
    print("scaler_toolkit.standard_scale()")
    scaler_toolkit.standard_scale()
    print("scaler_toolkit.min_max_scale()")
    scaler_toolkit.min_max_scale()
    print("scaler_toolkit.max_abs_scale()")
    scaler_toolkit.max_abs_scale()
    print("scaler_toolkit.robust_scale()")
    scaler_toolkit.robust_scale()
    print("scaler_toolkit.nomalize_scale()")
    scaler_toolkit.nomalize_scale()


if __name__ == '__main__':
    main()
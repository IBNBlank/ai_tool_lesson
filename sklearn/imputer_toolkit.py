#!/usr/bin/env python3
################################################################
# Copyright 2021 HUST Control Science Innovation Base.
# All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2021-07-13
################################################################
"""
缺失值填充
"""
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import numpy as np


class ImputerToolkit(object):
    def __init__(self, data):
        self.__data = data

    def __show_result(func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"imputed_data:\n{result}")

        return inner

    @__show_result
    def simple_impute(self):
        # strategy = 'mean', 'median', 'most_frequnt', 'constant'
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        return imputer.fit_transform(self.__data)

    @__show_result
    def simple_impute_with_sample(self, sample):
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # 可以使用较完整的数据进行fit
        imputer.fit(sample)
        return imputer.transform(self.__data)

    @__show_result
    def knn_impute(self):
        imputer = KNNImputer(n_neighbors=2, weights='uniform')
        return imputer.fit_transform(self.__data)


def main():
    sample = [[1, 2], [np.nan, 3], [7, 6]]
    data = [[np.nan, 2], [6, np.nan], [7, 6]]

    imputer_toolkit = ImputerToolkit(data)

    imputer_toolkit.simple_impute()
    imputer_toolkit.simple_impute_with_sample(sample)
    imputer_toolkit.knn_impute()


if __name__ == '__main__':
    main()
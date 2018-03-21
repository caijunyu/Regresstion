#!/usr/bin/env python
# coding=utf-8

import math
import numpy as np

__author__ = 'robocai'


def fit(instance_list, alpha, iteration=100):
    model = []
    theta = [ 10.0, 0.1, 0.1 ]
    # 开始保存一个
    model.append(tuple(theta))

    for i in range(iteration):  # 100次迭代
        gradient = [0.0, 0.0, 0.0]
        for instance in instance_list:
            f = 1.0 / (1 + math.exp(-(theta[0] + theta[1] * instance[0] + theta[2] * instance[1]))) # logistic sigmoid
            # 下面这里可以设x0=1, 把高维向量考虑进来一次性用向量操作
            # demo简单起见, 手写了
            gradient[0] += f - instance[2]
            gradient[1] += (f - instance[2]) * instance[0]
            gradient[2] += (f - instance[2]) * instance[1]
        # print gradient
        for j in range(len(theta)):
            theta[j] -= alpha * gradient[j]

        # 中间保存一次
        if i == 5:
            model.append(tuple(theta))

    # 最后保存一次
    model.append(tuple(theta))

    return model


def sigmoid_fit(data, alpha, iteration):
    model = []   #保存训练的参数
    n = len(data[0])
    theta = np.zeros(n)  #设置n个theta,包括theta0
    model.append(tuple(theta))
    m = len(data)
    for itera in range(iteration):
        gradient = np.zeros(n)
        for d in data:
            x = d[:-1]
            y = d[-1]
            f = 1.0 / (1 + math.exp(-(theta[0] + theta[1] * x[0] + theta[2] * x[1])))
            gradient[0] += (f - y)
            for g in range(n-1):
                gradient[g + 1] += (f - y) * x[g]
        for j in range(n):
            theta[j] -= alpha*gradient[j]
            
        # 中间保存一次
        if itera == 5:
            model.append(tuple(theta))

    # 最后保存一次
    model.append(tuple(theta))

    return model
    
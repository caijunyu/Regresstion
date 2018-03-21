#!/usr/bin/env python
# coding=utf-8

import numpy as np
__author__ = 'robocai'


def fit(instance_list, alpha, iteration=100):
    model = []
    theta = [0.0, 0.5]

    # 初始化保存依次
    model.append(tuple(theta))

    m = len(instance_list)
    for i in range(iteration):  # 100次迭代
        gradient = [0.0, 0.0]  # 一个给theta0, 一个给theta1
        for instance in instance_list:
            f = theta[0] + theta[1] * instance[0]
            # 下面这里可以设x0=1, 把高维向量考虑进来一次性用向量操作
            gradient[0] += f - instance[1]
            gradient[1] += (f - instance[1]) * instance[0]
        # print gradient
        for j in range(len(theta)):
            theta[j] -= alpha * (1.0 / m) * gradient[j] #用全部数据更新参数 

        # 中间保存1次
        if i == 5:
            model.append(tuple(theta))

    # 最后保存依次
    model.append(tuple(theta))
    print(model)
    return model


#线性回归 mse解法。 alpha:步长，lamda:正则化参数,iteration:迭代次数
#data:数据集是一个数组，一行是一个数据，有m个特征，n行代表n个数据。最后一个数据表示分类y
#batch梯度下降
def regression_batch_gd(data , alpha, lamda, iteration):  
    model = []
    n = len(data[0])  #提取第一行数据，去掉最后一个标记y，剩下的就是x。
    theta = np.zeros(n)
    model.append(tuple(theta))
    m = len(data)
    for itera in range(iteration):
        gradient = [0.0, 0.0]  # 一个给theta0, 一个给theta1
        for d in data:  #对每一个训练数据
            #gradient = [0.0, 0.0]  # 一个给theta0, 一个给theta1
            x = d[:-1]
            y = d[-1]
            f = theta[0] + np.dot(theta[1:], x)
            gradient[0] += f - y   #theta[0]因为没有x，所以单独处理。
            gradient[1] += (f - y)*x
        for j in range(len(theta)):
            theta[j] = theta[j] -alpha*(1.0 / m) *gradient[j]# + lamda*theta[j]  #因为模型只有两维，所以不用加正则项。
        #print("itera: %d" % itera)
        #print(theta)
        # 中间保存1次
        if itera == 50:
            model.append(tuple(theta))
    model.append(tuple(theta))
    print(model)
    print('=============')
    return model


#线性回归 mse解法。 alpha:步长，lamda:正则化参数,iteration:迭代次数
#data:数据集是一个数组，一行是一个数据，有m个特征，n行代表n个数据。最后一个数据表示分类y
#随机梯度下降
def regression_sgd(data , alpha, lamda, iteration):
    model = []
    n = len(data[0])  #提取第一行数据，去掉最后一个标记y，剩下的就是x。
    theta = np.zeros(n)
    model.append(tuple(theta))
    for itera in range(iteration):
        for d in data:  #对每一个训练数据
            gradient = [0.0, 0.0]  # 一个给theta0, 一个给theta1
            x = d[:-1]
            y = d[-1]
            f = theta[0] + np.dot(theta[1:], x)
            gradient[0] = f - y
            gradient[1] = (f - y)*x
            for j in range(len(theta)):                #sgd，每一个训练数据跟新一次theta
                theta[j] = theta[j] -alpha*gradient[j]# + lamda*theta[j]   #因为模型只有两维，所以不用加正则项。
        #print("itera: %d" % itera)
        #print(theta)
        # 中间保存1次
        if itera == 50:
            model.append(tuple(theta))
    model.append(tuple(theta))
    print(model)
    print('=============')
    return model
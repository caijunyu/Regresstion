#!/usr/bin/env python
# coding=utf-8

import random
import matplotlib.pyplot as plt
import max_likelihood
import numpy as np

__author__ = 'robocai'

instance_list = []

instance_x = []
instance_y_1 = []
instance_y_2 = []
instance_y_3 = []
for x in range(0, 180, 3):
    instance_x.append(x)
    instance_list.append((x, 8 + 0.1 * x, 1)) # 第三列是label
    instance_list.append((x, 5 + 0.1 * x, 1))
    instance_list.append((x, 0.1 * x, 0))
    # 下面的数据仅供画图
    instance_y_3.append(8 + 0.1 * x)
    instance_y_2.append(5 + 0.1 * x)
    instance_y_1.append(0.1 * x)

plt.plot(instance_x, instance_y_1, 'o', color='r')
plt.plot(instance_x, instance_y_2, 'o', color='b')
plt.plot(instance_x, instance_y_3, 'o', color='b')
plt.ylim(ymin=5, ymax=20)
#plt.show()

# 开始模型训练, f=logistic_sigmoid(theta0+theta1*x1+theta2*x2)
random.shuffle(instance_list)
iteration = 1500
#model = max_likelihood.fit(instance_list, 0.0001, iteration)
instance_list = np.array(instance_list)   #将list转成矩阵，便于min_square.regression使用。
model = max_likelihood.sigmoid_fit(instance_list, 0.0001, iteration)

x_list = []
y_1_list = []
y_2_list = []
y_3_list = []
for x in range(0, 180):
    x_list.append(x)
    # 取p=sigmoid(theta*X)=0.5为分类阈值, 则要求theta*X=0
    # 反过来推算, x2=-(theta0+theta1*x1)/theta2, 这样就可以画出一条直线
    # 对应到高维空间就是超平面
    y_1_list.append(-(model[0][0] + model[0][1] * x) / model[0][2])
    y_2_list.append(-(model[1][0] + model[1][1] * x) / model[1][2])
    y_3_list.append(-(model[2][0] + model[2][1] * x) / model[2][2])
plt.plot(x_list, y_1_list, '-.', color='y', linewidth=3.0, label=u'iter-0')
plt.plot(x_list, y_2_list, '-.', color='g', linewidth=3.0, label=u'iter-m')
plt.plot(x_list, y_3_list, '--', color='b', linewidth=3.0, label=u'iter-%d' % iteration)

plt.legend()
plt.show()

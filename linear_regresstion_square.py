#!/usr/bin/env python
# coding=utf-8

import random
import matplotlib.pyplot as plt
import min_square
import numpy as np

__author__ = 'robocai'

instance_list = []

instance_x = []
instance_y_1 = []
instance_y_2 = []
instance_y_3 = []
for x in range(0, 180, 3):
    instance_x.append(x)
    instance_list.append([x, 0.1 * x])
    instance_list.append([x, 5 + 0.1 * x])
    instance_list.append([x, 8 + 0.1 * x])
    instance_y_3.append(8 + 0.1 * x)
    instance_y_2.append(5 + 0.1 * x)
    instance_y_1.append(0.1 * x)
instance_list = np.array(instance_list)   #将list转成矩阵，便于min_square.regression使用。

plt.plot(instance_x, instance_y_1, 'o', color='r')
plt.plot(instance_x, instance_y_2, 'o', color='b')
plt.plot(instance_x, instance_y_3, 'o', color='b')
plt.ylim(ymin=5, ymax=20)
#plt.show()

# 开始模型训练, y=theta0+theta1*x
random.shuffle(instance_list)
iteration = 100
#model = min_square.fit(instance_list, 0.0001, iteration)
model = min_square.regression_sgd(instance_list,0.0001,0.5, iteration*100)
#model = min_square.regression_batch_gd(instance_list,0.0001,0.5, iteration*100)

x_list = []
y_1_list = []
y_2_list = []
y_3_list = []
for x in range(0, 180):
    x_list.append(x)
    y_1_list.append(model[0][0] + model[0][1] * x)
    y_2_list.append(model[1][0] + model[1][1] * x)
    y_3_list.append(model[2][0] + model[2][1] * x)
plt.plot(x_list, y_1_list, '-.', color='y', linewidth=3.0, label=u'iter-0')
plt.plot(x_list, y_2_list, '-.', color='g', linewidth=3.0, label=u'iter-m')
plt.plot(x_list, y_3_list, '--', color='b', linewidth=3.0, label=u'iter-%d' % iteration)

plt.legend()
plt.show()

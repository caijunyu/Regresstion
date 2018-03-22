#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


#使用lasso训练模型，并对alpha取10个候选参数，通过gridSearch来寻找最佳参数，并用五折交叉验证训练模型。
if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('8.Advertising.csv')    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print x
    print y

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # print x_train, y_train
    model = Lasso()  #使用lasso正则化的线性回归。
    # model = Ridge()

    alpha_can = np.logspace(-3, 2, 10)
    print('alpha_can = ' , alpha_can)
    np.set_printoptions(suppress = True)  #按照小数显示
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)  #五折交叉验证 
    lasso_model.fit(x, y)
    print '验证参数：\n', lasso_model.best_params_

    y_hat = lasso_model.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print mse, rmse

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

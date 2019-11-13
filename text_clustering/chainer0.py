# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:07:53 2019
Chainer的初步学习
https://www.cnblogs.com/demo-deng/p/9713471.html
@author: Wei
"""

"""
    测试使用
"""
#import pickle
#import time
import numpy as np
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

# 创建Chainer Variables变]量
a = Variable(np.array([3], dtype=np.float32))
b = Variable(np.array([4], dtype=np.float32))
c = a**2 +b**2

# 5通过data属性检查之前定义的变量
print('a.data:{0}, b.data{1}, c.data{2}'.format(a.data, b.data, c.data))

# 使用backward()方法，对变量c进行反向传播.对c进行求导
c.backward()
# 通过在变量中存储的grad属性，检查其导数
print('dc/da = {0}, dc/db={1}, dc/dc={2}'.format(a.grad, b.grad, c.grad))

# 在chainer中做线性回归
x = 30*np.random.rand(1000).astype(np.float32)
y = 7*x + 10
y += 10*np.random.randn(1000).astype(np.float32)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# 使用chainer做线性回归

# 从一个变量到另一个变量建立一个线性连接
linear_function = L.Linear(1, 1)
# 设置x和y作为chainer变量，以确保能够变形到特定形态
x_var = Variable(x.reshape(1000, -1))
y_var = Variable(y.reshape(1000, -1))
# 建立优化器
optimizer = optimizers.MomentumSGD(lr=0.001)
optimizer.setup(linear_function)


# 定义一个前向传播函数，数据作为输入，线性函数作为输出
def linear_forward(data):
    return linear_function(data)


# 定义一个训练函数，给定输入数据，目标数据，迭代数
def linear_train(train_data, train_traget, n_epochs=200):
    for _ in range(n_epochs):
        # 得到前向传播结果
        output = linear_forward(train_data)
        # 计算训练目标数据和实际标数据的损失
        loss = F.mean_squared_error(train_traget, output)
        # 在更新之前将梯度取零，线性函数和梯度有非常密切的关系
        # linear_function.zerograds()
        linear_function.cleargrads()
        # 计算并更新所有梯度
        loss.backward()
        # 优化器更新
        optimizer.update()


# 绘制训练结果
plt.scatter(x, y, alpha=0.5)
for i in range(150):
    # 训练
    linear_train(x_var, y_var, n_epochs=5)
    # 预测值
    y_pred = linear_forward(x_var).data
    plt.plot(x, y_pred, color=plt.cm.cool(i / 150.), alpha=0.4, lw=3)

slope = linear_function.W.data[0, 0]        # linear_function是之前定义的连接，线性连接有两个参数W和b，此种形式可以获取训练后参数的值，slope是斜率的意思
intercept = linear_function.b.data[0]       # intercept是截距的意思
plt.title("Final Line: {0:.3}x + {1:.3}".format(slope, intercept))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
import os
import numpy as np
# 屏蔽 TensorFlow 底层警告日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

# 数据预处理

# 加载训练集和测试集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 重塑训练集和测试集的形状
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
print(X_train.shape)
print(X_test.shape)
print(X_train.dtype)
print(X_test.dtype)

# 归一化
X_train /= 255
X_test /= 255
# 查看X_train的第2个例子的第100个到150个像素点的值
print(X_train[1, 100:151]) 

# one-hot编码
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)
print(Y_train[:5])

# 简单感知机
model = Sequential()
# 添加全连接层，输出10，输入784
model.add(Dense(10, input_shape=(784,)))
# 添加激活层
model.add(Activation('softmax'))
# 查看模型摘要
model.summary()

# 编译神经网络
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 训练神经网络
model.fit(X_train, Y_train, batch_size=128, epochs=200, verbose=1, validation_split=0.2)

# 评估神经网络
score = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test score {score[0]}")
print(f"Test accuracy {score[1]}")

# 更好的模型：增加隐藏层
# 模型改进
model = Sequential()
# 隐藏层：输出128，激活relu
model.add(Dense(128, input_shape=(784,), activation='relu'))
# 再加一个隐藏层
model.add(Dense(128, activation='relu'))
# 输出层：输出10，激活softmax
model.add(Dense(10, activation='softmax'))
model.summary()

# 编译神经网络
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 训练神经网络
model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)

# 评估神经网络
score = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")
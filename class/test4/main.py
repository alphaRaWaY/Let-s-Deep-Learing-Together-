import time
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical

# 数据处理
with np.load('data/mnist.npz') as f:
    xtr, ytr = f['x_train'], f['y_train']
    xte, yte = f['x_test'], f['y_test']

xtr = xtr.reshape(-1, 784).astype('float32') / 255.0
xte = xte.reshape(-1, 784).astype('float32') / 255.0
ytr, yte = to_categorical(ytr, 10), to_categorical(yte, 10)

# 简单感知机
m1 = Sequential([Input(shape=(784,)), Dense(10, activation='softmax')])
m1.compile(loss='crossentropy', optimizer='sgd', metrics=['accuracy'])

t0 = time.time()
m1.fit(xtr, ytr, batch_size=128, epochs=200, verbose=0)
t1 = time.time()
acc1 = m1.evaluate(xte, yte, verbose=0)[1]

# 深度多层感知器 
m2 = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
m2.compile(loss='crossentropy', optimizer='sgd', metrics=['accuracy'])

t2 = time.time()
m2.fit(xtr, ytr, batch_size=128, epochs=20, verbose=0)
t3 = time.time()
acc2 = m2.evaluate(xte, yte, verbose=0)[1]

# 结果比较
print(f"简单感知机          Acc: {acc1:.4f} | Time: {t1-t0:.2f}s (200 Epochs)")
print(f"深度多层感知器       Acc: {acc2:.4f} | Time: {t3-t2:.2f}s (20 Epochs)")
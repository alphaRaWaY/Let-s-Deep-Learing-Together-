# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

# 1.1.2 导入相关包
from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, Input, Reshape, MaxPooling2D, Conv2D, Dense, Flatten
from keras.optimizers import Adam
from keras import backend as K

# 1.2 载入数据
def load_local_mnist(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_local_mnist('data/mnist.npz')

class DataWrapper:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.img_size = 28
        self.img_size_flat = 784
        self.img_shape = (28, 28)
        self.img_shape_full = (28, 28, 1)
        self.num_classes = 10
        self.num_channels = 1
        # 预处理：归一化
        self.x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
        self.x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
        # 标签转 One-Hot
        self.y_train = tf.keras.utils.to_categorical(y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(y_test, 10)
        self.y_test_cls = y_test

data = DataWrapper(x_train_raw, y_train_raw, x_test_raw, y_test_raw)

# 1.3 配置神经网络
img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape
img_shape_full = data.img_shape_full
num_classes = data.num_classes
num_channels = data.num_channels

print(f"1.3 配置神经网络 图像尺寸 {img_size} 扁平化长度 {img_size_flat} 形状 {img_shape_full}")

# 1.4 绘制图像的辅助函数
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        xlabel = f"True: {cls_true[i]}"
        if cls_pred is not None: xlabel += f", Pred: {cls_pred[i]}"
        ax.set_xlabel(xlabel)
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()

# 1.5 绘制错误分类图像的辅助函数
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = data.x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.y_test_cls[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

# 2. 序列模型
# 2.3 训练
print("\n2.3 序列模型训练")
# 2.1 模型框架
model = Sequential([
    InputLayer(shape=(img_size_flat,)),
    Reshape(img_shape_full),
    Conv2D(kernel_size=5, filters=16, padding='same', activation='relu', name='layer_conv1'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(kernel_size=5, filters=36, padding='same', activation='relu', name='layer_conv2'),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# 2.2 模型编译
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# 开始训练
model.fit(x=data.x_train, y=data.y_train, epochs=1, batch_size=128)

# 2.4 评估输出
print("\n2.4 序列模型评估结果")
result = model.evaluate(x=data.x_test, y=data.y_test, verbose=0)
for name, val in zip(model.metrics_names, result):
    print(name, val)

# 2.5 预测前9张
print("\n2.5 绘制序列模型前九张预测结果")
images_9 = data.x_test[0:9]
cls_true_9 = data.y_test_cls[0:9]
y_pred_9 = model.predict(x=images_9, verbose=0)
cls_pred_9 = np.argmax(y_pred_9, axis=1)
plot_images(images=images_9, cls_true=cls_true_9, cls_pred=cls_pred_9)

# 2.6 错分类的图片
print("\n2.6 绘制序列模型错分类样本图片")
y_pred_all = model.predict(x=data.x_test, verbose=0)
cls_pred_all = np.argmax(y_pred_all, axis=1)
correct_all = (cls_pred_all == data.y_test_cls)
plot_example_errors(cls_pred=cls_pred_all, correct=correct_all)
print(f"2.6 测试集总数 {len(data.x_test)} 错误分类数 {np.sum(correct_all == False)}")

# 3. 功能模型
# 3.3 训练
print("\n3.3 功能模型训练")
# 3.1 模型框架
inputs = Input(shape=(img_size_flat,))
net = Reshape(img_shape_full)(inputs)
net = Conv2D(kernel_size=5, filters=16, padding='same', activation='relu', name='layer_conv1_func')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Conv2D(kernel_size=5, filters=36, padding='same', activation='relu', name='layer_conv2_func')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Flatten()(net)
net = Dense(128, activation='relu')(net)
outputs = Dense(num_classes, activation='softmax')(net)

model2 = Model(inputs=inputs, outputs=outputs)

# 3.2 模型编译
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 开始训练
model2.fit(x=data.x_train, y=data.y_train, epochs=1, batch_size=128)

# 3.4 评估模型
print("\n3.4 功能模型评估结果")
result2 = model2.evaluate(x=data.x_test, y=data.y_test, verbose=0)
for name, val in zip(model2.metrics_names, result2):
    print(name, val)

# 3.5 错误分类图片
print("\n3.5 绘制功能模型错分类图片")
y_pred_m2 = model2.predict(data.x_test, verbose=0)
cls_pred_m2 = np.argmax(y_pred_m2, axis=1)
correct_m2 = (cls_pred_m2 == data.y_test_cls)
plot_example_errors(cls_pred_m2, correct=correct_m2)

# 4. 保存与加载
# 4.1 保存Keras模型
path_model = 'model.keras'
model2.save(path_model)
# 4.2 删除模型
del model2
# 4.3 加载模型
model3 = load_model(path_model)
layer_conv1 = model3.get_layer('layer_conv1_func')
layer_conv2 = model3.get_layer('layer_conv2_func')

# 4.4 用加载的模型预测
print("\n4.4 使用加载的模型预测前九张图片")
images_load = data.x_test[0:9]
cls_true_load = data.y_test_cls[0:9]
y_pred_load = model3.predict(x=images_load, verbose=0)
cls_pred_load = np.argmax(y_pred_load, axis=1)
plot_images(images=images_load, cls_true=cls_true_load, cls_pred=cls_pred_load)
print("4.4 预测类数为", cls_pred_load)
print("4.4 真实类数为", cls_true_load)

# 5. 权重和输出可视化
# 5.1 & 5.4 辅助函数
def plot_conv_weights(weights, input_channel=0):
    w_min, w_max = np.min(weights), np.max(weights)
    num_filters = weights.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = weights[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()

def plot_conv_output(values):
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(values[0, :, :, i], interpolation='nearest', cmap='binary')
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()

# 5.2 模型摘要
print("\n5.2 模型网络结构摘要")
model3.summary()

# 5.3 绘制卷积权重
print("\n5.3 可视化卷积层权重")
weights_c1 = layer_conv1.get_weights()[0]
plot_conv_weights(weights=weights_c1, input_channel=0)
weights_c2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_c2, input_channel=0)

# 5.5 输入图像
def plot_image(image):
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.show()

image1 = data.x_test[0]
print("\n5.5 展示原始输入图像")
plot_image(image1)

# 5.6 卷积层输出之方法一
print("\n5.6 使用方法一获取卷积层输出")

def K_function_mock(inputs, outputs):
    model_temp = Model(inputs=inputs, outputs=outputs)
    def func(input_list):
        data_in = np.array(input_list[0])
        if len(data_in.shape) == 1:
            data_in = data_in[np.newaxis, :]
        return [model_temp(data_in).numpy()]
    return func

output_conv1 = K_function_mock(inputs=model3.input, outputs=layer_conv1.output)

# 5.6.2 获取并查看形状
layer_output1 = output_conv1([[image1]])[0]
print("5.6.2 卷积层1输出形状")
print(layer_output1.shape) 

# 5.6.3 绘制输出
plot_conv_output(values=layer_output1)

# 5.7 卷积层输出之方法二
print("\n5.7 使用方法二获取卷积层输出")
output_conv2 = Model(inputs=model3.input, outputs=layer_conv2.output)

# 5.7.2 获取并查看形状
layer_output2 = output_conv2.predict(np.array([image1]), verbose=0)
print("5.7.2 卷积层2输出形状")
print(layer_output2.shape) 

# 5.7.3 绘制输出
plot_conv_output(values=layer_output2)
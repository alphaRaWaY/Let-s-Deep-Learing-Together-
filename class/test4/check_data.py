import numpy as np

path = 'data/mnist.npz'

npz_file = np.load(path)

x_test = npz_file['x_test']
x_train = npz_file['x_train']
y_train = npz_file['y_train']
y_test = npz_file['y_test']

print(f"训练集图像数量: {x_train.shape[0]}, 标签数量: {y_train.shape[0]}")
print(f"测试集图像数量: {x_test.shape[0]}, 标签数量: {y_test.shape[0]}")

print(f"第一张训练图像的标签是: {y_train[0]}")

if 'npz_file' in locals() and hasattr(npz_file, 'close'):
    npz_file.close()
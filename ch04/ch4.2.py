# 从零实现多层感知机
import torch
from torch import nn
from d2l import torch as d2l

"""继续使用Fashion-MNIST图像分类数据集"""
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

"""定义参数"""
num_inputs,num_outputs,num_hiddens=784,10,256

"""输入层"""
W1 = nn.Parameter(torch.randn(
num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
"""中间隐藏层"""
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
"""合并参数"""
params = [W1, b1, W2, b2]

def relu(X):
    """激活函数"""
    a=torch.zeros_like(X)
    return torch.max(X,a)

def net(X):
    """定义模型"""
    X=X.reshape((-1,num_inputs))
    # 先进行隐藏层计算
    H=relu(X@W1+b1)# 这里“@”代表矩阵乘法
    # 返回输出层的计算结果
    return (H@W2+b2)

"""为了与softmax比较，使用相同的损失函数：交叉熵损失函数"""
loss=nn.CrossEntropyLoss(reduction='none')

def main():
    """训练"""
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    # 在测试数据集上应用模型
    d2l.predict_ch3(net, test_iter)

if __name__=='__main__':
    main()
    d2l.plt.show()

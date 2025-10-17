import torch
from torch import nn
from d2l import torch as d2l

"""
softmax回归的输出层是全连接层
在网络中添加一个有10个输出的全连接层
"""
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):
    """以0为均值，以0.01为标准差随机初始化权重"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)


def main():

    # 使用数据
    batch_size=256
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
    net.apply(init_weights)

    """
    优化之后的softmax交叉熵损失函数计算
    """
    loss = nn.CrossEntropyLoss(reduction='none')

    """
    优化算法
    使用学习率为0.1的小批度随机梯度下降作为优化算法
    """
    trainer=torch.optim.SGD(net.parameters(),lr=0.1)

    """训练"""
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    

if __name__ == '__main__':
    main()
    d2l.plt.show()
import torch
from d2l import torch as d2l

"""导入全局工具依赖"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
import my_utils 

"""使用Fashion_MNIST数据集"""
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
# 输入的特征为图像的各个像素,分辨率为28*28
num_input=28*28
# 输出特征为独热编码
num_output=10
# 定义权重向量
W=torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)
# 定义偏移量向量
b=torch.zeros(num_output,requires_grad=True)

def softmax(X):
    """
    softmax的三个步骤：
    1. 对每一项求幂（exp）
    2. 对每一行求和
    3. 每一行初一规范化常数，确保和为1
    """
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition #利用广播机制

def net(X):
    """定义softmax回归模型"""
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)

def cross_entropy(y_hat,y):
    return - torch.log(y_hat[range(len(y_hat)),y])

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def main():

    """定义softmax操作"""
    X = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    """对于一个矩阵，可以求同一个轴上的元素和"""
    print(X.sum(0,keepdim=True)) #横向求和
    print(X.sum(1,keepdim=True)) #纵向求和

    X=torch.normal(0,1,(2,5))
    X_prob=softmax(X)
    print(X_prob)
    print(X_prob.sum(1))

    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    """花式索引，获取y_hat[0][0]和y_hat[1][2]"""
    print(y_hat[[0, 1], y])
    print(cross_entropy(y_hat,y))

    print(my_utils.accuracy(y_hat,y)/len(y))

    print(my_utils.evaluate_accuracy(net,test_iter))

    num_epochs = 10
    my_utils.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    
    my_utils.predict_ch3(net, test_iter)

if __name__ == '__main__':
    main()
    d2l.plt.show()
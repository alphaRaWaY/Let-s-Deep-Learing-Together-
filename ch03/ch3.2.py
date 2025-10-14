# %matplotlib inline
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size,features,labels):
    """
    批次大小、特征、标签
    获取数据迭代器
    yield 允许你编写一个函数
    它能暂停执行并返回一个值
    然后在下次被调用时从上次暂停的地方继续执行。    
    """
    num_examples = len(features)
    indices=list(range(num_examples)) # 幂
    # 样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        yield features[batch_indices],labels[batch_indices]


def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b;    

def squared_loss(y_hat,y):
    """均方损失"""
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print('features:', features[0],'\nlabel:', labels[0])

    
    # 展示生成的随机数据
    d2l.set_figsize()
    d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
    d2l.plt.show()  
   

    batch_size=10
    """
    展示小批次样本的数据
    for X,y in data_iter(batch_size,features,labels):
        print(X,'\n',y)
        break
    """


    """使用正态分布初始化权重"""
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    """
    在初始化参数之后
    我们的任务是更新这些参数
    直到这些参数足够拟合我们的数据。
    """
    
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')


if __name__ == '__main__':
    main()
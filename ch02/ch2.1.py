import torch

# 入门
def tensor_learning():
    # 张量表示一个由数组组成的数组
    # 一个轴的张量对应向量
    # 两个轴的张量对应矩阵
    # =======================================
    # 使用arange创建行向量x
    x=torch.arange(12)
    print(x)
    # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    # =======================================
    # 通过张量的shape属性访问张量
    # 通过张量的shape属性访问张量Z
    print(x.shape)
    # torch.Size([12])
    # =======================================
    # 查看张量中的元素总数
    print(x.numel())
    # 12
    # =======================================
    # 改变张量的形状而不改变元素数量和元素值
    X=x.reshape(3,4)
    print(X)
    # tensor([[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]])
    # 可以调用-1来自动计算形状
    # x.reshape(-1,4)或者x.reshape(3,-1)
    # =======================================
    # 清零或者设置全部为1
    all_zero=torch.zeros((2,3,4))
    all_one=torch.zeros((2,3,4))
    print(all_zero)
    # tensor([[[0., 0., 0., 0.],
    #          [0., 0., 0., 0.],
    #          [0., 0., 0., 0.]],

    #         [[0., 0., 0., 0.],
    #          [0., 0., 0., 0.],
    #          [0., 0., 0., 0.]]])
    # =======================================
    # 从标准正态分布中随机采样构建张量
    __rand=torch.randn(3,4)
    print(__rand)
    # tensor([[ 2.7870,  0.0864,  0.4522,  1.1740],
    #         [-0.8204,  0.1954,  0.8858, -1.2325],
    #         [ 0.1918,  1.3803,  1.1960,  0.0302]])
    # =======================================

# 运算符
def culculator():
    x = torch.tensor([1.0,2,4,8])
    y = torch.tensor([2,2,2,2])
    print(x+y)
    print(x-y)
    print(x*y)
    print(x/y)
    print(x**y)
    # 求自然实数幂
    print(torch.exp(x))
    # 张量连接
    X=torch.arange(12,dtype=torch.float32).reshape((3,4))
    Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
    print(torch.cat((X,Y),dim=0))
    print(torch.cat((X,Y),dim=1))
    print(X==Y)
    print(X>Y)
    print(X<Y)
    print(X.sum())

# 广播
def broadcasting():
    # 绝大多数情况沿着长度为1的轴进行广播
    a=torch.arange(3).reshape((3,1))
    b=torch.arange(2).reshape((1,2))
    print(a)
    print(b)
    print(a+b)

# 索引和切片
def index_sample():
    X=torch.arange(12).reshape((3,4))
    print(X)
    # -1提取最后一个元素
    print(X[-1])
    # 使用冒号提取多个元素
    # 表示从1开始，到3结束（不包括3）
    print(X[1:3])
    # 通过索引写入矩阵
    tmp=X
    tmp[1,2]=9
    # 也可以使用低级语言规范：X[1][2]=9
    print(tmp)

    tmp=X
    tmp[0:2,:]=12
    print(X)


def better_allocate():
    # 在python中可以使用id()查看内存地址
    X=torch.arange(12,dtype=torch.float32).reshape((3,4))
    Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
    before = id(Y)
    Y=Y+X
    print(id(Y)==before)

    # 使用zeros_like()分配全为0的块
    Z=torch.zeros_like(Y)
    print("id(Z):",id(Z))
    Z[:]=X+Y
    print("id(Z):",id(Z))

    # 也可以使用X+=来见少内存开销
    before=id(X)
    X+=Y
    print(id(X)==before)

def transform_to_other_object():
    X=torch.arange(12,dtype=torch.float32).reshape((3,4))
    Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
    A=X.numpy()
    B=torch.tensor(A)
    print(type(A))
    print(type(B))
    a=torch.tensor([3.5])
    print(a)
    print(a.item())
    print(float(a))
    print(int(a))

# tensor_learning()
# culculator()
# broadcasting()
# index_sample()
# better_allocate()
# transform_to_other_object()
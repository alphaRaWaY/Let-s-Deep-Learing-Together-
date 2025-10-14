import torch

# 向量
def vector():
    x=torch.arange(4)
    print(x)
    # 访问向量元素
    print(x[3])
    # 访问向量长度
    print(len(x))
    # 查看向量每个轴的长度，向量一般只有一个轴
    print(x.shape)

# 矩阵
def matrix():
    A=torch.arange(20).reshape(5,4)
    print(A)
    print(A.T)

# 张量
def __tensor():
    X = torch.arange(24).reshape(2, 3, 4)
    print(X)

def tensor_culculator():
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
    A, A + B
    print(A)
    print(A+B)
    print(A*B) #Hadamard积
    print(2*A)
    # 降维
    x=torch.arange(4,dtype=torch.float32)
    print(x)
    print(x.sum())
    # 降维的维度
    A_sum_axis0=A.sum(axis=0)
    print(A_sum_axis0)
    print(A_sum_axis0.shape)
    # 指定两个维度
    print(A.sum(axis=[0,1])==A.sum())
    # 求平均值
    print(A.mean())
    print(A.sum()/A.numel())
    # 指定维度
    print(A.mean(axis=0))
    print(A.sum(axis=0)/A.shape[0])
    # 非降维求和
    sum_A=A.sum(axis=1,keepdim=True)
    print(sum_A)
    # 尤其可以用于求占比
    print(A/sum_A)
    # 按维度求前缀和的
    print(A.cumsum(axis=0))

    # 向量点积
    y=torch.ones(4,dtype=torch.float32)
    print(x)
    print(y)
    print(torch.dot(x,y))
    print(torch.sum(x*y)) #数学意义

    # 矩阵-向量积
    print(A)
    print(x)
    print(torch.mv(A,x))

    # 矩阵-矩阵乘法
    B=torch.ones(4,3)
    print(torch.mm(A,B))

    #范数
    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u)) #L2范数sqrt(sum_u)
    print(torch.abs(u).sum()) #L1范数

def main():
    # vector()
    # matrix()
    tensor_culculator()

main()
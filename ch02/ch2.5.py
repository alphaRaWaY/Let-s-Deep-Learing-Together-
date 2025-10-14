# 自动微分
# 自动反向传播
import torch
x=torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)

y=torch.dot(x,x)*2
print(y)

y.backward()
print(x.grad)

print(x.grad==x*4)

# 默认情况下，pytorch会积累梯度，每一次使用需要清楚之前的值
x.grad.zero_()
y=x.sum()
y.backward()
print(x.grad)

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)



x.grad.zero_()
y = x * x
u = y.detach() #断开计算图，仅保留数值不保留梯度关系
z = u * x

z.sum().backward()
print(x.grad==u)

# 其中y关于x的计算图依旧保留
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


"""
即使构建函数的计算图需要通过Python控制流
（例如，条件、循环或任意函数调用）
我们仍然可以计算得到的变量的梯度
"""

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d / a)
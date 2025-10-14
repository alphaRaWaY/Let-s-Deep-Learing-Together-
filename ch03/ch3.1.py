"""导入全局工具依赖"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
import my_utils 


# 矢量加速
# %matplotlib inline
import torch
import numpy as np
import math
"""
使用两种方法比较速度
使用python自带的for循环遍历向量
使用"+"的依赖
"""
n=10000
a=torch.ones([n])
b=torch.ones([n])

def use_for():
    c=torch.zeros(n)
    timer=my_utils.Timer()
    for i in range(n):
        c[i]=a[i]+b[i]
    print(f'使用for循环的耗时为:\t{timer.stop():.5f} 秒')

def use_plus():
    timer=my_utils.Timer()
    d=a+b
    print(f'使用+重载的耗时为:\t{timer.stop():.5f} 秒')

"""正态函数"""
def normal(x,mu,sigma):
    p=1/math.sqrt(2*math.pi*sigma**2)
    return p*np.exp(-0.5/sigma**2*(x-mu)**2)

def main():
    use_for()
    use_plus()

    # 再次使用numpy进行可视化
    x = np.arange(-7, 7, 0.01)

    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    my_utils.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
            ylabel='p(x)', figsize=(4.5, 2.5),
            legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
        
        #绘制正态分布的函数

if __name__ == '__main__':
    main()
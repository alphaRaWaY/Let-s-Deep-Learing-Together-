import os
import pandas as pd
import torch
def create_data_CSV(filename='house_tiny.csv'):
    # 在上一个目录创建data文件夹
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', filename)
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    print("文件成功保存到：",data_file)

def load_data_from_CSV(filename='house_tiny.csv'):
    data_file = os.path.join('..', 'data', filename)
    data = pd.read_csv(data_file)
    return data    

# 处理缺失值
# 典型方法有：_插值法_和_删除法_
# 这里考虑使用插值法
# 对于input缺失的数值，使用同一列的均值替换
def deal_NaN(data):
    # 数据的前两列，使用iloc方法
    inputs=data.iloc[:,0:2]
    inputs=inputs.fillna(inputs.mean()) #使用均值填充NaN
    print(inputs)
    inputs=pd.get_dummies(inputs,dummy_na=True) #离散化缺失的值
    print(inputs)


def transform_data_to_tensor(data):
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs=inputs.fillna(inputs.mean())
    inputs=pd.get_dummies(inputs,dummy_na=True)
    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    print(X)
    print(y)


def main():    
    # create_data_CSV()
    data=load_data_from_CSV()
    # deal_NaN(data)
    transform_data_to_tensor(data)

main()
import numpy as np
import pandas as pd

def _sigmoid(z):
    # 实验要求：使用 np.clip 避免溢出 
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)

def get_data(train_path, test_path):
    # 实验要求：skipinitialspace 处理空格 
    df_train = pd.read_csv(train_path, skipinitialspace=True)
    df_test = pd.read_csv(test_path, skipinitialspace=True)
    
    y_train = (df_train['income'].str.strip() == '>50K').astype(float).values
    x_train_raw = df_train.drop('income', axis=1)
    
    # 特征工程：分离连续型和离散型 
    num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    # 实验要求：指定列正态分布标准化 
    # 对训练集计算均值和标准差，并应用于测试集
    x_all = pd.concat([x_train_raw, df_test])
    x_all_encoded = pd.get_dummies(x_all) # One-hot 编码
    
    x_final = x_all_encoded.values.astype(float)
    
    # 标准化连续特征对应的索引
    num_indices = [x_all_encoded.columns.get_loc(c) for c in num_cols]
    mu = np.mean(x_final[:, num_indices], axis=0)
    std = np.std(x_final[:, num_indices], axis=0)
    x_final[:, num_indices] = (x_final[:, num_indices] - mu) / (std + 1e-8)
    
    return x_final[:len(df_train)], y_train, x_final[len(df_train):]

def train(X, Y, epochs=1000, lr=0.1, lam=0.001):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    
    for i in range(epochs):
        # 前向传播 
        z = X @ w + b
        pred = _sigmoid(z)
        err = Y - pred # y - y_hat
        
        # 计算梯度
        # w_grad = -mean(err * X) + lam * w
        w_grad = -(err @ X) / n + lam * w
        b_grad = -np.mean(err)
        
        w -= lr * w_grad
        b -= lr * b_grad
        
        if i % 100 == 0:
            loss = -np.mean(Y * np.log(pred) + (1 - Y) * np.log(1 - pred)) + 0.5 * lam * np.sum(w**2)
            print(f"Epoch {i}: Loss {loss:.4f}")
            
    return w, b

if __name__ == "__main__":
    # 加载与处理
    xtr, ytr, xte = get_data('data/train.csv', 'data/test.csv')
    
    # 训练逻辑回归
    w, b = train(xtr, ytr, epochs=2000, lr=0.2)
    
    prob = _sigmoid(xte @ w + b)
    ans = (prob > 0.5).astype(int)

    pd.DataFrame({
        'id': np.arange(1, len(ans) + 1),
        'label': ans
    }).to_csv('predict.csv', index=False)
    print("预测保存在predict.csv中")
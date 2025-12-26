import numpy as np
import pandas as pd
from numpy.linalg import inv

def solve():
    # 1. Load & Preprocess
    df_tr = pd.read_csv('data/train.csv', skipinitialspace=True)
    df_te = pd.read_csv('data/test.csv', skipinitialspace=True)
    
    y_tr = (df_tr['income'].str.strip() == '>50K').astype(int).values
    x_tr_raw = df_tr.drop('income', axis=1)
    
    num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    cat_cols = [c for c in x_tr_raw.columns if c not in num_cols]

    # 特征工程 (保持 One-Hot 和标准化)
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    
    tf = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    x_tr = tf.fit_transform(x_tr_raw)
    x_te = tf.transform(df_te)

    # 2. GDA Training
    n, d = x_tr.shape
    x0, x1 = x_tr[y_tr == 0], x_tr[y_tr == 1]
    phi = len(x1) / n
    mu0, mu1 = x0.mean(0), x1.mean(0)
    
    # 矩阵化计算共享协方差: (X-mu).T @ (X-mu)
    sigma = ((x0 - mu0).T @ (x0 - mu0) + (x1 - mu1).T @ (x1 - mu1)) / n
    
    # 3. Prediction
    sigma += 1e-6 * np.eye(d) # Regularization
    prec = inv(sigma)
    
    d0, d1 = x_te - mu0, x_te - mu1
    # 利用矩阵内积性质计算 Mahalanobis 距离: diag(D @ prec @ D.T)
    # 优化为 row-wise sum 以节省内存
    lp0 = -0.5 * np.sum((d0 @ prec) * d0, axis=1) + np.log(1 - phi)
    lp1 = -0.5 * np.sum((d1 @ prec) * d1, axis=1) + np.log(phi)
    
    ans = (lp1 > lp0).astype(int)

    # 4. Output
    pd.DataFrame({
        'id': np.arange(1, len(ans) + 1),
        'label': ans
    }).to_csv('predict.csv', index=False)
    print(f"Dim: {d} | Samples: {n}")

if __name__ == '__main__':
    solve()
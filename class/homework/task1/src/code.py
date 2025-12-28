"""
PyTorch 实现二手车价格回归预测 (仿线性回归简洁风格)

任务：回归预测二手车价格 (price)
模型：多层感知机 (MLP)
"""
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

# --- 0. 配置和数据路径 ---
DATA_PATH = os.path.join('..', 'data')
TRAIN_FILE = 'used_car_train.csv'
TEST_FILE = 'used_car_test.csv'
SUBMISSION_FILE = 'used_car_prediction_pytorch_sequential_submission.csv'

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# --- 1. 数据加载和预处理 ---

def preprocess_data():
    """进行特征工程和数据标准化，返回 PyTorch Tensor"""
    print("--- 1. 数据加载与预处理 ---")

    train_df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILE), sep=',')
    test_df = pd.read_csv(os.path.join(DATA_PATH, TEST_FILE), sep=',')
    
    train_len = len(train_df)
    test_sale_id = test_df['SaleID']
    
    # 目标变量：对 price 进行 log(1+x) 变换
    # 修正：在 log1p 之前确保价格非负
    train_df['price'] = train_df['price'].apply(lambda x: x if x >= 0 else 0)
    y_train = np.log1p(train_df['price'])
    
    train_df = train_df.drop('price', axis=1)
    data = pd.concat([train_df, test_df], ignore_index=True)
    
    # ************************ 关键修正 ************************
    # 1.3 优先删除 SaleID 和 name，确保它们不参与后续特征工程
    # 只保留 regDate 和 creatDate 用于计算 used_days
    data = data.drop(columns=['SaleID', 'name']) 
    # ********************************************************
    
    # 1.1 处理 'notRepairedDamage' 字符串
    data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan).astype(float).fillna(0.0)
    
    # 1.2 日期特征工程
    data['regDate'] = pd.to_datetime(data['regDate'].astype(str), format='%Y%m%d', errors='coerce')
    data['creatDate'] = pd.to_datetime(data['creatDate'].astype(str), format='%Y%m%d', errors='coerce')
    data['used_days'] = (data['creatDate'] - data['regDate']).dt.days
    
    # 过滤掉不合理的负天数，将其设为 NaN
    data['used_days'] = data['used_days'].apply(lambda x: x if x > 0 else np.nan)
    
    # ************************ 关键修正 ************************
    # 1. 使用 train_df 的部分来计算 median_days，避免测试集数据泄露
    # 2. 检查 train_set 的 used_days 是否全为 NaN

    used_days_train = data['used_days'].iloc[:train_len]

    # 计算中位数（安全地）
    median_days = np.nanmedian(used_days_train)

    # 如果 median_days 本身是 NaN (表示训练集所有值都是 NaN)，则设置为一个合理的默认值
    # 警告 'All-NaN slice encountered' 将会在这里产生，但会被 if 块捕获
    if np.isnan(median_days):
        print("警告：Used_days 训练集全为 NaN，将使用默认值 0 填充。")
        median_days = 0 
        
    # 使用计算出的中位数填充整个数据集的缺失值
    data['used_days'] = data['used_days'].fillna(median_days)
    # ********************************************************

    # 1.3 (后置) 删除原始日期列
    data = data.drop(columns=['regDate', 'creatDate']) # SaleID 和 name 已经在前面删除

    # 1.4 缺失值和类别特征处理
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # 对所有非数字列进行 Label Encoding (假设为低基数类别)
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    # ************************ 关键修正 ************************
    # 新增步骤 1.4.1：删除方差为零的常数特征
    print("正在检查并删除常数特征...")
    
    # 计算训练集部分的方差
    data_train = data.iloc[:train_len]
    variance = data_train.var()
    
    # 确定方差接近零的列（例如，小于 1e-6）
    const_cols = variance[variance < 1e-6].index.tolist()
    
    if len(const_cols) > 0:
        print(f"检测到以下常数特征 (方差接近零)，将删除: {const_cols}")
        data = data.drop(columns=const_cols)
    else:
        print("未检测到常数特征。")
    # ********************************************************

    # 1.5 数据标准化 (深度学习模型的关键)
    X = data.values.astype(np.float32)

    # ************************ 最终关键修正 ************************
    # 强制进行二次 NaN 检查和填充，以消除 Standard Scaler 的警告
    # 使用 numpy 的 nanmedian 代替 pandas 的 median，并进行填充
    if np.any(np.isnan(X)):
        print("检测到 NaN 值，进行强制中位数填充以消除 Standard Scaler 警告。")
        # 计算所有列的中位数（忽略 NaN）
        nan_mask = np.isnan(X)
        for i in range(X.shape[1]):
            # 找到当前列的非 NaN 值的数组
            col_data = X[~nan_mask[:, i], i]
            # 如果列有数据，计算中位数，否则使用 0 填充
            median_val = np.nanmedian(col_data) if len(col_data) > 0 else 0
            # 填充该列的 NaN 值
            X[nan_mask[:, i], i] = median_val
    else:
        print("未检测到 NaN 值。")
    # ************************************************************

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割回训练集和测试集
    X_train_scaled = X_scaled[:train_len]
    X_test_scaled = X_scaled[train_len:]
    
    # 转换为 PyTorch Tensor
    features = torch.tensor(X_train_scaled, dtype=torch.float32)
    labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) # 保证形状为 (N, 1)
    test_features = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    print(f"训练集特征形状: {features.shape}, 目标形状: {labels.shape}")
    return features, labels, test_features, test_sale_id


# --- 2. 仿照您的风格定义数据加载器 ---

def load_array(data_arrays, batch_size, is_train=True): # @save
    """构造一个PyTorch数据迭代器"""
    # 修正：当只有特征时，data_arrays只有一个元素
    if isinstance(data_arrays, torch.Tensor):
        data_arrays = (data_arrays,)
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# --- 3. 主函数和训练流程 ---

def main():
    
    # 3.1 数据准备
    features, labels, test_features, test_sale_id = preprocess_data()
    
    # 划分训练集和验证集 (用于监控模型在未知数据上的表现)
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.1, random_state=SEED
    )

    batch_size = 256 # 适当增大批次大小以加快训练
    train_iter = load_array((X_train, y_train), batch_size)
    val_iter = load_array((X_val, y_val), batch_size, is_train=False)
    test_iter = load_array(test_features, batch_size, is_train=False)
    
    # 3.2 定义模型 (使用 nn.Sequential 定义一个简单的 MLP)
    input_dim = features.shape[1]
    
    # 仿照您的风格，定义一个多层感知机 (MLP)
    net = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1) # 最终输出维度为 1
    )
    
    # 3.3 初始化操作 (仿照您的风格)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01) # 使用标准正态分布初始化权重
            nn.init.constant_(m.bias, 0) # 偏置初始化为 0

    net.apply(init_weights)

    # 3.4 定义损失函数
    loss = nn.MSELoss() # 均方误差 (Mean Squared Error)

    # 3.5 定义优化算法
    lr = 0.01
    trainer = torch.optim.Adam(net.parameters(), lr=lr) # 使用 Adam 优化器，更适合深度学习

    # 3.6 开始训练
    print("\n--- 3. 开始训练 ---")
    num_epochs = 30
    best_val_loss = float('inf')

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    for epoch in range(1, num_epochs + 1):
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        
        # 计算验证集损失 (用于监控和早停)
        net.eval()
        val_loss_sum = 0
        val_samples = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_iter:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                l_val = loss(net(X_val_batch), y_val_batch)
                val_loss_sum += l_val.item() * X_val_batch.size(0)
                val_samples += X_val_batch.size(0)
        
        avg_val_loss = val_loss_sum / val_samples
        
        # 简化版早停：如果您想保存最佳模型，可以在这里实现
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save(net.state_dict(), 'best_model.pth') # 保存最佳模型
        
        print(f'epoch {epoch}, validation loss {avg_val_loss:.6f}')
        
    # 3.7 预测并保存结果
    print("\n--- 4. 进行测试集预测 ---")
    net.eval()
    predictions = []
    
    with torch.no_grad():
        for X_test_batch in test_iter:
            X_test_batch = X_test_batch[0].to(device) # load_array 会返回 (features,) 的元组
            outputs = net(X_test_batch)
            predictions.extend(outputs.cpu().numpy().flatten())

    # 逆变换：从 log(1+x) 变回价格 x
    test_predictions = np.expm1(np.array(predictions))
    test_predictions = np.maximum(0, test_predictions) # 确保价格非负
    
    # 4.1 生成提交文件
    submission = pd.DataFrame({
        'SaleID': test_sale_id,
        'price': test_predictions
    })
    
    submission.to_csv(os.path.join('.', SUBMISSION_FILE), index=False)
    print(f"预测结果已保存到：{SUBMISSION_FILE}")

if __name__ == '__main__':
    main()
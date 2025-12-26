import numpy as np
import pandas as pd
from numpy.linalg import inv, det
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- GDA Model Core Functions ---
def train_gda(X, Y):
    """
    训练概率生成模型 (GDA) 并计算模型参数 (均值和共享协方差)。
    Trains the Gaussian Discriminant Analysis (GDA) model and computes parameters (means and shared covariance).
    """
    N = X.shape[0] 
    N1 = np.sum(Y == 1) # 类别1 (>50K) 的样本数 / Number of samples in class 1 (>50K)
    N0 = N - N1        # 类别0 (<=50K) 的样本数 / Number of samples in class 0 (<=50K)
    
    phi = N1 / N # P(Y=1)
    
    # 划分数据 / Split data by class
    X0 = X[Y == 0] 
    X1 = X[Y == 1] 
    
    # 计算均值向量 / Compute mean vectors
    mu0 = np.mean(X0, axis=0) 
    mu1 = np.mean(X1, axis=0) 
    
    # 计算共享协方差矩阵 / Compute shared covariance matrix
    D = X.shape[1] 
    Sigma = np.zeros((D, D))
    
    # 累计协方差 / Accumulate covariance
    for i in range(N0):
        diff = X0[i] - mu0
        Sigma += np.outer(diff, diff) 

    for i in range(N1):
        diff = X1[i] - mu1
        Sigma += np.outer(diff, diff)

    # 共享协方差矩阵（除以总样本数 N）
    Sigma /= N
    
    return phi, mu0, mu1, Sigma

def predict_gda(X_test, phi, mu0, mu1, Sigma, lambda_reg=1e-6):
    """
    【优化后的向量化版本】
    使用 GDA 模型参数对测试数据进行预测。
    Predicts test data using GDA parameters. Includes regularization for singular matrices.
    """
    D = Sigma.shape[0]
    
    # 添加正则化项，避免协方差矩阵奇异或不可逆
    Sigma_reg = Sigma + lambda_reg * np.eye(D)
    
    try:
        Sigma_inv = inv(Sigma_reg)
        log_det_Sigma = np.log(det(Sigma_reg))
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("协方差矩阵正则化后仍奇异，无法计算逆矩阵和行列式。请检查特征或增加正则化项。")

    # ----------------------------------------------------
    # 核心优化：向量化计算所有测试样本的 Mahalanobis 距离
    # ----------------------------------------------------
    
    # 1. 计算与 mu0 的差异矩阵 (N_test, D)
    diff0 = X_test - mu0
    # 2. 计算与 mu1 的差异矩阵 (N_test, D)
    diff1 = X_test - mu1

    # Mahalanobis 距离项： (X - mu)^T Sigma_inv (X - mu)
    # 向量化实现为： np.sum((Diff @ Sigma_inv) * Diff, axis=1) -> (N_test, 1)
    
    # 计算类别 0 的二次项 / Quadratic term for class 0
    # (N_test, D) @ (D, D) -> (N_test, D). (N_test, D) * (N_test, D) -> (N_test, D). sum(axis=1) -> (N_test,)
    quad_term0 = np.sum((diff0 @ Sigma_inv) * diff0, axis=1)
    
    # 计算类别 1 的二次项 / Quadratic term for class 1
    quad_term1 = np.sum((diff1 @ Sigma_inv) * diff1, axis=1)
    
    # log(P(X|Y=k)) = -0.5 * log(det(Sigma)) - 0.5 * quad_term_k
    log_prob_x_y0 = -0.5 * log_det_Sigma - 0.5 * quad_term0
    log_prob_x_y1 = -0.5 * log_det_Sigma - 0.5 * quad_term1
    
    # log(P(Y|X)) ∝ log(P(X|Y)) + log(P(Y))
    log_posterior0 = log_prob_x_y0 + np.log(1 - phi)
    log_posterior1 = log_prob_x_y1 + np.log(phi)
    
    # 决策规则: log_posterior1 > log_posterior0 则预测为 1
    predictions = (log_posterior1 > log_posterior0).astype(int)
            
    return predictions

def save_predictions(predictions, file_path="predict.csv"):
    """
    将预测结果保存到CSV文件中，格式要求：id,label。
    Saves predictions to a CSV file in the required format: id, label.
    """
    ids = np.arange(1, len(predictions) + 1) # id 从 1 开始 / ID starts from 1
    results = pd.DataFrame({'id': ids, 'label': predictions})
    results.to_csv(file_path, index=False)
    print(f"✅ 预测结果已保存至 {file_path}")

# --- Data Loading and Feature Engineering ---

def load_and_transform_data(train_path, test_path):
    """
    加载原始数据，执行特征工程：One-Hot编码和标准化。
    Loads raw data and performs feature engineering: One-Hot Encoding and Standardization.
    """
    
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
        'marital_status', 'occupation', 'relationship', 'race', 
        'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
        'native_country'
    ]
    LABEL_COLUMN = 'income'

    # 1. 加载数据
    try:
        # 使用 header=0 来读取第一行作为列名
        # skipinitialspace=True 用于处理属性值前的空格 (如 " Private" -> "Private")
        train_df = pd.read_csv(train_path, header=0, skipinitialspace=True)
        test_df = pd.read_csv(test_path, header=0, skipinitialspace=True)
    except FileNotFoundError:
        print(f"❌ 错误：未找到数据文件。请确保文件存在于指定的路径：{train_path} 和 {test_path}")
        return None, None, None
    except Exception as e:
        print(f"❌ 错误：加载数据文件时发生问题: {e}")
        return None, None, None


    # 2. 分离标签，并将标签转换为数字 0/1
    # 假设 '>50K' 为 1，'<=50K' 或其他为 0
    Y_train_raw = train_df[LABEL_COLUMN].str.strip().apply(lambda x: 1 if x == '>50K' else 0).values.flatten()
    X_train_raw = train_df.drop(columns=[LABEL_COLUMN])
    X_test_raw = test_df.copy()

    # 3. 识别特征类型
    CONTINUOUS_FEATURES = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    CATEGORICAL_FEATURES = [col for col in COLUMNS if col not in CONTINUOUS_FEATURES]
    
    print("--- 正在执行特征工程 (One-Hot 编码 & 标准化) ---")

    # 4. 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 
             StandardScaler(), # 标准化连续特征
             CONTINUOUS_FEATURES),
            ('cat', 
             OneHotEncoder(handle_unknown='ignore', sparse_output=False), # 独热编码离散特征
             CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    # 5. 拟合训练数据并转换
    # 统一处理缺失值 (Pandas会自动将缺失值转换为NaN，StandardScaler会处理NaN，但OneHotEncoder需要我们确保没有NaN)
    # 对于 census 数据集，空格 ' ?' 通常表示缺失值。这里我们依赖于 OneHotEncoder 的 handle_unknown='ignore' 
    # 和 StandardScaler 的鲁棒性。

    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw) 
    
    # 6. 验证维度
    D_transformed = X_train_transformed.shape[1]
    print(f"原始训练集样本数: {X_train_raw.shape[0]}, 标签数: {Y_train_raw.shape[0]}")
    print(f"原始测试集样本数: {X_test_raw.shape[0]}")
    print(f"转换后的特征维度: {D_transformed} (指导书要求为 106 维，实际取决于数据中的类别数)")

    if D_transformed != 106:
        print(f"⚠️ 警告: 实际特征维度为 {D_transformed}。继续执行，但请注意这可能与指导书要求的 106 维不完全一致。")
    
    return X_train_transformed, Y_train_raw, X_test_transformed

# --- 主执行部分 ---
def main():
    # 1. 设置文件路径 (指向 data/ 目录)
    TRAIN_FILE = 'data/train.csv'
    TEST_FILE = 'data/test.csv'
    
    print(f"加载数据文件: {TRAIN_FILE} 和 {TEST_FILE}")
    
    # 2. 加载并进行特征工程
    X_train, Y_train, X_test = load_and_transform_data(TRAIN_FILE, TEST_FILE)
    
    if X_train is None:
        return

    # 3. 训练 GDA 模型
    print("--- 3. 训练 GDA 模型 (计算均值和共享协方差) ---")
    try:
        phi, mu0, mu1, Sigma = train_gda(X_train, Y_train)
        
        print(f"先验概率 P(Y=1) (phi): {phi:.4f}")
        print(f"共享协方差矩阵维度: {Sigma.shape}")

    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        return

    # 4. 预测测试集
    print("--- 4. 预测测试集 ---")
    try:
        # 使用向量化预测，速度会快很多
        predictions = predict_gda(X_test, phi, mu0, mu1, Sigma, lambda_reg=1e-6)
        print(f"预测完成，得到 {len(predictions)} 个预测结果。")
    except np.linalg.LinAlgError as e:
        print(f"❌ 预测失败，矩阵不可逆: {e}")
        print("请尝试在 predict_gda 函数中调整 lambda_reg 正则化参数。")
        return

    # 5. 保存结果
    save_predictions(predictions, "predict.csv")

if __name__ == '__main__':
    main()
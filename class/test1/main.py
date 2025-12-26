# -*- coding: utf-8 -*-
import numpy as np
import csv
import math
import os

# 设置超参数
LR = 0.01          # 初始学习率
ITERATION = 1000   # 迭代次数
EPSILON = 1e-8     # Adagrad平滑项
FEATURE_HOURS = 9  # 用于预测的数据的时间窗口宽度
TOTAL_POLLUTANTS = 18 # 总污染物的数量

# 数据处理
def process_train_data():
    """读取并规整化数据，创建特征矩阵 X_train 和目标向量 y_train"""
    print("正在规整化数据……\n")
    
    # 使用gbk编码而不是utf-8，
    text = open('data/train.csv', 'r', encoding='gbk', errors='ignore')
    
    row_reader = csv.reader(text, delimiter=",")
    next(row_reader) # 跳过第一行

    data_raw = {}
    pollutant_order = []
    
    train_raw_lines = 0

    for row in row_reader:
        train_raw_lines += 1
        if not row or len(row) < 3 + 24:
            continue
            
        pollutant_name = row[2] # 污染物名称
        pollutant_values = row[3:]

        if pollutant_name not in data_raw:
            data_raw[pollutant_name] = []
            pollutant_order.append(pollutant_name) # 记录出现的顺序
            
        try:
            for val in pollutant_values:
                processed_val = val.replace('NR', '0')
                data_raw[pollutant_name].append(float(processed_val))
        except ValueError as e:
            print(f"跳过报错数据。錯誤: {e}")
            continue
            
    text.close()

    print(f"原始数据共 {train_raw_lines} 行")
        
    data = [data_raw[name] for name in pollutant_order]
    
    PM25_INDEX = pollutant_order.index('PM2.5')
    print(f"PM25 的污染物索引是: {PM25_INDEX} ({pollutant_order[PM25_INDEX]})")

    # 构造样本
    X_train = []
    y_train = []
            
    total_hours = len(data[0])
    
    # 粗略计算一个月或一个数据块的小时数
    HOURS_PER_DATA_CHUNK = 20 * 24
    num_chunks = total_hours // HOURS_PER_DATA_CHUNK
    
    for chunk in range(num_chunks):
        chunk_start_idx = chunk * HOURS_PER_DATA_CHUNK
        num_samples_in_chunk = HOURS_PER_DATA_CHUNK - FEATURE_HOURS
        
        for sample_idx in range(num_samples_in_chunk):
            # i 当前样本的起始小时索引
            i = chunk_start_idx + sample_idx
            
            feature_vector = []
            for pollutant_idx in range(TOTAL_POLLUTANTS):
                # 提取污染物9小时的数据
                features = data[pollutant_idx][i : i + FEATURE_HOURS]
                feature_vector.extend(features)
            
            # 预测第十小时的数据
            target_pm25 = data[PM25_INDEX][i + FEATURE_HOURS]

            X_train.append(feature_vector)
            y_train.append(target_pm25)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

    print(f"训练样本总数: {X_train.shape[0]}")
    print(f"含偏置值的特征维度数量: {X_train.shape[1]}")
    print("数据规整化完成")
    return X_train, y_train

# 训练模型
def train_model(X_train, y_train):
    print("Adagrad训练模型\n")
    # 初始化权重
    w = np.random.rand(X_train.shape[1]) * 0.01 
    prev_gra = np.zeros(X_train.shape[1])
    train_size = X_train.shape[0]

    for i in range(1, ITERATION + 1):
        y_pred = X_train.dot(w)
        loss = y_pred - y_train
        gradient = 2 * X_train.T.dot(loss) / train_size
        
        prev_gra += gradient**2
        ada = np.sqrt(prev_gra + EPSILON) 
        w -= LR * gradient / ada
        
        if i % 100 == 0:
            cost = np.sum(loss**2) / train_size # 平方差 MSE
            cost_a = math.sqrt(cost) # 標準差 RMSE
            print(f"迭代 {i}/{ITERATION}, MSE: {cost:.4f}, RMSE: {cost_a:.4f}")

    print("导出权重")
    np.save('model.npy', w)
    print("模型权重存储为 model.npy")
    return w


def process_test_data():
    """预处理测试数据"""
    print("读取并规整测试数据\n")
    text = open('data/test.csv', 'r', encoding='utf-8', errors='ignore') 

    row_reader = csv.reader(text, delimiter=",")
    
    first_row = next(row_reader)
    # 简单判断数据头是否为空
    if not first_row[0].startswith('id_'):
        pass
    else:
        row_reader = [first_row] + list(row_reader)


    X_test_raw = []
    test_id_list = []
    
    current_sample_features = {}
    current_id = None
    row_count = 0
    test_raw_lines = 0
    
    for row in row_reader:
        test_raw_lines += 1
        if not row or len(row) < 2 + FEATURE_HOURS:
            continue
        
        sample_id = row[0]       
        pollutant_name = row[1]  
        pollutant_values = row[2:2 + FEATURE_HOURS] 

        if current_id is None:
            current_id = sample_id
            
        # 检测id变化
        if sample_id != current_id:
            if len(current_sample_features) == TOTAL_POLLUTANTS:
                feature_vector = []
                for name in sorted(current_sample_features.keys()): 
                    features = [float(val.replace('NR', '0')) for val in current_sample_features[name]]
                    feature_vector.extend(features)
                
                if feature_vector is not None:
                    X_test_raw.append(feature_vector)
                    test_id_list.append(current_id)

            current_id = sample_id
            current_sample_features = {}
            row_count = 0

        current_sample_features[pollutant_name] = pollutant_values
        row_count += 1

    text.close()
    print(f"测试数据行数: {test_raw_lines}")

    # 处理最后一个样本
    if current_id and len(current_sample_features) == TOTAL_POLLUTANTS:
        feature_vector = []
        is_valid = True
        for name in sorted(current_sample_features.keys()):
            features = [float(val.replace('NR', '0')) for val in current_sample_features[name]]
            feature_vector.extend(features)
        if is_valid:
            X_test_raw.append(feature_vector)
            test_id_list.append(current_id)

    X_test = np.array(X_test_raw)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

    print(f"测试样本总数: {X_test.shape[0]}")
    print(f"特征维度数量: {X_test.shape[1]}")
    print("测试数据预处理完成")
    return X_test, test_id_list


def predict_and_save(X_test, test_id_list, w):
    """使用训练好的模型预测"""
    print("预测并保存结果\n")
    
    y_pred_raw = X_test.dot(w) 
    y_pred = np.maximum(0, y_pred_raw) 

    filename = "data/predict.csv"
    with open(filename, "w+", newline='', encoding='utf-8') as text:
        s = csv.writer(text, delimiter=',')
        s.writerow(["id", "value"]) 
        
        for i, pred_value in enumerate(y_pred):
            # 保留两位小数
            s.writerow([test_id_list[i], f'{pred_value:.2f}'])

    print(f"预测结果已成功保存到 {filename}, 共 {len(y_pred)} 行。")


def main():
    # 步驟 1: 處理訓練數據
    X_train, y_train = process_train_data()
    if X_train is None:
        return

    # 步驟 2: 訓練模型
    w = train_model(X_train, y_train)
    if w is None:
        return

    # 步驟 3: 處理測試數據
    X_test, test_id_list = process_test_data()
    if X_test is None or not test_id_list:
        return

    # 步驟 4: 預測並保存結果
    predict_and_save(X_test, test_id_list, w)
    
    print("\nPM2.5 預測腳本執行完畢。")

if __name__ == "__main__":
    main()
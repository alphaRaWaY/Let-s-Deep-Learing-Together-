import os
import sys

# 锁定 PYTHONPATH 逻辑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from model_torch import DrugSentimentLSTM
from model import clean_text

def plot_torch_history(history):
    """绘制 PyTorch 版本的训练过程指标"""
    acc = history['acc']
    loss = history['loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(16, 7))

    # 1. Accuracy 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, marker='s', color='green', label='PyTorch Training Acc')
    plt.title('PyTorch Version: Training Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    # 2. Loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, marker='s', color='red', label='PyTorch Training Loss')
    plt.title('PyTorch Version: Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    # 保存图片
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/drug_torch_training_curves.png')
    print("PyTorch 训练曲线已保存至 results/drug_torch_training_curves.png")
    plt.show()

def torch_train():
    # --- 1. 数据准备 (读取数据并清洗) ---
    print("正在加载数据用于 PyTorch 复现任务...")
    df = pd.read_csv('data/drugsComTrain_raw.csv')
    df['review'] = df['review'].apply(clean_text)
    
    # 标签映射 (1-4->0, 5-6->1, 7-10->2)
    labels = df['rating'].apply(lambda x: 0 if x<=4 else (1 if x<=6 else 2)).values
    
    # --- 2. 简易词表构建 (选做任务复现核心) ---
    # 为了简化，我们统计词频前 10000 的词
    all_text = ' '.join(df['review'].astype(str).tolist())
    words = all_text.split()
    unique_words = list(set(words[:50000])) # 增加词量提高精度
    vocab = {word: i+1 for i, word in enumerate(unique_words[:10000])} 
    
    def tokenize(text, max_len=128):
        seq = [vocab.get(w, 0) for w in str(text).split()]
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq)) # Padding
        return seq[:max_len] # Truncating

    print("正在进行分词处理...")
    X = np.array([tokenize(t) for t in df['review']])
    X_tensor = torch.LongTensor(X)
    y_tensor = torch.LongTensor(labels)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # --- 3. 模型初始化 ---
    device = torch.device("cpu")
    model = DrugSentimentLSTM(vocab_size=len(vocab)+1, 
                              embedding_dim=100, 
                              hidden_dim=128, 
                              output_dim=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # --- 4. 训练循环 (手动实现 .fit 逻辑) ---
    print("\n--- 开始执行 PyTorch 训练循环 ---")
    history = {'loss': [], 'acc': []}
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == batch_y).sum().item()
            
        avg_loss = total_loss / len(loader)
        avg_acc = correct / len(dataset)
        
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    # --- 5. 绘图与保存 ---
    plot_torch_history(history)
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/drug_sentiment_torch.pth')
    print("PyTorch 模型权重已保存。")

if __name__ == "__main__":
    torch_train()
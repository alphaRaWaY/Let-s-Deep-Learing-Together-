import os
import sys

# 锁定 PYTHONPATH 逻辑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.drug_sentiment.model import load_and_preprocess, build_model

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_history(history):
    """按照实验报告要求绘制训练过程指标"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(16, 7))

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, marker='o', label='Training acc')
    plt.plot(epochs, val_acc, marker='o', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, marker='o', label='Training loss')
    plt.plot(epochs, val_loss, marker='o', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    # 保存结果
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/drug_training_curves.png')
    print("训练曲线已保存至 results/drug_training_curves.png")
    plt.show()

def main():
    cfg = load_config('configs/drug_sentiment.yaml')
    
    # 数据加载
    X, y, tokenizer = load_and_preprocess(cfg)
    
    # 模型构建
    model = build_model(cfg)
    model.summary()
    
    # 开始训练
    print("\n--- 开始训练药物评价情感分析模型 ---")
    history = model.fit(
        X, y,
        epochs=cfg['train_params']['epochs'],
        batch_size=cfg['train_params']['batch_size'],
        validation_split=0.2
    )
    
    # 保存结果
    os.makedirs('models', exist_ok=True)
    model.save('models/drug_sentiment_lstm.h5')
    print("模型已保存至 models/drug_sentiment_lstm.h5")
    plot_history(history)

if __name__ == "__main__":
    main()
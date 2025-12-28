import torch
import os
import sys
import matplotlib.pyplot as plt

# 将项目根目录加入路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.pneumonia_task.dataset import get_dataloaders
from src.pneumonia_task.model import DenoisingAutoencoder, PneumoniaCNN
from src.pneumonia_task.train import train_ae, train_cnn

def plot_training_results(ae_hist, cnn_hist):
    """绘制 Train vs Test 的对比折线图，同时包含 AE 的 Loss"""
    epochs_ae = range(1, len(ae_hist['loss']) + 1)
    epochs_cnn = range(1, len(cnn_hist['train_acc']) + 1)
    
    plt.figure(figsize=(18, 5))

    # 1. AE Loss (去噪阶段)
    plt.subplot(1, 3, 1)
    plt.plot(epochs_ae, ae_hist['loss'], 'r-o', label='AE Train Loss')
    plt.title('DAE Denoising Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True, linestyle='--')

    # 2. CNN Accuracy 对比图 (实验报告核心要求)
    plt.subplot(1, 3, 2)
    plt.plot(epochs_cnn, cnn_hist['train_acc'], 'b-o', label='train_acc')
    plt.plot(epochs_cnn, cnn_hist['test_acc'], 'g-o', label='test_acc')
    plt.title('CNN Training vs Test Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(axis='y', linestyle='--')

    # 3. CNN Loss 对比图
    plt.subplot(1, 3, 3)
    plt.plot(epochs_cnn, cnn_hist['train_loss'], 'b-o', label='train_loss')
    plt.plot(epochs_cnn, cnn_hist['test_loss'], 'g-o', label='test_loss')
    plt.title('CNN Training vs Test Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(axis='y', linestyle='--')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/pneumonia_comparison_curves.png')
    print("对比曲线已成功保存至 results/pneumonia_comparison_curves.png")

def main():
    # 1. 加载配置
    cfg = load_config("configs/pneumonia.yaml")
    device = torch.device("cpu") # 强制使用 CPU

    # 2. 准备数据
    train_loader, test_loader, train_size = get_dataloaders(cfg)
    test_size = len(test_loader.dataset) # 获取测试集样本总数

    # 3. 初始化模型
    ae = DenoisingAutoencoder().to(device)
    cnn = PneumoniaCNN(img_size=cfg['image_processing']['size']).to(device)

    # 4. 执行训练并获取历史数据
    print("--- 阶段 1: 训练自编码器 (AE) ---")
    ae_history = train_ae(ae, train_loader, cfg, device)

    print("\n--- 阶段 2: 训练分类器 (CNN) ---")
    cnn_history = train_cnn(cnn, ae, train_loader, test_loader, cfg, device, train_size, test_size)
    # 5. 绘图
    plot_training_results(ae_history, cnn_history)

    # 6. 保存模型
    os.makedirs(cfg['paths']['model_save_path'], exist_ok=True)
    torch.save(ae.state_dict(), os.path.join(cfg['paths']['model_save_path'], "ae_model.pth"))
    torch.save(cnn.state_dict(), os.path.join(cfg['paths']['model_save_path'], "cnn_model.pth"))
    print("\n[完成] 模型与指标图已更新。")

if __name__ == "__main__":
    main()
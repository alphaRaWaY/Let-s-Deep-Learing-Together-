import torch
import os
import sys

# 将上一级目录加入路径以加载 utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config_loader import load_config
from dataset import get_dataloaders
from model import DenoisingAutoencoder, PneumoniaCNN
from train import train_ae, train_cnn

def main():
    # 1. 加载配置 (假设你在项目根目录运行)
    config_path = "configs/pneumonia.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        return
    cfg = load_config(config_path)

    # 2. 设置设备
    device = torch.device("cpu")
    print(f"正在使用设备: {device}")

    # 3. 准备数据
    train_loader, test_loader, train_size = get_dataloaders(cfg)

    # 4. 初始化模型
    ae = DenoisingAutoencoder().to(device)
    cnn = PneumoniaCNN(img_size=cfg['image_processing']['size']).to(device)

    # 5. 执行训练
    print("--- 开始训练自编码器 (去噪任务) ---")
    train_ae(ae, train_loader, cfg, device)

    print("\n--- 开始训练 CNN (分类任务) ---")
    train_cnn(cnn, ae, train_loader, cfg, device, train_size)

    # 6. 保存模型
    os.makedirs(cfg['paths']['model_save_path'], exist_ok=True)
    torch.save(ae.state_dict(), os.path.join(cfg['paths']['model_save_path'], "ae_model.pth"))
    torch.save(cnn.state_dict(), os.path.join(cfg['paths']['model_save_path'], "cnn_model.pth"))
    print("\n模型已保存至 models/ 目录")

if __name__ == "__main__":
    main()
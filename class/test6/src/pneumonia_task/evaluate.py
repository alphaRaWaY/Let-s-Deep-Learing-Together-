import torch
import os
import sys

# 将上一级目录加入路径以加载 utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from model import DenoisingAutoencoder, PneumoniaCNN
from dataset import get_dataloaders
from utils.config_loader import load_config

def evaluate():
    cfg = load_config('configs/pneumonia.yaml')
    device = torch.device("cpu")
    
    # 1. 加载数据
    _, test_loader, _ = get_dataloaders(cfg)
    class_names = ['Covid', 'Normal', 'Viral Pneumonia']

    # 2. 加载模型
    ae = DenoisingAutoencoder().to(device)
    cnn = PneumoniaCNN(img_size=cfg['image_processing']['size']).to(device)
    ae.load_state_dict(torch.load("models/ae_model.pth"))
    cnn.load_state_dict(torch.load("models/cnn_model.pth"))
    ae.eval()
    cnn.eval()

    all_preds = []
    all_labels = []

    print("开始在测试集上评估...")
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            # 步骤：AE去噪 -> CNN分类
            denoised_imgs = ae(imgs)
            outputs = cnn(denoised_imgs)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            # 保存第一组数据的去噪对比图
            if i == 0:
                save_comparison(imgs[0], denoised_imgs[0])

    # 3. 输出报表
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 4. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix on Noisy Test Set')
    plt.savefig("results/confusion_matrix.png")
    print("混淆矩阵图已保存至 results/confusion_matrix.png")

def save_comparison(noisy, clean):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy[0].cpu().numpy(), cmap='gray')
    plt.title("Noisy Input")
    plt.subplot(1, 2, 2)
    plt.imshow(clean[0].cpu().numpy(), cmap='gray')
    plt.title("Denoised Output")
    plt.savefig("results/denoise_test.png")
    print("去噪效果图已保存至 results/denoise_test.png")

if __name__ == "__main__":
    evaluate()
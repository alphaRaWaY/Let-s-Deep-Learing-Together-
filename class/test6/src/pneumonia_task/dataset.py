import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(cfg):
    """
    根据配置文件加载训练集和测试集
    """
    img_size = cfg['image_processing']['size']
    batch_size = cfg['train_params']['batch_size']
    
    # 定义转换操作：缩放、灰度化（实验要求）、转为张量
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=cfg['paths']['train_data'], transform=transform)
    test_dataset = datasets.ImageFolder(root=cfg['paths']['test_data'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_dataset)
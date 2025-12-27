import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> 32
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2), # -> 64
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2), # -> 128
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class PneumoniaCNN(nn.Module):
    def __init__(self, img_size, num_classes=3):
        super(PneumoniaCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 动态计算全连接层输入维度
        # 经过两次 MaxPool2d(2,2)，尺寸变为原来的 1/4
        fc_input_dim = (img_size // 4) * (img_size // 4) * 32
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
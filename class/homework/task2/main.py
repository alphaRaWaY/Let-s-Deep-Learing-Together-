import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# --- 1. 数据集定义 (适配你的 CSV 结构) ---
class VideoDataset(Dataset):
    def __init__(self, csv_file, root_dir, num_frames=16, transform=None):
        # 这里的 root_dir 应该是 data/ 目录
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # csv 里的 video_path 已经是 "videos_train/a001.mp4"
        rel_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        video_full_path = os.path.join(self.root_dir, rel_path)
        
        frames = self._load_video(video_full_path)
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        
        return frames, torch.tensor(label, dtype=torch.long), rel_path

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return [Image.new('RGB', (224, 224))] * self.num_frames
            
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        return frames[:self.num_frames]

# --- 2. 视频分类模型 (ResNet18 + Transformer Encoder) ---
# [cite_start]符合作业要求的：ResNet18 提取帧特征 + Transformer 时序建模 
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoClassifier, self).__init__()
        # [cite_start]帧级特征提取器 [cite: 6]
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # [cite_start]时序建模层 [cite: 6]
        self.embedding_dim = 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # [cite_start]线性分类头 [cite: 6]
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x).view(batch_size, seq_len, -1)
        
        # [cite_start]Transformer 编码 [cite: 6]
        t_out = self.transformer(features)
        # 取序列平均值作为视频表示
        video_feat = t_out.mean(dim=1)
        return self.fc(video_feat)

# --- 3. 视频文字描述 (BLIP 可选加分项) ---
# [cite_start]使用 BLIP 为采样帧生成语句并合并 [cite: 8]
class VideoCaptioner:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, pil_frames):
        # [cite_start]采样关键帧生成描述 [cite: 8]
        inputs = self.processor(pil_frames[len(pil_frames)//2], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

# --- 4. 训练与推理流程 ---
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_data_dir = "data" # 对应你 ll data/ 显示的目录
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # [cite_start]1. 必做任务：视频二分类训练 [cite: 5, 6]
    train_ds = VideoDataset(os.path.join(root_data_dir, 'labels_train.csv'), root_data_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    model = VideoClassifier(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("--- 正在训练视频分类器 ---")
    for epoch in range(2): 
        for frames, labels, _ in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} 完成")

    # [cite_start]2. 测试与加分任务：视频描述 [cite: 7, 8]
    test_ds = VideoDataset(os.path.join(root_data_dir, 'labels_test.csv'), root_data_dir, transform=transform)
    captioner = VideoCaptioner()

    print("\n--- 推理结果 ---")
    model.eval()
    with torch.no_grad():
        for i in range(len(test_ds)):
            frames, label, path = test_ds[i]
            # 分类预测
            pred = torch.argmax(model(frames.unsqueeze(0).to(device)), dim=1).item()
            
            # [cite_start]生成描述 [cite: 8]
            raw_frames = test_ds._load_video(os.path.join(root_data_dir, path))
            caption = captioner.generate(raw_frames)
            
            print(f"视频: {path}")
            print(f"  > 真实标签: {label} | 预测标签: {pred}")
            print(f"  > 视频描述: {caption}")

if __name__ == "__main__":
    run()
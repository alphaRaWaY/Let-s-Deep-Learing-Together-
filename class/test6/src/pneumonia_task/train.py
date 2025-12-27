import torch
import torch.nn as nn
import torch.optim as optim

def train_ae(model, loader, cfg, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['train_params']['lr'])
    noise_factor = cfg['image_processing']['noise_factor']
    
    model.train()
    for epoch in range(cfg['train_params']['ae_epochs']):
        total_loss = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # 添加噪声
            noisy_imgs = torch.clamp(imgs + noise_factor * torch.randn_like(imgs), 0, 1)
            
            outputs = model(noisy_imgs)
            loss = criterion(outputs, imgs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"AE Epoch [{epoch+1}/{cfg['train_params']['ae_epochs']}], Loss: {total_loss/len(loader):.4f}")

def train_cnn(cnn, ae, loader, cfg, device, train_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=cfg['train_params']['lr'])
    
    ae.eval() # 锁定自编码器，只作为去噪器使用
    for epoch in range(cfg['train_params']['cnn_epochs']):
        cnn.train()
        correct = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.no_grad():
                denoised_imgs = ae(imgs) # 先去噪
            
            outputs = cnn(denoised_imgs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct += (outputs.argmax(1) == labels).sum().item()
        print(f"CNN Epoch [{epoch+1}/{cfg['train_params']['cnn_epochs']}], Acc: {100.*correct/train_size:.2f}%")
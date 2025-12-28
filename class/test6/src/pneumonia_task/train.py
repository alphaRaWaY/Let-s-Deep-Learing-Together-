import torch
import torch.nn as nn
import torch.optim as optim

def train_ae(model, train_loader, cfg, device):
    """训练自编码器并记录 Loss"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['train_params']['lr'])
    
    history = {'loss': []}
    
    for epoch in range(cfg['train_params']['ae_epochs']):
        model.train()
        running_loss = 0.0
        for data, _ in train_loader:
            # 模拟噪声输入
            inputs = data.to(device)
            noisy_inputs = inputs + 0.1 * torch.randn_like(inputs)
            
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history['loss'].append(epoch_loss)
        print(f"AE Epoch [{epoch+1}/{cfg['train_params']['ae_epochs']}], Loss: {epoch_loss:.4f}")
    
    return history

def train_cnn(model, ae, train_loader, test_loader, cfg, device, train_size, test_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['train_params']['lr'])
    
    # 记录双向数据
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    for epoch in range(cfg['train_params']['cnn_epochs']):
        # --- 训练阶段 ---
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                inputs = ae(inputs) # 去噪
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_correct += (pred == labels).sum().item()

        # --- 验证阶段 (关键：为了画对比图) ---
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = ae(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                test_correct += (pred == labels).sum().item()

        # 记录结果
        history['train_loss'].append(train_running_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_size)
        history['test_loss'].append(test_running_loss / len(test_loader))
        history['test_acc'].append(test_correct / test_size)
        
        print(f"Epoch {epoch+1} | Train Acc: {history['train_acc'][-1]:.4f} | Test Acc: {history['test_acc'][-1]:.4f}")
    
    return history
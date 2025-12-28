import torch
import torch.nn as nn

class DrugSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
        super(DrugSentimentLSTM, self).__init__()
        # 1. Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM 层 (设置 batch_first=True 匹配数据格式)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           batch_first=True, 
                           dropout=0.2 if n_layers > 1 else 0)
        
        # 3. 全连接层与 Dropout
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x) # [batch_size, seq_len, emb_dim]
        
        # lstm_out 为所有时序的输出，hidden 为最后时刻的状态
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 我们取 LSTM 最后一个时间步的输出作为特征
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out
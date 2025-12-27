import pandas as pd
import re
import tensorflow as tf

# 使用兼容性更好的 API 访问方式
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

# 注意：如果上面依然报错，请尝试直接使用 tf.keras 前缀，例如：
# Tokenizer = tf.keras.preprocessing.text.Tokenizer

def clean_text(text):
    """清洗文本中的 HTML 实体和非字母字符"""
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#039;', "'", text)
    text = re.sub(r'&amp;', "&", text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

def load_and_preprocess(cfg):
    """加载数据并处理成 Keras 可用的格式"""
    train_df = pd.read_csv(cfg['data']['train_path'])
    
    # 标签转换 (指导书要求: 1-4->0, 5-6->1, 7-10->2)
    def label_rating(r):
        if r <= 4: return 0
        if r <= 6: return 1
        return 2

    print("正在预处理文本数据...")
    train_df['review'] = train_df['review'].apply(clean_text)
    train_df['label'] = train_df['rating'].apply(label_rating)

    # 分词
    tokenizer = Tokenizer(num_words=cfg['model_params']['max_words'])
    tokenizer.fit_on_texts(train_df['review'])
    
    sequences = tokenizer.texts_to_sequences(train_df['review'])
    X = pad_sequences(sequences, maxlen=cfg['model_params']['max_len'])
    y = to_categorical(train_df['label'], num_classes=3)
    
    return X, y, tokenizer

def build_model(cfg):
    """构建 LSTM 网络"""
    # 因为你前面已经 from ...layers import Embedding, LSTM ...
    # 所以这里直接写 Embedding，不要写 layers.Embedding
    model = Sequential([
        Embedding(input_dim=cfg['model_params']['max_words'], 
                  output_dim=cfg['model_params']['embedding_dim']),
        SpatialDropout1D(0.2),
        # 针对 CPU 优化：recurrent_dropout 在某些 TF 版本 CPU 上不支持
        LSTM(cfg['model_params']['lstm_units'], dropout=0.2, recurrent_dropout=0),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
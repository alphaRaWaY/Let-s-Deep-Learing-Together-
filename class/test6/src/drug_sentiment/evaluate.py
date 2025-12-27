import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 路径处理
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.drug_sentiment.model import clean_text, load_and_preprocess

def evaluate():
    cfg = load_config('configs/drug_sentiment.yaml')
    
    # 1. 加载模型
    model = load_model('models/drug_sentiment_lstm.h5')
    
    # 2. 加载测试数据 (这里我们复用之前的 preprocess 逻辑，但针对 Test CSV)
    test_df = pd.read_csv(cfg['data']['test_path'])
    test_df['review'] = test_df['review'].apply(clean_text)
    
    # 获取训练时的 tokenizer (为了简单起见，这里重新运行一次 fit，实际工程建议保存 tokenizer.json)
    _, _, tokenizer = load_and_preprocess(cfg) 
    
    sequences = tokenizer.texts_to_sequences(test_df['review'].astype(str))
    X_test = pad_sequences(sequences, maxlen=cfg['model_params']['max_len'])
    
    def label_map(r):
        if r <= 4: return 0
        if r <= 6: return 1
        return 2
    y_true = test_df['rating'].apply(label_map).values

    # 3. 推理
    print("正在测试集上进行评估...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. 报告与绘图
    class_names = ['Negative', 'Neutral', 'Positive']
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Drug Sentiment Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/drug_confusion_matrix.png')
    print("混淆矩阵已保存至 results/drug_confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
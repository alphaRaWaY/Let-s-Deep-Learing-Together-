import os
import sys

# 锁定 PYTHONPATH 逻辑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.drug_sentiment.model import load_and_preprocess, build_model

def main():
    cfg = load_config('configs/drug_sentiment.yaml')
    
    # 数据加载
    X, y, tokenizer = load_and_preprocess(cfg)
    
    # 模型构建
    model = build_model(cfg)
    model.summary()
    
    # 开始训练
    print("\n--- 开始训练药物评价情感分析模型 ---")
    history = model.fit(
        X, y,
        epochs=cfg['train_params']['epochs'],
        batch_size=cfg['train_params']['batch_size'],
        validation_split=0.2
    )
    
    # 保存结果
    os.makedirs('models', exist_ok=True)
    model.save('models/drug_sentiment_lstm.h5')
    print("模型已保存至 models/drug_sentiment_lstm.h5")

if __name__ == "__main__":
    main()
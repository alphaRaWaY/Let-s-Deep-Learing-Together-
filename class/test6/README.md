# 科学计算与数学建模：深度学习实验项目

[![Environment](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange)](https://ubuntu.com/)
[![Framework](https://img.shields.io/badge/Backend-PyTorch%20%2F%20Keras-red)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU%20Only-blue)](#)

本项目是《科学计算与数学建模》课程的实验实现。项目包含两个核心深度学习模块：

肺炎图像识别：基于卷积自编码器（DAE）去噪与 CNN 分类。

药物评价情感分析：基于 Embedding 层与双向 LSTM 的文本分类。

由于数据集过大，本仓库不跟踪数据集。

## 项目架构 (Project Structure)
本项目采用配置驱动设计，将模型逻辑与超参数解耦，方便在不同硬件环境（如 CPU vs GPU）下快速切换。

``` Plaintext
.
├── configs/                # 配置文件 (YAML)
│   ├── pneumonia.yaml      # 肺炎识别任务超参数
│   └── drug_sentiment.yaml # 情感分析任务超参数
├── data/                   # 数据集 (CSV & Images)
│   ├── drugsCom*.csv       # 药物评价数据集
│   ├── train/              # 肺炎训练集
│   └── noisy_test/         # 肺炎噪声测试集
├── src/                    # 源代码
│   ├── pneumonia_task/     # 模块一逻辑 (PyTorch)
│   │   ├── app.py          # Flask 部署接口
│   │   └── train.py        # 训练脚本
│   ├── drug_sentiment/     # 模块二逻辑 (Keras/LSTM)
│   │   ├── train_drug.py   # 训练脚本
│   │   └── evaluate.py     # 评估脚本 (混淆矩阵)
│   └── utils/              # 通用配置加载工具
├── models/                 # 已训练权重 (.pth / .h5)
├── results/                # 评估报告与可视化图表
├── templates/              # Web 前端 HTML 页面
└── static/                 # CSS/JS 静态资源
```
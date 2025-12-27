import torch
import os
import sys
import io
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms

# 确保能找到 src 目录下的模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from src.pneumonia_task.model import DenoisingAutoencoder, PneumoniaCNN
from src.utils.config_loader import load_config

# 初始化 Flask：明确指定模板和静态文件路径
app = Flask(__name__, 
            template_folder="../../templates", 
            static_folder="../../static")

# 加载配置
cfg_path = os.path.join(project_root, 'configs/pneumonia.yaml')
cfg = load_config(cfg_path)
device = torch.device("cpu")

# --- 模型加载 ---
# 动态获取图片尺寸以适配 CNN 展平层
IMG_SIZE = cfg['image_processing']['size']

ae = DenoisingAutoencoder().to(device)
cnn = PneumoniaCNN(img_size=IMG_SIZE).to(device)

# 加载训练好的权重
ae.load_state_dict(torch.load(os.path.join(project_root, "models/ae_model.pth"), map_location=device))
cnn.load_state_dict(torch.load(os.path.join(project_root, "models/cnn_model.pth"), map_location=device))
ae.eval()
cnn.eval()

# --- 图像处理流水线 ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    """渲染前端主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理上传图片并返回 JSON 结果"""
    if 'file' not in request.files:
        return jsonify({"error": "未接收到文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    try:
        # 读取并预处理图片
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # 推理流程：AE 去噪 -> CNN 分类
        with torch.no_grad():
            denoised_img = ae(img_tensor)
            output = cnn(denoised_img)
            prediction_idx = output.argmax(dim=1).item()
        
        classes = ['新冠肺炎 (Covid-19)', '健康 (Normal)', '病毒性肺炎 (Viral Pneumonia)']
        result = classes[prediction_idx]

        # 返回 JSON 格式响应给前端 JavaScript 处理
        return jsonify({
            "prediction": result,
            "status": "success",
            "class_idx": prediction_idx
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("-------------------------------------------")
    print("肺炎医疗影像辅助诊断系统已启动")
    print("本地访问地址: http://127.0.0.1:5000")
    print("-------------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False)
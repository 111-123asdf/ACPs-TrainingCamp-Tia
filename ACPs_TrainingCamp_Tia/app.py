import gradio as gr
import torch
import numpy as np
from model import FlexibleMLP
from config import BaseConfig
from data import load_scaler

cfg = BaseConfig()
model = FlexibleMLP(cfg.input_dim, cfg.hidden_dims, 10,
                    activation=cfg.activation, dropout_rate=cfg.dropout_rate, use_bn=cfg.use_bn)
model.load_state_dict(torch.load('saved_models/best_model.pth', map_location='cpu')['model_state_dict'])
model.eval()

# 加载训练时使用的 scaler，如果不存在则创建默认 scaler
import os
if os.path.exists(cfg.scaler_path):
    scaler = load_scaler(cfg.scaler_path)
else:
    from sklearn.preprocessing import StandardScaler
    from torchvision.datasets import MNIST
    from torchvision import transforms
    scaler = StandardScaler()
    # 使用 MNIST 训练集拟合 scaler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    X_train = np.array([img.numpy() for img, _ in train_dataset])
    scaler.fit(X_train)
    # 保存 scaler 供后续使用
    from data import save_scaler
    os.makedirs(os.path.dirname(cfg.scaler_path), exist_ok=True)
    save_scaler(scaler, cfg.scaler_path)

def digit_from_sketch(image):
    from PIL import Image
    import numpy as np

    if image is None:
        return {}

    try:
        # 处理 dict 类型：优先取 'composite'
        if isinstance(image, dict):
            img_data = image.get('composite')
            if img_data is None:
                raise ValueError("dict 中没有 'composite' 键")
        else:
            img_data = image

        # 转为 numpy 数组
        if isinstance(img_data, np.ndarray):
            arr = img_data
        elif hasattr(img_data, 'convert'):
            arr = np.array(img_data)
        else:
            raise ValueError(f"不支持的图像类型: {type(img_data)}")

        # 转为灰度（假设 RGB）
        if len(arr.shape) == 3:
            gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = arr

        # 颜色反转：MNIST 是白字黑底，Sketchpad 是黑字白底
        gray = 255 - gray

        # 缩放到 28x28
        pil_img = Image.fromarray(gray)
        resized = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        flat = np.array(resized, dtype=np.float32).reshape(1, -1)

        # 归一化到 [0,1]（与训练时 ToTensor 一致）
        flat = flat / 255.0

        # 标准化（使用训练时拟合的 scaler）
        flat = scaler.transform(flat)

        with torch.no_grad():
            x = torch.tensor(flat, dtype=torch.float32)
            output = model(x)
            prob = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(prob).item()

        return {str(i): float(prob[i]) for i in range(10)}

    except Exception as e:
        import traceback
        print("识别错误:", e)
        traceback.print_exc()
        return {}

with gr.Blocks() as demo:
    gr.Markdown("# 手写数字识别 (MNIST 28x28)")
    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad()
            btn = gr.Button("识别")
        with gr.Column():
            label = gr.Label(num_top_classes=3)
    btn.click(digit_from_sketch, inputs=sketch, outputs=label)

demo.launch()

import torch
from model import FlexibleMLP
from config import BaseConfig

def export():
    cfg = BaseConfig()
    model = FlexibleMLP(cfg.input_dim, cfg.hidden_dims, 10,
                        activation=cfg.activation, dropout_rate=cfg.dropout_rate, use_bn=cfg.use_bn)
    model.load_state_dict(torch.load('saved_models/best_model.pth', map_location='cpu')['model_state_dict'])
    model.eval()
    dummy_input = torch.randn(1, cfg.input_dim)  # cfg.input_dim 现在是 784
    torch.onnx.export(model, dummy_input, "mlp_digits.onnx",
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print("ONNX 模型已导出为 mlp_digits.onnx")

if __name__ == '__main__':
    export()

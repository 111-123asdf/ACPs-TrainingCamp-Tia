# 手写数字识别 (MNIST)

基于 PyTorch 的 MNIST 手写数字识别项目，使用 FlexibleMLP 模型，支持 Gradio 交互界面。

## 项目结构

```
.
├── app.py              # Gradio 交互界面
├── config.py           # 超参数配置
├── data.py             # 数据加载、预处理、增强
├── export_onnx.py      # 导出 ONNX 模型
├── hyper_tune.py       # 超参数搜索
├── model.py            # FlexibleMLP 模型定义
├── train.py            # 训练脚本
├── train_utils.py      # 训练工具（训练、评估、早停）
├── requirements.txt    # 依赖列表
└── README.md           # 本文件
```

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

训练完成后会在 `saved_models/` 下生成 `best_model.pth` 和 `scaler.pkl`。

## 启动交互界面

```bash
python app.py
```

在浏览器中打开显示的 URL，使用 Sketchpad 绘制数字，点击“识别”即可看到预测结果。

## 超参数搜索

```bash
python hyper_tune.py
```

结果保存在 `hyper_results/tune_results.json`。

## 导出 ONNX

```bash
python export_onnx.py
```

生成 `mlp_digits.onnx` 文件。

## 依赖

- Python >= 3.8
- PyTorch >= 1.9
- torchvision >= 0.10
- scikit-learn
- numpy
- gradio
- Pillow
- matplotlib

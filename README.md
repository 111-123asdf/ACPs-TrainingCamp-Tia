# ACPs-TrainingCamp-Tia

这个仓库是为了参加 ACPs 智能体互联网实训营提交的代码作品。  
项目是用 PyTorch 训练的一个手写数字识别模型，可以在网页上画数字让它识别。

## 我对 ACPs 的理解

ACPs 是一种让不同 AI 智能体互相通信、合作的协议。我觉得它就像给程序定了一套“共同语言”，让它们能互相调用能力，一起完成更难的任务。

这个项目虽然只是一个简单的识别模型，但我试着把它导出成 ONNX 格式，这样别的程序或智能体也能调用它。这种“把模型变成一个标准接口”的思路，和 ACPs 想让智能体互联的想法有点相似。我希望在实训营里能学到怎么真正让智能体之间“对话”。

## 项目做了什么

用 MNIST 数据集训练了一个多层感知机（MLP），可以识别 0-9 的手写数字。  
还做了一个 Gradio 网页，打开后能在画板上写数字，点“识别”就能看到结果。

## 灵感来源

这个项目是从 `basic_example/SimpleNumberRecognitio.py` 扩展而来的。  
`SimpleNumberRecognitio.py` 用 sklearn 自带的 8x8 手写数字数据集，训练了一个简单的两层 MLP，功能包括：

- 数据划分、标准化、创建 DataLoader
- 定义模型、训练、验证、测试
- 保存加载模型、绘制损失曲线

整个流程注释详细，适合作为入门练习。  
本项目在此基础上，把数据集换成了更大的 MNIST（28x28），增加了数据增强、早停、超参数搜索和 Gradio 网页交互，让模型更实用。

## 文件是干什么的

- `train.py` 用来训练模型，跑完会保存 best_model.pth 和 scaler.pkl
- `app.py` 启动网页画板，可以在浏览器里手写识别
- `model.py` 定义神经网络的结构
- `config.py` 存放参数，比如学习率、隐藏层大小
- `data.py` 负责加载数据、做标准化和加噪声
- `train_utils.py` 训练用的工具：训练一轮、验证、早停
- `hyper_tune.py` 自动搜索超参数（可选）
- `export_onnx.py` 把模型导出成 ONNX 格式

## 怎么运行

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 训练模型
```bash
python train.py
```

3. 启动网页识别
```bash
python app.py
```

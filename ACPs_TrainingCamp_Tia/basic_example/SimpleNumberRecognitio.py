#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(0)
np.random.seed(0)
#%%
# ----------------------------- 1. 数据加载与预处理 -----------------------------
print("1. 加载数据并划分训练/验证/测试集")

# 加载 digits 数据集
digits = load_digits()
X, y = digits.data, digits.target

print(f"原始数据：X shape = {X.shape}, y shape = {y.shape}")
print(f"类别数：{len(digits.target_names)} 类 ")

# 先分出训练+验证集（70%）和测试集（30%）
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# 再从训练+验证集中分出训练集（70% * 70% ≈ 49%）和验证集（70% * 30% ≈ 21%）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.3, random_state=0, stratify=y_train_val
)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# 标准化：对每个特征（像素位置）减去均值除以标准差，使数据分布更稳定
# 注意：只在训练集上 fit，然后转换验证集和测试集，避免数据泄露
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # 计算均值和标准差并转换训练集
X_val = scaler.transform(X_val)           # 使用训练集的参数转换验证集
X_test = scaler.transform(X_test)         # 使用训练集的参数转换测试集

print("标准化完成。")
#%%
# ----------------------------- 2. 转换为 PyTorch 张量并创建 DataLoader -----------------------------
print("2. 转换为张量并创建 DataLoader")

# 将 numpy 数组转为 torch 张量
# 特征要求 float32 类型，标签要求 long 类型（CrossEntropyLoss 需要）
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 使用 TensorDataset 将特征和标签打包在一起
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

# 创建 DataLoader：批量加载数据，支持打乱和多线程
batch_size_train = 64
batch_size_val_test = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size_val_test, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size_val_test, shuffle=False)

print(f"train_loader 批次数: {len(train_loader)} (batch_size={batch_size_train})")
print(f"val_loader   批次数: {len(val_loader)}   (batch_size={batch_size_val_test})")
print(f"test_loader  批次数: {len(test_loader)}  (batch_size={batch_size_val_test})")
#%%
# ----------------------------- 3.（a） 定义 MLP 模型 -----------------------------
print("3. 定义两层 MLP 模型")

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        # 线性层1: input_dim -> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 激活函数
        self.relu = nn.ReLU()
        # 线性层2: hidden_dim -> output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)   # 输出未经过 softmax，因为 CrossEntropyLoss 内部会做
        return out

# 实例化模型
input_dim = 64      # 输入特征维度
hidden_dim = 128    # 隐藏层神经元个数
output_dim = 10     # 10 个类别

model = TwoLayerMLP(input_dim, hidden_dim, output_dim)

# 如果有 GPU，将模型移动到 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备: {device}")
print(model)
#%%
# ----------------------------- 3.(b) 定义损失函数和优化器 -----------------------------
criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器，学习率 0.001
#%%
# ----------------------------- 3.(c) 训练循环 -----------------------------
print("5. 开始训练")

num_epochs = 50   # 训练 50 轮
train_losses = [] # 记录每个 epoch 的平均训练损失

for epoch in range(num_epochs):
    # ---------- 训练阶段 ----------
    model.train()           # 设置为训练模式
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        # 将数据移到 device (CPU/GPU)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 梯度清零（否则梯度会累加）
        optimizer.zero_grad()

        # 前向传播：计算预测输出
        outputs = model(batch_x)          # shape: (batch_size, 10)

        # 计算损失
        loss = criterion(outputs, batch_y)

        # 反向传播：计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)   # 累加该 batch 的总损失

    avg_train_loss = total_loss / len(train_dataset)   # 平均每个样本的损失
    train_losses.append(avg_train_loss)

    # ---------- 验证阶段 ----------
    model.eval()            # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():   # 不计算梯度，节省内存和计算
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)   # 取最大值的索引作为预测类别
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_acc = 100.0 * correct / total

    # 打印每个 epoch 的信息
    print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

print("训练完成！")
#%%
# ----------------------------- 4. 测试集评估 -----------------------------
print("6. 在测试集上评估最终模型")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

test_acc = 100.0 * correct / total
print(f"测试集准确率: {test_acc:.2f}%")
#%%
# ----------------------------- 5. 保存和加载模型 -----------------------------
print("7. 保存模型，然后新建模型并加载权重再次测试")

# 保存模型的状态字典（只保存参数，不保存结构）
torch.save(model.state_dict(), "mlp_digits.pth")
print("模型已保存为 mlp_digits.pth")

# 新建一个相同结构的模型（随机初始化）
new_model = TwoLayerMLP(input_dim, hidden_dim, output_dim)
new_model.to(device)

# 加载之前保存的权重
new_model.load_state_dict(torch.load("mlp_digits.pth", map_location=device, weights_only=True))
print("权重已加载到新模型中")

# 对新模型进行测试
new_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = new_model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

new_test_acc = 100.0 * correct / total
print(f"加载权重后的新模型测试准确率: {new_test_acc:.2f}%")
#%%
# ----------------------------- 6. 绘制训练损失曲线 -----------------------------
print("\n" + "=" * 50)
print("8. 绘制训练损失曲线")
print("=" * 50)

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

print("\n实验全部完成！")
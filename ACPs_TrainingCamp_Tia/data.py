import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import MNIST
from torchvision import transforms
import pickle

def load_data(config, standardize=True):
    """加载 MNIST 数据，返回标准化后的 NumPy 数组
    standardize=False 时返回原始像素值，用于交叉验证
    """
    transform = transforms.Compose([
        transforms.ToTensor(),               # 转换为 [0,1] 的 Tensor
        transforms.Lambda(lambda x: x.view(-1))  # 展平为 784 维
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = MNIST(root='./data', train=False, download=True, transform=transform)

    X_train_val = np.array([img.numpy() for img, _ in train_dataset])
    y_train_val = np.array([label for _, label in train_dataset])
    X_test      = np.array([img.numpy() for img, _ in test_dataset])
    y_test      = np.array([label for _, label in test_dataset])

    # 划分训练集 / 验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y_train_val
    )

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler
    else:
        # 不标准化，直接返回原始数据（用于交叉验证）
        return X_train, y_train, X_val, y_val, X_test, y_test, None

def save_scaler(scaler, path):
    """保存 StandardScaler 对象"""
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(path):
    """加载 StandardScaler 对象"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def add_noise(x, noise_std=0.05):
    """对单个样本添加高斯噪声（数据增强）"""
    noise = torch.randn_like(x) * noise_std
    return x + noise

class AugmentedDataset(Dataset):
    """支持数据增强的数据集"""
    def __init__(self, dataset, noise_std):
        self.dataset = dataset
        self.noise_std = noise_std
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.noise_std > 0:
            x = add_noise(x, self.noise_std)
        return x, y

def create_loaders(X_train, y_train, X_val, y_val, batch_size, aug_noise_std=None):
    """创建训练和验证 DataLoader，训练时可加增强"""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)

    if aug_noise_std and aug_noise_std > 0:
        train_dataset = AugmentedDataset(train_dataset, aug_noise_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

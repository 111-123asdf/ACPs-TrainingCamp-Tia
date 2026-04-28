import itertools, json, os
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import numpy as np
from config import BaseConfig
from data import load_data, create_loaders
from model import FlexibleMLP
from train_utils import train_one_epoch, evaluate, EarlyStopping

def cross_val_score(config, X_raw, y, n_folds=5):
    """X_raw: 未标准化的原始数据（shape [n_samples, 784])
    每折内部重新拟合 Stade人Scaler， 避免数据泄露
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 基于当前折的训练数据拟合 scaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        train_loader, val_loader = create_loaders(
            X_train, y_train, X_val, y_val,
            config.batch_size, config.aug_noise_std
        )
        model = FlexibleMLP(
            config.input_dim, config.hidden_dims, 10,
            config.activation, config.dropout_rate, config.use_bn
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience
        )
        early_stop = EarlyStopping(patience=config.patience, verbose=False)
        use_amp = config.use_amp and device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(1, config.max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            if early_stop(val_loss):
                break
        _, fold_acc = evaluate(model, val_loader, criterion, device)
        scores.append(fold_acc)
    return np.mean(scores)

def main():
    cfg = BaseConfig()
    # 获得未标准化的 48 000 训练样本（与原来超参数搜索数据量一致）
    X_train_val, y_train_val, _, _, _, _, _ = load_data(cfg)
    param_grid = {
        'hidden_dims': [[256, 128], [128, 64], [128]],
        'activation': ['relu', 'gelu'],
        'lr': [0.001, 0.01],
        'batch_size': [64],
    }
    best_score = 0
    best_params = None
    results = []
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        cfg.hidden_dims = params['hidden_dims']
        cfg.activation = params['activation']
        cfg.lr = params['lr']
        cfg.batch_size = params['batch_size']
        score = cross_val_score(cfg, X_train_val, y_train_val, n_folds=cfg.n_folds)
        print(f"Params: {params} -> CV Acc: {score:.2f}%")
        results.append({'params': params, 'cv_acc': score})
        if score > best_score:
            best_score = score
            best_params = params
    print(f"\n最佳参数: {best_params}, 准确率: {best_score:.2f}%")
    os.makedirs('hyper_results', exist_ok=True)
    with open('hyper_results/tune_results.json', 'w') as f:
        json.dump({'best_params': best_params, 'best_cv_acc': best_score, 'all': results}, f, indent=2)

if __name__ == '__main__':
    main()

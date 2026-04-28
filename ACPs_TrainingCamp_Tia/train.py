import os
import torch
import torch.optim as optim
import torch.nn as nn
from config import BaseConfig
from data import load_data, create_loaders, save_scaler
from model import FlexibleMLP
from train_utils import train_one_epoch, evaluate, EarlyStopping

def main():
    cfg = BaseConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 MNIST 数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data(cfg)

    # 创建 DataLoader
    train_loader, val_loader = create_loaders(
        X_train, y_train, X_val, y_val,
        cfg.batch_size, cfg.aug_noise_std
    )

    # 模型
    model = FlexibleMLP(
        cfg.input_dim, cfg.hidden_dims, 10,
        cfg.activation, cfg.dropout_rate, cfg.use_bn
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    early_stop = EarlyStopping(patience=cfg.patience, verbose=True)

    use_amp = cfg.use_amp and device.type == 'cuda'
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    os.makedirs(cfg.model_dir, exist_ok=True)

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler_amp, use_amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, os.path.join(cfg.model_dir, 'best_model.pth'))
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

        if early_stop(val_loss):
            print("Early stopping 触发")
            break

    # 保存 scaler
    save_scaler(scaler, cfg.scaler_path)
    print(f"Scaler 已保存至 {cfg.scaler_path}")

    # 最终测试集评估
    test_loader, _ = create_loaders(X_test, y_test, X_test, y_test, cfg.batch_size)
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"测试集准确率: {test_acc:.2f}%")

if __name__ == '__main__':
    main()

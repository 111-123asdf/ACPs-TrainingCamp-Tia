""" 超参数配置 """

class BaseConfig:
    random_seed = 0
    test_size = 0.2
    n_folds = 5
    batch_size = 64
    max_epochs = 200
    patience = 5
    lr_patience = 3
    lr_factor = 0.5
    use_amp = True

    # 模型结构
    input_dim = 784
    hidden_dims = [512, 256]
    activation = 'gelu'
    dropout_rate = 0.3
    use_bn = True

    # 数据增强
    aug_noise_std = 0.05

    # 优化器
    lr = 0.001
    weight_decay = 1e-4

    # 保存路径
    model_dir = "saved_models"
    scaler_path = "saved_models/scaler.pkl"

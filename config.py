# config.py - Configurações do projeto

# Configurações de dados
SEQ_LENGTH = 100  # Tamanho da sequência temporal (aumentado para capturar mais padrões)
TEST_SIZE = 0.2  # Proporção dos dados para teste
USE_MULTIPLE_FEATURES = True  # Usar múltiplas features ou apenas close

# Configurações do LSTM
LSTM_CONFIG = {
    'layers': [100, 100, 50, 50],  # Neurônios em cada camada LSTM
    'dropout_rates': [0.3, 0.3, 0.2, 0.2],  # Taxa de dropout para cada camada
    'dense_layers': [50, 25],  # Camadas densas finais
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'patience_early_stop': 15,
    'patience_reduce_lr': 10,
    'loss_function': 'huber'  # 'mse', 'mae', 'huber'
}

# Configurações do Random Forest
RF_CONFIG = {
    'n_estimators_options': [100, 200, 300, 500],
    'max_depth_options': [10, 20, 30, None],
    'min_samples_split_options': [2, 5, 10],
    'min_samples_leaf_options': [1, 2, 4],
    'max_features_options': ['sqrt', 'log2', None],
    'bootstrap_options': [True, False],
    'random_search_iterations': 50,
    'cv_folds': 3
}

# Configurações do Ensemble
ENSEMBLE_WEIGHTS = {
    'lstm_weight': 0.7,
    'rf_weight': 0.3
}

# Features técnicas para calcular
TECHNICAL_FEATURES = {
    'price_change': True,
    'volatility': True,
    'price_position': True,
    'moving_averages': {
        'enabled': True,
        'short_window': 5,
        'long_window': 10
    },
    'rsi': {
        'enabled': True,
        'window': 14
    },
    'bollinger_bands': {
        'enabled': True,
        'window': 20,
        'std_dev': 2
    },
    'macd': {
        'enabled': True,
        'fast': 12,
        'slow': 26,
        'signal': 9
    }
}

# Configurações de visualização
PLOT_CONFIG = {
    'figure_size': (15, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'colors': {
        'real': 'blue',
        'lstm': 'red',
        'rf': 'green',
        'ensemble': 'purple'
    }
}

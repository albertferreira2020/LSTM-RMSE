# config.py - Configurações do projeto

# Configurações de dados
SEQ_LENGTH = 120  # Aumentado para capturar padrões temporais mais longos (4 meses de dados)
TEST_SIZE = 0.12  # Reduzido para máximo de dados de treino disponíveis
USE_MULTIPLE_FEATURES = True  # Usar múltiplas features ou apenas close

# Configurações do LSTM
LSTM_CONFIG = {
    'layers': [512, 384, 256, 192, 128, 96, 64, 32],  # Arquitetura muito mais profunda para captura de padrões complexos
    'dropout_rates': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],  # Dropout progressivo refinado
    'dense_layers': [256, 128, 64, 32, 16],  # Camadas densas mais profundas e graduais
    'epochs': 1000,  # Épocas aumentadas para treino profundo
    'batch_size': 8,  # Batch menor para gradientes mais precisos
    'learning_rate': 0.0001,  # Learning rate mais baixo para treino estável e preciso
    'patience_early_stop': 50,  # Mais paciência para convergência profunda
    'patience_reduce_lr': 20,  # Redução de LR mais conservadora
    'min_lr': 0.000001,  # LR mínimo muito baixo para ajuste fino
    'reduce_lr_factor': 0.2,  # Fator de redução mais agressivo
    'loss_function': 'huber',  # Huber loss para robustez contra outliers
    'validation_split': 0.15,  # Validação separada do teste
    'shuffle': False,  # Manter ordem temporal
    'bidirectional': True,  # LSTM bidirecional para capturar dependências futuras/passadas
    'attention': True,  # Mecanismo de atenção para focar em partes importantes da sequência
    'l1_reg': 0.001,  # Regularização L1 para esparsidade
    'l2_reg': 0.01,   # Regularização L2 para suavização
    'clipnorm': 1.0   # Gradient clipping para estabilidade
}

# Configurações do Random Forest
RF_CONFIG = {
    'n_estimators_options': [500, 800, 1200, 1500, 2000],  # Muito mais árvores para precisão máxima
    'max_depth_options': [20, 25, 30, 35, 40, None],  # Profundidades maiores para capturar complexidade
    'min_samples_split_options': [2, 3, 4],  # Valores mais refinados
    'min_samples_leaf_options': [1, 2],  # Folhas menores para máxima granularidade
    'max_features_options': ['sqrt', 'log2', 0.7, 0.8, 0.9, 1.0],  # Mais opções incluindo todas features
    'bootstrap_options': [True],  # Sempre usar bootstrap
    'random_search_iterations': 200,  # Dobrado para busca mais exhaustiva
    'cv_folds': 7,  # Mais folds para validação robusta
    'n_jobs': -1,  # Usar todos os cores
    'random_state': 42,
    'oob_score': True,  # Out-of-bag scoring
    'class_weight': 'balanced',  # Balanceamento automático
    'ccp_alpha': 0.0,  # Pruning mínimo
    'max_samples': 0.9  # 90% das amostras para cada árvore
}

# Configurações do Gradient Boosting
GB_CONFIG = {
    'n_estimators_options': [500, 800, 1200, 1500],  # Muito mais estimadores para precisão
    'learning_rate_options': [0.005, 0.01, 0.02, 0.05, 0.08],  # Learning rates mais refinados
    'max_depth_options': [6, 8, 10, 12, 15],  # Profundidades maiores
    'min_samples_split_options': [2, 3, 4, 5],  # Splits mais refinados
    'min_samples_leaf_options': [1, 2, 3],  # Folhas pequenas para granularidade
    'subsample_options': [0.7, 0.8, 0.9, 1.0],  # Mais opções de subsampling
    'max_features_options': ['sqrt', 'log2', 0.7, 0.8, 0.9],  # Seleção refinada de features
    'random_search_iterations': 150,  # Busca muito mais extensiva
    'cv_folds': 7,  # Validação mais robusta
    'validation_fraction': 0.1,  # Validação interna
    'n_iter_no_change': 20,  # Early stopping mais paciente
    'random_state': 42,
    'tol': 1e-6,  # Tolerância mais baixa para convergência
    'warm_start': True  # Aproveitar treinos anteriores
}

# Configurações do XGBoost (adicional)
XGB_CONFIG = {
    'n_estimators_options': [500, 800, 1200, 1500],  # Muito mais estimadores
    'learning_rate_options': [0.005, 0.01, 0.02, 0.05],  # Learning rates mais baixos
    'max_depth_options': [6, 8, 10, 12, 15],  # Profundidades maiores
    'subsample_options': [0.7, 0.8, 0.9],  # Subsampling refinado
    'colsample_bytree_options': [0.7, 0.8, 0.9, 1.0],  # Seleção de features por árvore
    'colsample_bylevel_options': [0.8, 0.9, 1.0],  # Seleção por nível
    'reg_alpha_options': [0, 0.01, 0.1, 0.5, 1.0],  # L1 regularization expandida
    'reg_lambda_options': [1, 1.5, 2, 3, 5],  # L2 regularization expandida
    'gamma_options': [0, 0.1, 0.5, 1.0],  # Pruning mínimo
    'min_child_weight_options': [1, 3, 5, 7],  # Peso mínimo das folhas
    'random_search_iterations': 100,  # Busca mais extensiva
    'cv_folds': 7,  # Validação robusta
    'early_stopping_rounds': 30,  # Early stopping mais paciente
    'random_state': 42,
    'tree_method': 'hist',  # Método otimizado para velocidade
    'objective': 'reg:squarederror',  # Objetivo de regressão
    'eval_metric': 'rmse'  # Métrica de avaliação
}

# Configurações do Ensemble
ENSEMBLE_WEIGHTS = {
    'lstm_weight': 0.45,  # Peso aumentado do LSTM (modelo principal)
    'rf_weight': 0.2,     # Peso do Random Forest
    'gb_weight': 0.2,     # Peso do Gradient Boosting
    'xgb_weight': 0.15,   # Peso aumentado do XGBoost
    'use_weighted_average': True,  # Usar média ponderada
    'use_stacking': True,  # Stacking ensemble ativado para máxima precisão
    'meta_model': 'ridge',  # Modelo meta para stacking: 'ridge', 'lasso', 'elastic'
    'stacking_cv': 7,     # Cross-validation para stacking
    'blend_method': 'average',  # 'average', 'weighted', 'rank'
    'dynamic_weights': True,  # Pesos dinâmicos baseados na performance recente
    'ensemble_diversity': True  # Promover diversidade entre modelos
}

# Features técnicas para calcular
TECHNICAL_FEATURES = {
    'price_change': True,
    'volatility': True,
    'price_position': True,
    'moving_averages': {
        'enabled': True,
        'windows': [3, 5, 7, 10, 14, 20, 30, 50, 100, 200],  # Janelas muito mais abrangentes
        'types': ['sma', 'ema', 'wma', 'hull']  # Múltiplos tipos de médias móveis
    },
    'rsi': {
        'enabled': True,
        'windows': [9, 14, 21, 25, 30]  # Múltiplos períodos RSI refinados
    },
    'bollinger_bands': {
        'enabled': True,
        'windows': [10, 20, 30, 50],  # Múltiplas janelas
        'std_devs': [1.5, 2, 2.5, 3]  # Diferentes desvios padrão
    },
    'macd': {
        'enabled': True,
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'stochastic': {
        'enabled': True,
        'k_window': 14,
        'd_window': 3
    },
    'williams_r': {
        'enabled': True,
        'window': 14
    },
    'atr': {
        'enabled': True,
        'window': 14
    },
    'cci': {
        'enabled': True,
        'window': 20
    },
    'momentum': {
        'enabled': True,
        'windows': [3, 5, 10, 14, 20, 30]  # Janelas expandidas
    },
    'rate_of_change': {
        'enabled': True,
        'windows': [3, 5, 10, 14, 20, 30]  # Janelas expandidas
    },
    'price_channels': {
        'enabled': True,
        'window': 20
    },
    'fibonacci_retracements': {
        'enabled': True,
        'window': 50
    },
    'ichimoku': {
        'enabled': True,
        'tenkan': 9,
        'kijun': 26,
        'senkou_b': 52
    },
    'adx': {
        'enabled': True,
        'window': 14
    },
    'parabolic_sar': {
        'enabled': True,
        'step': 0.02,
        'max_step': 0.2
    },
    'obv': {
        'enabled': True
    },
    'mfi': {
        'enabled': True,
        'window': 14
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

# Configurações de Pré-processamento
PREPROCESSING_CONFIG = {
    'scaling_method': 'robust',  # 'standard', 'minmax', 'robust', 'quantile'
    'handle_outliers': True,
    'outlier_method': 'isolation_forest',  # Método mais sofisticado para outliers
    'outlier_threshold': 2.5,  # Threshold mais restritivo
    'outlier_contamination': 0.05,  # 5% de contaminação esperada
    'fill_missing': 'interpolate',  # 'forward', 'backward', 'interpolate', 'mean'
    'feature_selection': {
        'enabled': True,
        'method': 'mutual_info',  # 'correlation', 'mutual_info', 'rfe', 'lasso'
        'n_features': 80,  # Número aumentado de features
        'threshold': 0.005,  # Threshold mais baixo para incluir mais features
        'variance_threshold': 0.01  # Remover features com baixa variância
    },
    'lag_features': {
        'enabled': True,
        'lags': [1, 2, 3, 5, 7, 10, 14, 21],  # Múltiplos lags expandidos
        'rolling_stats': {
            'windows': [3, 5, 7, 10, 14, 20, 30],  # Janelas expandidas
            'stats': ['mean', 'std', 'min', 'max', 'skew', 'kurt', 'median']  # Estatísticas expandidas
        }
    },
    'fourier_features': {
        'enabled': True,
        'n_components': 10  # Componentes de Fourier para capturar ciclicidade
    },
    'polynomial_features': {
        'enabled': True,
        'degree': 2,  # Features polinomiais de grau 2
        'interaction_only': True  # Apenas interações, não potências
    }
}

# Configurações de Otimização
OPTIMIZATION_CONFIG = {
    'use_optuna': True,  # Ativar Optuna para otimização bayesiana
    'optuna_trials': 200,  # Mais trials para otimização profunda
    'optuna_timeout': 7200,  # 2 horas de timeout
    'cross_validation': {
        'method': 'time_series',  # 'time_series', 'stratified', 'group'
        'n_splits': 7,  # Mais splits para validação robusta
        'test_size': 0.15,  # Tamanho de teste reduzido
        'gap': 5  # Gap entre treino e teste
    },
    'hyperparameter_search': {
        'method': 'bayesian',  # Otimização bayesiana para máxima eficiência
        'scoring': 'neg_mean_squared_error',
        'refit': 'neg_mean_squared_error',
        'n_iter': 200,  # Mais iterações
        'random_state': 42
    },
    'ensemble_optimization': {
        'enabled': True,
        'method': 'genetic_algorithm',  # Algoritmo genético para pesos
        'population_size': 50,
        'generations': 100
    }
}

# Configurações de Monitoramento
MONITORING_CONFIG = {
    'save_models': True,
    'model_checkpoint_dir': 'checkpoints/',
    'tensorboard_logs': 'logs/tensorboard/',
    'save_predictions': True,
    'save_metrics': True,
    'plot_training_curves': True,
    'save_feature_importance': True,
    'generate_reports': True,
    'log_level': 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
}

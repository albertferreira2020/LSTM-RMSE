# config_optimized.py - Configurações otimizadas para performance

# Configurações de dados OTIMIZADAS
SEQ_LENGTH = 60  # Reduzido de 120 para 60 - mais rápido, ainda efetivo
TEST_SIZE = 0.2  # Mantido para boa validação
USE_MULTIPLE_FEATURES = True

# Configurações do LSTM OTIMIZADAS
LSTM_CONFIG_OPTIMIZED = {
    'layers': [128, 64, 32],  # Arquitetura mais simples e rápida
    'dropout_rates': [0.2, 0.3, 0.4],  # Dropout simplificado
    'dense_layers': [64, 32],  # Menos camadas densas
    'epochs': 200,  # Reduzido de 1000 para 200 - muito mais rápido
    'batch_size': 32,  # Aumentado de 8 para 32 - processamento mais eficiente
    'learning_rate': 0.001,  # Aumentado para convergência mais rápida
    'patience_early_stop': 20,  # Reduzido para parar mais cedo
    'patience_reduce_lr': 10,  # Mais agressivo
    'min_lr': 0.00001,
    'reduce_lr_factor': 0.5,  # Redução mais rápida
    'loss_function': 'mse',  # MSE é mais rápido que Huber
    'validation_split': 0.2,
    'shuffle': False,
    'bidirectional': False,  # Desabilitado - dobra o tempo de processamento
    'attention': False,  # Desabilitado - muito custoso computacionalmente
    'l1_reg': 0.0001,  # Reduzido
    'l2_reg': 0.001,   # Reduzido
    'clipnorm': 1.0
}

# Configurações do Random Forest OTIMIZADAS
RF_CONFIG_OPTIMIZED = {
    'n_estimators_options': [100, 200, 300],  # Muito reduzido de [500-2000] para [100-300]
    'max_depth_options': [10, 15, 20],  # Reduzido de [20-40] para [10-20]
    'min_samples_split_options': [2, 5],  # Simplificado
    'min_samples_leaf_options': [1, 2],  # Mantido
    'max_features_options': ['sqrt', 'log2'],  # Reduzido para opções mais rápidas
    'bootstrap_options': [True],
    'random_search_iterations': 30,  # Drasticamente reduzido de 200 para 30
    'cv_folds': 3,  # Reduzido de 7 para 3 - muito mais rápido
    'n_jobs': -1,
    'random_state': 42,
    'oob_score': True,
    'max_samples': 0.8  # Reduzido para treino mais rápido
}

# Configurações do Gradient Boosting OTIMIZADAS
GB_CONFIG_OPTIMIZED = {
    'n_estimators_options': [100, 200, 300],  # Reduzido de [500-1500] para [100-300]
    'learning_rate_options': [0.05, 0.1, 0.2],  # Maiores para convergência mais rápida
    'max_depth_options': [4, 6, 8],  # Reduzido de [6-15] para [4-8]
    'min_samples_split_options': [2, 5],  # Simplificado
    'min_samples_leaf_options': [1, 2],
    'subsample_options': [0.8, 0.9],  # Reduzido
    'max_features_options': ['sqrt', 'log2'],  # Simplificado
    'random_search_iterations': 20,  # Reduzido de 150 para 20
    'cv_folds': 3,  # Reduzido de 7 para 3
    'validation_fraction': 0.1,
    'n_iter_no_change': 10,  # Reduzido para parar mais cedo
    'random_state': 42,
    'tol': 1e-4,  # Tolerância maior para convergência mais rápida
    'warm_start': False  # Desabilitado para simplicidade
}

# Features técnicas OTIMIZADAS (reduzidas para performance)
TECHNICAL_FEATURES_OPTIMIZED = {
    'price_change': True,
    'volatility': True,
    'price_position': True,
    'moving_averages': {
        'enabled': True,
        'windows': [5, 10, 20, 50],  # Reduzido de 10 janelas para 4
        'types': ['sma', 'ema']  # Apenas 2 tipos ao invés de 4
    },
    'rsi': {
        'enabled': True,
        'windows': [14, 21]  # Reduzido de 5 para 2 períodos
    },
    'bollinger_bands': {
        'enabled': True,
        'windows': [20],  # Apenas 1 janela ao invés de 4
        'std_devs': [2]  # Apenas 1 desvio ao invés de 4
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
        'enabled': False,  # Desabilitado para reduzir features
    },
    'atr': {
        'enabled': True,
        'window': 14
    },
    'cci': {
        'enabled': False,  # Desabilitado para reduzir features
    },
    'momentum': {
        'enabled': True,
        'windows': [5, 10, 20]  # Reduzido de 6 para 3 janelas
    },
    'rate_of_change': {
        'enabled': True,
        'windows': [5, 10, 20]  # Reduzido de 6 para 3 janelas
    },
    # Desabilitados indicadores menos importantes para ganhar velocidade
    'obv': {'enabled': False},
    'adx': {'enabled': False},
    'fibonacci_retracements': {'enabled': False},
    'ichimoku': {'enabled': False},
    'parabolic_sar': {'enabled': False}
}

# Configurações para modo RÁPIDO (datasets pequenos/médios)
QUICK_MODE_CONFIG = {
    'max_features_to_use': 30,  # Limita número total de features
    'use_feature_selection': True,  # Ativa seleção automática de features
    'feature_selection_method': 'mutual_info',  # Método rápido de seleção
    'reduce_lstm_complexity': True,  # Simplifica ainda mais o LSTM
    'skip_hyperparameter_tuning': False,  # Mantém busca de hiperparâmetros mas reduzida
    'use_early_stopping_aggressive': True,  # Early stopping mais agressivo
    'parallel_model_training': True,  # Treina modelos em paralelo quando possível
}

# Configurações do Ensemble OTIMIZADAS
ENSEMBLE_WEIGHTS_OPTIMIZED = {
    'lstm_weight': 0.4,
    'rf_weight': 0.3,
    'gb_weight': 0.3,
    'use_weighted_average': True,
    'use_stacking': False,  # Desabilitado - muito custoso para datasets pequenos
    'blend_method': 'average',  # Mais simples e rápido
    'dynamic_weights': False,  # Desabilitado para simplicidade
    'ensemble_diversity': False  # Desabilitado para performance
}

# Função para detectar automaticamente o melhor modo baseado no tamanho dos dados
def get_optimal_config(data_size):
    """
    Retorna configuração otimizada baseada no tamanho dos dados
    
    Args:
        data_size (int): Número de registros no dataset
    
    Returns:
        dict: Configurações otimizadas
    """
    if data_size < 500:
        mode = "ultrafast"
        seq_length = 30
        lstm_epochs = 50
        rf_estimators = [50, 100]
        search_iterations = 10
    elif data_size < 1000:
        mode = "fast"
        seq_length = 40
        lstm_epochs = 100
        rf_estimators = [100, 150]
        search_iterations = 15
    elif data_size < 2000:
        mode = "balanced"
        seq_length = 60
        lstm_epochs = 200
        rf_estimators = [100, 200, 300]
        search_iterations = 30
    elif data_size < 5000:
        mode = "thorough"
        seq_length = 80
        lstm_epochs = 300
        rf_estimators = [200, 300, 500]
        search_iterations = 50
    else:
        mode = "comprehensive"
        seq_length = 120
        lstm_epochs = 500
        rf_estimators = [300, 500, 800]
        search_iterations = 100
    
    config = {
        'mode': mode,
        'seq_length': seq_length,
        'lstm_epochs': lstm_epochs,
        'rf_estimators': rf_estimators,
        'search_iterations': search_iterations,
        'batch_size': min(32, max(8, data_size // 50)),  # Batch size adaptativo
        'cv_folds': 3 if data_size < 1000 else 5,
        'early_stopping_patience': 10 if data_size < 1000 else 20
    }
    
    print(f"🚀 Modo otimizado selecionado: {mode.upper()}")
    print(f"📊 Para {data_size} registros:")
    print(f"   • Sequence Length: {seq_length}")
    print(f"   • LSTM Epochs: {lstm_epochs}")
    print(f"   • RF Estimators: {rf_estimators}")
    print(f"   • Search Iterations: {search_iterations}")
    print(f"   • Batch Size: {config['batch_size']}")
    
    return config

# Configuração padrão otimizada para 1500 registros
DEFAULT_OPTIMIZED_CONFIG = get_optimal_config(1500)

print("✅ Configurações otimizadas carregadas!")
print("💡 Para usar: from config_optimized import *")
print("📈 Esperado: 3-5x mais rápido que configuração original")

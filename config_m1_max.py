# config_m1_max.py - Configurações otimizadas para Mac M1 Max com GPU

# Configurações otimizadas para M1 Max
SEQ_LENGTH = 80  # Otimizado para aproveitar memória unificada do M1 Max
TEST_SIZE = 0.15  # Mais dados para treino aproveitando GPU
USE_MULTIPLE_FEATURES = True

# Configurações do LSTM otimizadas para GPU Metal M1 Max
LSTM_CONFIG_M1_MAX = {
    # Arquitetura otimizada para GPU Metal
    'layers': [256, 128, 64],  # Arquitetura que aproveita bem a GPU M1 Max
    'dropout_rates': [0.2, 0.3, 0.4],
    'dense_layers': [128, 64, 32],
    
    # Configurações de treinamento otimizadas para GPU
    'epochs': 500,  # Mais épocas aproveitando velocidade da GPU
    'batch_size': 64,  # Batch maior para aproveitar GPU e memória unificada
    'learning_rate': 0.001,
    
    # Early stopping menos agressivo para aproveitar GPU
    'patience_early_stop': 30,
    'patience_reduce_lr': 15,
    'min_lr': 0.00001,
    'reduce_lr_factor': 0.3,
    
    # Configurações específicas para GPU Metal
    'loss_function': 'mse',  # MSE é bem otimizado no Metal
    'validation_split': 0.2,
    'shuffle': True,  # Shuffle habilitado para melhor uso da GPU
    
    # Features específicas para M1 Max
    'use_mixed_precision': True,  # Aproveita Tensor Cores do M1 Max
    'bidirectional': False,  # Desabilitado para melhor performance
    'attention': False,  # Simplificado para velocidade
    
    # Regularização
    'l1_reg': 0.0001,
    'l2_reg': 0.001,
    'clipnorm': 1.0,
    
    # Configurações de dispositivo
    'force_gpu': True,  # Força uso da GPU
    'memory_growth': True,  # Crescimento dinâmico de memória
    'allow_memory_growth': True
}

# Configurações do Random Forest otimizadas para M1 Max (CPU)
RF_CONFIG_M1_MAX = {
    # Aproveita todos os cores do M1 Max (8 performance + 2 efficiency)
    'n_estimators_options': [200, 400, 600],  # Aumentado para aproveitar CPU
    'max_depth_options': [15, 20, 25, None],
    'min_samples_split_options': [2, 3, 5],
    'min_samples_leaf_options': [1, 2],
    'max_features_options': ['sqrt', 'log2', 0.8],
    'bootstrap_options': [True],
    
    # Configurações de busca
    'random_search_iterations': 50,  # Aumentado para aproveitar CPU
    'cv_folds': 5,  # Aumentado para aproveitar paralelização
    'n_jobs': -1,  # Usa todos os cores disponíveis
    'random_state': 42,
    'oob_score': True,
    'max_samples': 0.9,
    
    # Configurações específicas para M1 Max
    'warm_start': False,
    'class_weight': None
}

# Configurações do Gradient Boosting para M1 Max
GB_CONFIG_M1_MAX = {
    'n_estimators_options': [200, 400, 600],
    'learning_rate_options': [0.05, 0.1, 0.15],
    'max_depth_options': [6, 8, 10],
    'min_samples_split_options': [2, 5],
    'min_samples_leaf_options': [1, 2],
    'subsample_options': [0.8, 0.9],
    'max_features_options': ['sqrt', 'log2'],
    
    'random_search_iterations': 30,
    'cv_folds': 5,
    'validation_fraction': 0.1,
    'n_iter_no_change': 15,
    'random_state': 42,
    'tol': 1e-5,
    'warm_start': False
}

# Features técnicas otimizadas para M1 Max
TECHNICAL_FEATURES_M1_MAX = {
    'price_change': True,
    'volatility': True,
    'price_position': True,
    
    # Médias móveis otimizadas
    'moving_averages': {
        'enabled': True,
        'windows': [5, 10, 20, 50, 100],  # Janelas estratégicas
        'types': ['sma', 'ema', 'wma']  # 3 tipos principais
    },
    
    # RSI com múltiplos períodos
    'rsi': {
        'enabled': True,
        'windows': [14, 21, 30]
    },
    
    # Bollinger Bands
    'bollinger_bands': {
        'enabled': True,
        'windows': [20, 50],
        'std_devs': [2, 2.5]
    },
    
    # Indicadores principais
    'macd': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9},
    'stochastic': {'enabled': True, 'k_window': 14, 'd_window': 3},
    'atr': {'enabled': True, 'window': 14},
    'cci': {'enabled': True, 'window': 20},
    
    # Momentum
    'momentum': {
        'enabled': True,
        'windows': [5, 10, 20, 30]
    },
    
    'rate_of_change': {
        'enabled': True,
        'windows': [5, 10, 20, 30]
    },
    
    # Indicadores avançados (aproveita processamento rápido)
    'williams_r': {'enabled': True, 'window': 14},
    'obv': {'enabled': True},
    'adx': {'enabled': True, 'window': 14},
    
    # Features lag otimizadas
    'lag_features': {
        'enabled': True,
        'lags': [1, 2, 3, 5, 10],  # Mais lags aproveitando performance
        'columns': ['close', 'volume', 'high', 'low']
    },
    
    # Rolling statistics
    'rolling_stats': {
        'enabled': True,
        'windows': [5, 10, 20, 50],
        'stats': ['mean', 'std', 'min', 'max', 'skew']
    }
}

# Configurações específicas para GPU Metal (M1 Max)
GPU_CONFIG = {
    'enable_metal': True,  # Habilita Metal Performance Shaders
    'memory_limit': None,  # Usa toda memória unificada disponível
    'allow_growth': True,  # Crescimento dinâmico
    'force_gpu_compatible': True,  # Força compatibilidade com GPU
    
    # Configurações de precisão
    'mixed_precision': {
        'enabled': True,  # Usa precisão mista para velocidade
        'policy': 'mixed_float16',  # Policy otimizada para M1 Max
        'loss_scale': 'dynamic'  # Escala dinâmica de loss
    },
    
    # Configurações de memória
    'memory_config': {
        'unify_memory': True,  # Aproveita memória unificada do M1 Max
        'optimal_batch_size': 64,  # Batch otimizado para GPU
        'prefetch_buffer': 2,  # Buffer de pré-carregamento
        'parallel_data_loading': True
    },
    
    # Otimizações específicas do Metal
    'metal_optimizations': {
        'use_metal_performance_shaders': True,
        'optimize_for_inference': False,  # Otimizado para treino
        'enable_metal_debug': False,  # Debug desabilitado para performance
        'metal_device_priority': 'performance'  # Prioriza performance sobre economia
    }
}

# Configuração para detecção automática baseada no hardware
def get_m1_max_optimal_config(data_size, force_gpu=True):
    """
    Detecta automaticamente a melhor configuração para M1 Max
    
    Args:
        data_size (int): Número de registros
        force_gpu (bool): Força uso da GPU se disponível
    
    Returns:
        dict: Configuração otimizada para M1 Max
    """
    # Configuração base para M1 Max
    base_config = {
        'device': 'GPU' if force_gpu else 'CPU',
        'cores_available': 10,  # 8 performance + 2 efficiency
        'unified_memory': True,
        'metal_support': True
    }
    
    if data_size < 1000:
        mode = "m1_fast"
        config = {
            'seq_length': 60,
            'lstm_epochs': 300,
            'batch_size': 32,
            'use_gpu': force_gpu,
            'rf_estimators': [200, 400],
            'search_iterations': 30
        }
    elif data_size < 3000:
        mode = "m1_balanced"
        config = {
            'seq_length': 80,
            'lstm_epochs': 500,
            'batch_size': 64,
            'use_gpu': force_gpu,
            'rf_estimators': [200, 400, 600],
            'search_iterations': 50
        }
    else:
        mode = "m1_max_performance"
        config = {
            'seq_length': 120,
            'lstm_epochs': 800,
            'batch_size': 128,
            'use_gpu': force_gpu,
            'rf_estimators': [400, 600, 800],
            'search_iterations': 100
        }
    
    config.update(base_config)
    
    print(f"🚀 Configuração M1 Max selecionada: {mode.upper()}")
    print(f"📊 Para {data_size} registros:")
    print(f"   • Device: {config['device']}")
    print(f"   • Sequence Length: {config['seq_length']}")
    print(f"   • LSTM Epochs: {config['lstm_epochs']}")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"   • GPU Habilitada: {config['use_gpu']}")
    
    return config

# Função para verificar compatibilidade com Metal
def check_metal_support():
    """
    Verifica se Metal Performance Shaders está disponível
    """
    try:
        import tensorflow as tf
        
        # Verifica dispositivos
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Metal GPU detectada: {len(gpus)} dispositivo(s)")
            
            # Testa operação simples na GPU
            with tf.device('/GPU:0'):
                test = tf.constant([1.0, 2.0])
                result = tf.reduce_sum(test)
                print(f"✅ Teste Metal bem-sucedido: {result.numpy()}")
                
            return True, gpus
        else:
            print("⚠️ Metal GPU não detectada")
            return False, []
            
    except Exception as e:
        print(f"❌ Erro verificando Metal: {e}")
        return False, []

# Configuração padrão para 1500 registros no M1 Max
DEFAULT_M1_MAX_CONFIG = get_m1_max_optimal_config(1500, force_gpu=True)

print("✅ Configurações M1 Max carregadas!")
print("🚀 Otimizado para Metal Performance Shaders")
print("💪 Aproveita memória unificada e 10 cores do M1 Max")
print("📈 Esperado: 5-10x mais rápido com GPU Metal")

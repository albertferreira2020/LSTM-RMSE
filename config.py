# config_parallel_max.py - Configuração para máxima paralelização

import os
import multiprocessing

# Detecta automaticamente o número de cores disponíveis
CPU_CORES = multiprocessing.cpu_count()
print(f"🔧 Cores CPU detectados: {CPU_CORES}")

# Para M1 Max: 10 cores total (8 performance + 2 efficiency)
# Otimizações específicas para M1 Max
M1_MAX_PERFORMANCE_CORES = 8
M1_MAX_EFFICIENCY_CORES = 2
M1_MAX_TOTAL_CORES = M1_MAX_PERFORMANCE_CORES + M1_MAX_EFFICIENCY_CORES

# Configurações de paralelização máxima
PARALLEL_CONFIG = {
    'tensorflow_threads': {
        'inter_op_parallelism': M1_MAX_PERFORMANCE_CORES,  # 8 cores para operações entre grafos
        'intra_op_parallelism': M1_MAX_PERFORMANCE_CORES,  # 8 cores para operações dentro do grafo
        'omp_num_threads': M1_MAX_PERFORMANCE_CORES,       # OpenMP threads
        'use_multiprocessing': True,
        'workers': 6,  # Aumentado de 4 para 6 (deixa 2 cores livres para sistema)
        'max_queue_size': 32,  # Aumentado para maior throughput
    },
    'sklearn_models': {
        'n_jobs': -1,  # Usa todos os cores disponíveis
        'parallel_backend': 'threading',  # Para modelos que suportam
        'batch_size': 'auto',  # Deixa o sklearn otimizar
    },
    'data_processing': {
        'chunk_size': 10000,  # Processa dados em chunks paralelos
        'parallel_feature_engineering': True,
        'n_jobs_features': M1_MAX_PERFORMANCE_CORES,
    }
}

# Configurações TensorFlow otimizadas para máxima performance
TENSORFLOW_CONFIG_MAX_PARALLEL = {
    'mixed_precision': True,    # Precision mista para velocidade
    'xla_acceleration': True,   # XLA compilation para otimização
    'memory_growth': True,      # Crescimento dinâmico de memória GPU
    'allow_growth': True,
    'device_placement': True,   # Placement automático otimizado
    'inter_op_parallelism_threads': M1_MAX_PERFORMANCE_CORES,
    'intra_op_parallelism_threads': M1_MAX_PERFORMANCE_CORES,
    'use_multiprocessing': True,
    'workers': 6,
    'max_queue_size': 32,
    'threads_per_gpu': 2,  # Para múltiplas streams GPU
}

# LSTM otimizado para paralelização máxima
LSTM_CONFIG_PARALLEL_MAX = {
    'layers': [256, 128, 64],  # Arquitetura otimizada para GPU paralela
    'dropout_rates': [0.2, 0.3, 0.4],
    'dense_layers': [128, 64],
    'epochs': 300,
    'batch_size': 64,  # Batch maior para melhor GPU utilization
    'learning_rate': 0.001,
    'patience_early_stop': 25,
    'patience_reduce_lr': 10,
    'min_lr': 0.00001,
    'reduce_lr_factor': 0.5,
    'loss_function': 'mse',
    'validation_split': 0.2,
    'shuffle': False,
    'bidirectional': False,  # Mais rápido sem bidirecional
    'attention': False,      # Mais rápido sem attention
    'l1_reg': 0.0001,
    'l2_reg': 0.001,
    'clipnorm': 1.0,
    # Configurações de paralelização máxima
    'use_mixed_precision': True,
    'use_xla': True,
    'force_gpu': True,
    'prefetch_buffer_size': 'AUTOTUNE',
    'num_parallel_calls': 'AUTOTUNE',
}

# Random Forest com paralelização máxima
RF_CONFIG_PARALLEL_MAX = {
    'n_estimators_options': [200, 400, 600],  # Aumentado mas mantendo eficiência
    'max_depth_options': [15, 20, 25],
    'min_samples_split_options': [2, 4],
    'min_samples_leaf_options': [1, 2],
    'max_features_options': ['sqrt', 'log2', 0.8],
    'bootstrap_options': [True],
    'random_search_iterations': 50,  # Aumentado de 30 para 50
    'cv_folds': 5,  # Aumentado de 3 para 5
    'n_jobs': -1,   # Usa todos os cores
    'random_state': 42,
    'oob_score': True,
    'max_samples': 0.9,
    # Configurações extras para performance
    'warm_start': False,
    'class_weight': None,  # Mais rápido sem balanceamento
    'criterion': 'squared_error',  # Padrão otimizado
}

# Gradient Boosting paralelizado
GB_CONFIG_PARALLEL_MAX = {
    'n_estimators_options': [200, 400, 600],
    'learning_rate_options': [0.05, 0.1, 0.15],
    'max_depth_options': [6, 8, 10],
    'min_samples_split_options': [2, 4],
    'min_samples_leaf_options': [1, 2],
    'subsample_options': [0.8, 0.9],
    'max_features_options': ['sqrt', 'log2'],
    'random_search_iterations': 30,  # Mantido para balance
    'cv_folds': 5,
    'validation_fraction': 0.1,
    'n_iter_no_change': 15,
    'random_state': 42,
    'tol': 1e-4,
    'warm_start': False,
}

# Features técnicas otimizadas para processamento paralelo
TECHNICAL_FEATURES_PARALLEL = {
    'price_change': True,
    'volatility': True,
    'price_position': True,
    'moving_averages': {
        'enabled': True,
        'windows': [5, 10, 20, 50],
        'types': ['sma', 'ema'],
        'parallel_computation': True,  # Calcula MAs em paralelo
    },
    'rsi': {
        'enabled': True,
        'windows': [14, 21],
        'parallel_computation': True,
    },
    'bollinger_bands': {
        'enabled': True,
        'windows': [20],
        'std_devs': [2],
        'parallel_computation': True,
    },
    'macd': {
        'enabled': True,
        'fast': 12,
        'slow': 26,
        'signal': 9,
        'parallel_computation': True,
    },
    'stochastic': {
        'enabled': True,
        'k_window': 14,
        'd_window': 3,
    },
    'atr': {
        'enabled': True,
        'window': 14,
    },
    # Configurações de paralelização
    'parallel_feature_computation': True,
    'chunk_processing': True,
    'n_jobs': M1_MAX_PERFORMANCE_CORES,
}

# Configurações de dados otimizadas
SEQ_LENGTH_PARALLEL = 60  # Otimizado para balance velocidade/precisão
TEST_SIZE_PARALLEL = 0.2
USE_MULTIPLE_FEATURES_PARALLEL = True

# Configurações de ensemble paralelo
ENSEMBLE_CONFIG_PARALLEL = {
    'parallel_training': True,  # Treina modelos em paralelo quando possível
    'parallel_prediction': True,  # Previsões em paralelo
    'n_jobs': M1_MAX_PERFORMANCE_CORES,
    'use_joblib_parallel': True,
    'backend': 'threading',  # Para I/O bound operations
    'batch_size': 'auto',
}

print("✅ Configuração de paralelização máxima carregada!")
print(f"🚀 Configurado para {CPU_CORES} cores ({M1_MAX_PERFORMANCE_CORES} performance + {M1_MAX_EFFICIENCY_CORES} efficiency)")
print(f"🔧 TensorFlow workers: {TENSORFLOW_CONFIG_MAX_PARALLEL['workers']}")
print(f"📊 Batch size LSTM: {LSTM_CONFIG_PARALLEL_MAX['batch_size']}")
print(f"🌲 Random Forest paralelo: {RF_CONFIG_PARALLEL_MAX['n_jobs']} jobs")

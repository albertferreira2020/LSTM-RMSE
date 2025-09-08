# config_deep_training.py - Configurações específicas para treino profundo e precisão máxima

import tensorflow as tf

# Configurações avançadas do TensorFlow/Keras
TENSORFLOW_CONFIG = {
    'mixed_precision': True,  # Usar mixed precision para velocidade
    'xla_acceleration': True,  # XLA compilation para otimização
    'memory_growth': True,    # Crescimento dinâmico da memória GPU
    'device_placement': True, # Log de placement de dispositivos
    'inter_op_parallelism': 0,  # Auto
    'intra_op_parallelism': 0,  # Auto
    'allow_soft_placement': True
}

# Configurações de GPU (se disponível)
GPU_CONFIG = {
    'use_gpu': True,
    'gpu_memory_limit': None,  # None = usar toda memória disponível
    'allow_growth': True,      # Permitir crescimento dinâmico
    'virtual_gpu_devices': 1   # Número de GPUs virtuais
}

# Callbacks avançados para LSTM
ADVANCED_CALLBACKS = {
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 50,
        'restore_best_weights': True,
        'verbose': 1,
        'mode': 'min',
        'min_delta': 1e-6
    },
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.2,
        'patience': 20,
        'min_lr': 1e-7,
        'verbose': 1,
        'mode': 'min',
        'cooldown': 10
    },
    'model_checkpoint': {
        'filepath': 'checkpoints/best_model_{epoch:02d}_{val_loss:.6f}.h5',
        'monitor': 'val_loss',
        'save_best_only': True,
        'save_weights_only': False,
        'verbose': 1,
        'mode': 'min'
    },
    'csv_logger': {
        'filename': 'logs/training_log.csv',
        'separator': ',',
        'append': True
    },
    'tensorboard': {
        'log_dir': 'logs/tensorboard',
        'histogram_freq': 10,
        'write_graph': True,
        'write_images': True,
        'update_freq': 'batch',
        'profile_batch': '10,20'
    },
    'lr_scheduler': {
        'type': 'cosine_annealing',  # 'cosine_annealing', 'exponential', 'step'
        'T_max': 100,                # Para cosine annealing
        'eta_min': 1e-7             # LR mínimo
    },
    'plateau_scheduler': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 15,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 5,
        'min_lr': 1e-7
    }
}

# Configurações de regularização avançada
REGULARIZATION_CONFIG = {
    'dropout_schedule': {
        'enabled': True,
        'initial_rate': 0.1,
        'final_rate': 0.5,
        'schedule_type': 'linear'  # 'linear', 'exponential', 'cosine'
    },
    'batch_normalization': {
        'enabled': True,
        'momentum': 0.99,
        'epsilon': 1e-3,
        'center': True,
        'scale': True
    },
    'layer_normalization': {
        'enabled': True,
        'epsilon': 1e-6
    },
    'weight_decay': {
        'enabled': True,
        'rate': 1e-4
    },
    'gaussian_noise': {
        'enabled': True,
        'stddev': 0.1
    },
    'alpha_dropout': {
        'enabled': True,
        'rate': 0.1
    }
}

# Arquiteturas alternativas do LSTM
LSTM_ARCHITECTURES = {
    'deep_residual': {
        'use_residual_connections': True,
        'residual_every_n_layers': 2,
        'dense_shortcuts': True
    },
    'attention_mechanism': {
        'use_self_attention': True,
        'attention_heads': 8,
        'attention_dropout': 0.1,
        'use_positional_encoding': True
    },
    'encoder_decoder': {
        'enabled': True,
        'encoder_layers': [512, 256, 128],
        'decoder_layers': [128, 256, 512],
        'attention_bridge': True
    },
    'transformer_blocks': {
        'enabled': True,
        'num_blocks': 4,
        'head_size': 64,
        'num_heads': 8,
        'ff_dim': 256,
        'dropout': 0.1
    }
}

# Configurações de ensemble avançado
ADVANCED_ENSEMBLE = {
    'meta_learners': {
        'ridge': {'alpha': [0.1, 1.0, 10.0]},
        'lasso': {'alpha': [0.1, 1.0, 10.0]},
        'elastic_net': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
        'svr': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
        'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
        'xgboost': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
    },
    'blending_strategies': {
        'simple_average': {'weights': 'equal'},
        'weighted_average': {'weights': 'performance_based'},
        'rank_average': {'method': 'borda_count'},
        'bayesian_model_averaging': {'prior': 'uniform'}
    },
    'dynamic_ensemble': {
        'enabled': True,
        'window_size': 100,        # Janela para calcular performance recente
        'adaptation_rate': 0.1,    # Taxa de adaptação dos pesos
        'min_weight': 0.05,        # Peso mínimo para qualquer modelo
        'diversity_penalty': 0.1   # Penalidade por baixa diversidade
    }
}

# Configurações de validação cruzada temporal
TIME_SERIES_CV = {
    'method': 'expanding_window',  # 'expanding_window', 'sliding_window', 'blocked_cv'
    'initial_train_size': 0.6,     # 60% dos dados para treino inicial
    'step_size': 30,               # Dias para avançar a janela
    'test_size': 30,               # Dias para teste
    'gap': 5,                      # Gap entre treino e teste
    'purged_cv': {
        'enabled': True,
        'embargo_td': 5            # Embargo de 5 períodos
    }
}

# Configurações de feature engineering avançado
ADVANCED_FEATURES = {
    'market_regime': {
        'enabled': True,
        'lookback_window': 252,    # 1 ano de dados
        'volatility_threshold': 0.02,
        'trend_threshold': 0.01
    },
    'volatility_clustering': {
        'enabled': True,
        'garch_model': True,
        'arch_lags': 5,
        'garch_lags': 5
    },
    'fractal_dimension': {
        'enabled': True,
        'window_sizes': [10, 20, 50]
    },
    'hurst_exponent': {
        'enabled': True,
        'window_size': 100
    },
    'entropy_measures': {
        'enabled': True,
        'sample_entropy': True,
        'approximate_entropy': True,
        'permutation_entropy': True
    },
    'wavelets': {
        'enabled': True,
        'wavelet_type': 'db4',
        'levels': 4
    }
}

# Configurações de monitoramento em tempo real
MONITORING_CONFIG = {
    'performance_tracking': {
        'enabled': True,
        'metrics': ['mse', 'mae', 'mape', 'directional_accuracy', 'sharpe_ratio'],
        'rolling_window': 100,
        'alert_thresholds': {
            'mse_increase': 0.5,      # 50% de aumento no MSE
            'directional_accuracy': 0.45  # Abaixo de 45%
        }
    },
    'drift_detection': {
        'enabled': True,
        'method': 'page_hinkley',  # 'page_hinkley', 'adwin', 'kswin'
        'threshold': 50,
        'alpha': 0.9999
    },
    'model_retraining': {
        'enabled': True,
        'trigger_threshold': 0.1,   # 10% de degradação
        'min_retrain_interval': 7,  # Mínimo 7 dias entre retreinos
        'retrain_data_window': 252  # 1 ano de dados para retreino
    }
}

# Configurações de deployment e produção
PRODUCTION_CONFIG = {
    'model_versioning': {
        'enabled': True,
        'storage_backend': 'local',  # 'local', 's3', 'gcs'
        'max_versions': 10,
        'auto_cleanup': True
    },
    'inference_optimization': {
        'batch_prediction': True,
        'model_caching': True,
        'prediction_caching': {
            'enabled': True,
            'ttl_seconds': 300  # 5 minutos
        }
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'handlers': ['file', 'console'],
        'file_path': 'logs/production.log',
        'max_file_size': '10MB',
        'backup_count': 5
    }
}

# Configurações de segurança
SECURITY_CONFIG = {
    'data_encryption': {
        'enabled': False,  # Ativar se necessário
        'algorithm': 'AES-256'
    },
    'access_control': {
        'enabled': False,  # Ativar se necessário
        'api_key_required': False
    },
    'audit_logging': {
        'enabled': True,
        'log_predictions': True,
        'log_model_updates': True,
        'retention_days': 90
    }
}

def setup_tensorflow_config():
    """Configura o TensorFlow para performance otimizada"""
    if TENSORFLOW_CONFIG['mixed_precision']:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    if TENSORFLOW_CONFIG['xla_acceleration']:
        tf.config.optimizer.set_jit(True)
    
    # Configurar GPUs se disponíveis
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and GPU_CONFIG['use_gpu']:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, GPU_CONFIG['allow_growth'])
                if GPU_CONFIG['gpu_memory_limit']:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=GPU_CONFIG['gpu_memory_limit']
                        )]
                    )
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")

def get_advanced_callbacks():
    """Retorna callbacks avançados para treino do LSTM"""
    callbacks = []
    
    # Early Stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(**ADVANCED_CALLBACKS['early_stopping']))
    
    # Reduce LR
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**ADVANCED_CALLBACKS['reduce_lr']))
    
    # Model Checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(**ADVANCED_CALLBACKS['model_checkpoint']))
    
    # CSV Logger
    callbacks.append(tf.keras.callbacks.CSVLogger(**ADVANCED_CALLBACKS['csv_logger']))
    
    # TensorBoard
    callbacks.append(tf.keras.callbacks.TensorBoard(**ADVANCED_CALLBACKS['tensorboard']))
    
    return callbacks

def print_config_summary():
    """Imprime um resumo das configurações de treino profundo"""
    print("=" * 80)
    print("CONFIGURAÇÕES DE TREINO PROFUNDO E PRECISÃO MÁXIMA")
    print("=" * 80)
    print(f"GPU habilitada: {GPU_CONFIG['use_gpu']}")
    print(f"Mixed Precision: {TENSORFLOW_CONFIG['mixed_precision']}")
    print(f"XLA Acceleration: {TENSORFLOW_CONFIG['xla_acceleration']}")
    print(f"Regularização avançada: {REGULARIZATION_CONFIG['batch_normalization']['enabled']}")
    print(f"Attention mechanism: {LSTM_ARCHITECTURES['attention_mechanism']['use_self_attention']}")
    print(f"Ensemble dinâmico: {ADVANCED_ENSEMBLE['dynamic_ensemble']['enabled']}")
    print(f"Monitoramento drift: {MONITORING_CONFIG['drift_detection']['enabled']}")
    print(f"Features avançadas: {len([k for k, v in ADVANCED_FEATURES.items() if v.get('enabled', False)])}/6 habilitadas")
    print("=" * 80)

if __name__ == "__main__":
    print_config_summary()

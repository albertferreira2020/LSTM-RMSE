# hyperparameters_optimized.py - Hiperparâmetros otimizados para máxima precisão

"""
Hiperparâmetros otimizados baseados em:
1. Pesquisa bibliográfica de melhores práticas
2. Testes empíricos em dados financeiros
3. Otimização bayesiana
4. Cross-validation temporal
"""

# ============================================================================
# HIPERPARÂMETROS LSTM PROFUNDO
# ============================================================================

LSTM_DEEP_CONFIG = {
    # Arquitetura da rede
    'architecture': {
        'type': 'bidirectional_stacked',  # Tipo de arquitetura
        'layers': [512, 384, 256, 192, 128, 96, 64, 32],  # Camadas decrescentes
        'return_sequences': [True, True, True, True, True, True, True, False],
        'bidirectional_layers': [True, True, True, False, False, False, False, False],
        'attention_layers': [False, False, True, False, True, False, False, False]
    },
    
    # Regularização por camada
    'regularization': {
        'dropout_rates': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        'recurrent_dropout': [0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25],
        'l1_regularization': [0.0001, 0.0001, 0.0005, 0.0005, 0.001, 0.001, 0.001, 0.001],
        'l2_regularization': [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01],
        'activity_regularization': [0.0001, 0.0001, 0.0001, 0.0005, 0.0005, 0.001, 0.001, 0.001]
    },
    
    # Configurações de treino
    'training': {
        'epochs': 1000,
        'batch_size': 4,  # Batch muito pequeno para gradientes precisos
        'learning_rate': 0.00005,  # LR muito baixo para treino estável
        'beta_1': 0.9,    # Adam beta_1
        'beta_2': 0.999,  # Adam beta_2
        'epsilon': 1e-8,  # Adam epsilon
        'amsgrad': True,  # Usar AMSGrad
        'clipnorm': 0.5,  # Gradient clipping
        'clipvalue': None
    },
    
    # Callbacks e monitoramento
    'callbacks': {
        'early_stopping': {
            'patience': 100,
            'min_delta': 1e-7,
            'restore_best_weights': True
        },
        'reduce_lr': {
            'patience': 30,
            'factor': 0.1,
            'min_lr': 1e-8,
            'cooldown': 20
        },
        'cosine_annealing': {
            'T_max': 200,
            'eta_min': 1e-8
        }
    }
}

# ============================================================================
# HIPERPARÂMETROS RANDOM FOREST OTIMIZADOS
# ============================================================================

RF_OPTIMIZED_CONFIG = {
    'base_estimator': {
        'criterion': 'squared_error',
        'splitter': 'best',
        'max_features': 'sqrt',
        'random_state': 42
    },
    
    'ensemble_parameters': {
        'n_estimators': 2000,  # Muitas árvores para estabilidade
        'max_depth': 35,       # Profundidade para capturar complexidade
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.8,   # 80% das features por árvore
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1,
        'verbose': 1
    },
    
    'advanced_parameters': {
        'max_samples': 0.9,      # 90% das amostras por árvore
        'max_leaf_nodes': None,  # Sem limite de folhas
        'min_impurity_decrease': 1e-7,
        'min_weight_fraction_leaf': 0.0,
        'ccp_alpha': 0.0,       # Sem pruning
        'class_weight': None    # Não aplicável para regressão
    }
}

# ============================================================================
# HIPERPARÂMETROS GRADIENT BOOSTING OTIMIZADOS
# ============================================================================

GB_OPTIMIZED_CONFIG = {
    'boosting_parameters': {
        'n_estimators': 1500,
        'learning_rate': 0.01,  # LR baixo para convergência estável
        'max_depth': 12,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.85,      # Subsampling para regularização
        'max_features': 0.8,
        'random_state': 42
    },
    
    'regularization': {
        'validation_fraction': 0.1,
        'n_iter_no_change': 30,
        'tol': 1e-6,
        'alpha': 0.9           # Quantile loss alpha
    },
    
    'advanced_settings': {
        'loss': 'huber',       # Robust loss function
        'criterion': 'friedman_mse',
        'warm_start': True,
        'init': None,
        'verbose': 1,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 1e-7,
        'ccp_alpha': 0.0
    }
}

# ============================================================================
# HIPERPARÂMETROS XGBOOST OTIMIZADOS
# ============================================================================

XGB_OPTIMIZED_CONFIG = {
    'boosting_parameters': {
        'n_estimators': 1500,
        'learning_rate': 0.01,
        'max_depth': 10,
        'min_child_weight': 1,
        'gamma': 0.1,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'colsample_bynode': 0.9
    },
    
    'regularization': {
        'reg_alpha': 0.1,      # L1 regularization
        'reg_lambda': 2.0,     # L2 regularization
        'alpha': 0.1,          # L1 reg term on weights
        'lambda': 2.0          # L2 reg term on weights
    },
    
    'performance': {
        'tree_method': 'hist',
        'sketch_eps': 0.03,
        'scale_pos_weight': 1,
        'max_delta_step': 0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    },
    
    'early_stopping': {
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse',
        'maximize': False
    }
}

# ============================================================================
# CONFIGURAÇÕES DE ENSEMBLE AVANÇADO
# ============================================================================

ENSEMBLE_OPTIMIZED_CONFIG = {
    'weights': {
        'lstm': 0.50,      # Peso principal para LSTM
        'rf': 0.18,        # Random Forest
        'gb': 0.18,        # Gradient Boosting
        'xgb': 0.14        # XGBoost
    },
    
    'stacking': {
        'enabled': True,
        'meta_learner': 'ridge',
        'cv_folds': 7,
        'use_probas': False,
        'stack_method': 'auto',
        'passthrough': False
    },
    
    'blending': {
        'method': 'weighted_rank',  # Método de blending
        'rank_weights': True,
        'outlier_detection': True,
        'outlier_threshold': 3.0
    },
    
    'dynamic_weighting': {
        'enabled': True,
        'window_size': 50,
        'adaptation_rate': 0.05,
        'performance_metric': 'mse',
        'min_weight': 0.05,
        'max_weight': 0.70
    }
}

# ============================================================================
# CONFIGURAÇÕES DE FEATURE ENGINEERING OTIMIZADAS
# ============================================================================

FEATURE_ENGINEERING_CONFIG = {
    'sequence_length': 120,    # 4 meses de dados diários
    'prediction_horizon': 1,   # Prever 1 dia à frente
    
    'technical_indicators': {
        'moving_averages': {
            'periods': [3, 5, 8, 13, 21, 34, 55, 89, 144, 233],  # Fibonacci
            'types': ['sma', 'ema', 'wma', 'hull', 'tema']
        },
        'momentum': {
            'rsi_periods': [7, 14, 21, 28],
            'macd_config': [(12, 26, 9), (5, 35, 5), (19, 39, 9)],
            'stoch_periods': [14, 21],
            'williams_r_periods': [14, 21],
            'roc_periods': [5, 10, 20, 30]
        },
        'volatility': {
            'bollinger_periods': [10, 20, 30],
            'bollinger_stds': [1.5, 2.0, 2.5],
            'atr_periods': [7, 14, 21],
            'keltner_periods': [10, 20],
            'donchian_periods': [10, 20, 50]
        },
        'volume': {
            'obv': True,
            'mfi_periods': [14, 21],
            'vwap': True,
            'ad_line': True,
            'cmf_periods': [20, 21]
        }
    },
    
    'statistical_features': {
        'rolling_stats': {
            'windows': [5, 10, 20, 30, 60],
            'stats': ['mean', 'std', 'skew', 'kurt', 'min', 'max', 'median', 'mad']
        },
        'lag_features': {
            'lags': [1, 2, 3, 5, 8, 13, 21],
            'diff_lags': [1, 2, 3, 5]
        },
        'fourier_transform': {
            'n_components': 15,
            'drop_sum': True
        }
    },
    
    'market_microstructure': {
        'bid_ask_spread': True,
        'tick_direction': True,
        'volume_weighted_price': True,
        'trade_intensity': True,
        'order_flow_imbalance': True
    }
}

# ============================================================================
# CONFIGURAÇÕES DE PREPROCESSAMENTO OTIMIZADAS
# ============================================================================

PREPROCESSING_OPTIMIZED_CONFIG = {
    'scaling': {
        'method': 'robust_quantile',  # Robusto a outliers
        'quantile_range': (5.0, 95.0),
        'feature_wise': True,
        'preserve_zeros': False
    },
    
    'outlier_handling': {
        'method': 'isolation_forest',
        'contamination': 0.03,  # 3% de outliers esperados
        'n_estimators': 200,
        'max_features': 1.0,
        'bootstrap': False,
        'behaviour': 'new'  # Para versões mais recentes
    },
    
    'feature_selection': {
        'method': 'hybrid',  # Combinação de métodos
        'methods': ['mutual_info', 'f_regression', 'rfe'],
        'n_features_to_select': 80,
        'scoring': 'r2',
        'cross_validation': 5
    },
    
    'missing_values': {
        'strategy': 'iterative',
        'estimator': 'bayesian_ridge',
        'max_iter': 20,
        'tol': 1e-3,
        'n_nearest_features': 50,
        'initial_strategy': 'median'
    }
}

# ============================================================================
# CONFIGURAÇÕES DE VALIDAÇÃO E TESTE
# ============================================================================

VALIDATION_CONFIG = {
    'cross_validation': {
        'method': 'purged_group_time_series',
        'n_splits': 7,
        'embargo': 5,          # 5 dias de embargo
        'test_size': 30,       # 30 dias de teste
        'gap': 2,              # 2 dias de gap
        'max_train_size': None
    },
    
    'walk_forward': {
        'enabled': True,
        'training_window': 252,  # 1 ano
        'step_size': 21,         # 1 mês
        'expanding_window': True,
        'refit_frequency': 21    # Retreinar mensalmente
    },
    
    'holdout_validation': {
        'test_size': 0.1,        # 10% para teste final
        'validation_size': 0.15, # 15% para validação
        'stratify': False,       # Não aplicável para séries temporais
        'shuffle': False         # Manter ordem temporal
    }
}

# ============================================================================
# MÉTRICAS DE AVALIAÇÃO OTIMIZADAS
# ============================================================================

METRICS_CONFIG = {
    'regression_metrics': [
        'mse', 'rmse', 'mae', 'mape', 'r2_score', 'explained_variance',
        'mean_poisson_deviance', 'mean_gamma_deviance', 'mean_absolute_percentage_error'
    ],
    
    'financial_metrics': [
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown',
        'directional_accuracy', 'hit_ratio', 'profit_factor', 'win_rate'
    ],
    
    'statistical_tests': [
        'ljung_box', 'adf_test', 'kpss_test', 'jarque_bera',
        'shapiro_wilk', 'anderson_darling'
    ],
    
    'custom_metrics': {
        'trend_prediction_accuracy': True,
        'volatility_prediction_accuracy': True,
        'regime_detection_accuracy': True,
        'tail_risk_prediction': True
    }
}

# ============================================================================
# CONFIGURAÇÕES DE OTIMIZAÇÃO BAYESIANA
# ============================================================================

BAYESIAN_OPTIMIZATION_CONFIG = {
    'optuna_settings': {
        'n_trials': 500,
        'timeout': 14400,        # 4 horas
        'n_jobs': -1,
        'show_progress_bar': True,
        'study_name': 'financial_prediction_optimization',
        'storage': None,         # In-memory
        'load_if_exists': True
    },
    
    'search_spaces': {
        'lstm_layers': {
            'type': 'categorical',
            'choices': [
                [256, 128, 64],
                [512, 256, 128],
                [512, 384, 256, 128],
                [512, 384, 256, 192, 128],
                [512, 384, 256, 192, 128, 64]
            ]
        },
        'learning_rate': {
            'type': 'log_uniform',
            'low': 1e-6,
            'high': 1e-2
        },
        'batch_size': {
            'type': 'categorical',
            'choices': [4, 8, 16, 32]
        },
        'dropout_rate': {
            'type': 'uniform',
            'low': 0.1,
            'high': 0.6
        }
    },
    
    'pruning': {
        'enabled': True,
        'pruner': 'median',
        'n_startup_trials': 20,
        'n_warmup_steps': 50,
        'interval_steps': 10
    }
}

def get_optimized_config(model_type='lstm'):
    """
    Retorna configuração otimizada para o modelo especificado
    
    Args:
        model_type: 'lstm', 'rf', 'gb', 'xgb', 'ensemble'
    
    Returns:
        dict: Configuração otimizada
    """
    configs = {
        'lstm': LSTM_DEEP_CONFIG,
        'rf': RF_OPTIMIZED_CONFIG,
        'gb': GB_OPTIMIZED_CONFIG,
        'xgb': XGB_OPTIMIZED_CONFIG,
        'ensemble': ENSEMBLE_OPTIMIZED_CONFIG
    }
    
    return configs.get(model_type, LSTM_DEEP_CONFIG)

def print_optimization_summary():
    """Imprime resumo das otimizações implementadas"""
    print("=" * 100)
    print("HIPERPARÂMETROS OTIMIZADOS PARA MÁXIMA PRECISÃO")
    print("=" * 100)
    
    print(f"📊 LSTM: {len(LSTM_DEEP_CONFIG['architecture']['layers'])} camadas profundas")
    print(f"🌲 Random Forest: {RF_OPTIMIZED_CONFIG['ensemble_parameters']['n_estimators']} árvores")
    print(f"🚀 Gradient Boosting: {GB_OPTIMIZED_CONFIG['boosting_parameters']['n_estimators']} estimadores")
    print(f"⚡ XGBoost: {XGB_OPTIMIZED_CONFIG['boosting_parameters']['n_estimators']} estimadores")
    print(f"🎯 Ensemble: {len(ENSEMBLE_OPTIMIZED_CONFIG['weights'])} modelos combinados")
    print(f"📈 Features: {len(FEATURE_ENGINEERING_CONFIG['technical_indicators'])} grupos de indicadores")
    print(f"🔍 Validação: {VALIDATION_CONFIG['cross_validation']['n_splits']} folds com embargo")
    print(f"🎛️ Otimização: {BAYESIAN_OPTIMIZATION_CONFIG['optuna_settings']['n_trials']} trials bayesianos")
    
    print("\n📋 CARACTERÍSTICAS PRINCIPAIS:")
    print("• Arquitetura LSTM bidirecional com atenção")
    print("• Ensemble stacking com meta-learner")
    print("• Regularização L1/L2 otimizada por camada")
    print("• Feature engineering com 80+ indicadores técnicos")
    print("• Validação cruzada temporal com embargo")
    print("• Otimização bayesiana de hiperparâmetros")
    print("• Detecção automática de outliers")
    print("• Seleção híbrida de features")
    print("• Métricas financeiras especializadas")
    print("=" * 100)

if __name__ == "__main__":
    print_optimization_summary()

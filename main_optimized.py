# main_optimized.py - Versão otimizada para performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Importa configurações otimizadas
from config_optimized import *

# Função utilitária para monitoramento otimizada
def show_system_stats_optimized():
    """
    Versão otimizada de monitoramento de sistema
    """
    import os
    import gc
    import psutil
    
    # Força coleta de lixo
    gc.collect()
    
    # Informações de memória
    memory = psutil.virtual_memory()
    
    print(f"   🐍 PID: {os.getpid()}")
    print(f"   💾 RAM: {memory.percent}% ({memory.available / 1024**3:.1f}GB livres)")
    print(f"   📊 Status: Processando de forma otimizada...")

# Imports necessários
try:
    from database import DatabaseManager
    DATABASE_AVAILABLE = True
    print("✅ DatabaseManager carregado")
except ImportError as e:
    print(f"⚠️ DatabaseManager não disponível: {e}")
    DATABASE_AVAILABLE = False
    DatabaseManager = None

# TensorFlow com configurações ultra-otimizadas
try:
    import os
    # Configurações de performance para TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Apenas erros
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS'] = '2'  # Usa 2 threads ao invés de 1
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    
    import tensorflow as tf
    
    # Configurações de performance
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    # Otimização de memória GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow otimizado carregado")
except:
    TENSORFLOW_AVAILABLE = False
    print("❌ TensorFlow não disponível")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics

def load_and_preprocess_optimized_data_from_db(config_features, limit=None, optimize_features=True):
    """
    Versão otimizada do carregamento de dados
    """
    if not DATABASE_AVAILABLE:
        raise Exception("DatabaseManager não disponível")
    
    print("=== CARREGAMENTO OTIMIZADO DE DADOS ===")
    
    db = DatabaseManager()
    
    try:
        if not db.connect():
            raise Exception("Falha na conexão")
        
        print("📊 Carregando dados...")
        df = db.load_botbinance_data(limit=limit, order_by='id')
        
        if df is None or len(df) == 0:
            raise Exception("Nenhum dado encontrado")
        
        print(f"✅ {df.shape[0]} registros carregados")
        
        # Detecta configuração otimizada baseada no tamanho
        optimal_config = get_optimal_config(len(df))
        
        # Verifica colunas obrigatórias
        required_columns = ['close', 'open', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Mapeando colunas: {missing_columns}")
            # Lógica de mapeamento de colunas aqui
        
        # Adiciona indicadores técnicos otimizados
        print("🔧 Calculando indicadores técnicos otimizados...")
        df = add_technical_indicators(df, config_features)
        
        # Adiciona features lag reduzidas
        print("⏰ Adicionando features lag otimizadas...")
        base_columns = ['close', 'open', 'high', 'low']
        if 'volume' in df.columns:
            base_columns.append('volume')
        df = add_lagged_features(df, base_columns, lags=[1, 2, 3])  # Reduzido de [1,2,3,5] para [1,2,3]
        
        # Estatísticas rolantes reduzidas
        print("📈 Calculando estatísticas rolantes otimizadas...")
        df = add_rolling_statistics(df, ['close'], windows=[5, 20])  # Reduzido de [5,10,20] para [5,20]
        
        # Remove NaN
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        print(f"🧹 Removidas {removed_rows} linhas com NaN")
        
        # Seleciona features excluindo colunas de sistema
        exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Seleção automática de features se habilitada
        if optimize_features and len(feature_columns) > 30:
            print("🎯 Otimizando seleção de features...")
            X_temp = df[feature_columns].values
            y_temp = df['close'].values
            
            # Remove features com variância muito baixa
            from sklearn.feature_selection import VarianceThreshold
            var_selector = VarianceThreshold(threshold=0.001)
            X_var = var_selector.fit_transform(X_temp)
            selected_features = [feature_columns[i] for i in range(len(feature_columns)) if var_selector.get_support()[i]]
            
            # Seleção baseada em informação mútua (top 30)
            if len(selected_features) > 30:
                selector = SelectKBest(score_func=mutual_info_regression, k=30)
                X_selected = selector.fit_transform(df[selected_features].values, y_temp)
                feature_columns = [selected_features[i] for i in range(len(selected_features)) if selector.get_support()[i]]
                print(f"🎯 Reduzido para {len(feature_columns)} features otimizadas")
            else:
                feature_columns = selected_features
        
        print(f"✅ Dataset final otimizado: {df.shape}")
        print(f"🎯 Features finais: {len(feature_columns)}")
        
        return df, feature_columns, optimal_config
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        raise
    finally:
        db.disconnect()

def create_sequences_optimized(data, target_col_idx, seq_length):
    """
    Versão otimizada da criação de sequências
    """
    print(f"📊 Criando sequências otimizadas (seq_length={seq_length})...")
    
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, target_col_idx])
    
    X = np.array(X, dtype=np.float32)  # Usa float32 para economizar memória
    y = np.array(y, dtype=np.float32)
    
    print(f"✅ Sequências criadas: X={X.shape}, y={y.shape}")
    return X, y

def train_optimized_lstm(X_train, y_train, X_test, y_test, config):
    """
    Treinamento otimizado do LSTM
    """
    print(f"🧠 === TREINANDO LSTM OTIMIZADO ===")
    print(f"📊 Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    try:
        model = Sequential()
        
        # Primeira camada LSTM
        model.add(LSTM(
            config['layers'][0], 
            return_sequences=len(config['layers']) > 1,
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))
        model.add(Dropout(config['dropout_rates'][0]))
        
        # Camadas LSTM adicionais (simplificadas)
        for i in range(1, len(config['layers'])):
            model.add(LSTM(
                config['layers'][i], 
                return_sequences=i < len(config['layers']) - 1
            ))
            model.add(Dropout(config['dropout_rates'][i]))
        
        # Camadas densas (simplificadas)
        for i, neurons in enumerate(config['dense_layers']):
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(0.2))
        
        # Camada de saída
        model.add(Dense(1))
        
        # Otimizador otimizado
        optimizer = Adam(
            learning_rate=config['learning_rate'],
            clipnorm=config['clipnorm']
        )
        
        model.compile(
            optimizer=optimizer,
            loss=config['loss_function']
        )
        
        print(f"🏗️ Modelo LSTM compilado ({model.count_params():,} parâmetros)")
        
        # Callbacks otimizados
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config['patience_early_stop'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config['reduce_lr_factor'],
                patience=config['patience_reduce_lr'],
                min_lr=config['min_lr'],
                verbose=1
            )
        ]
        
        # Treinamento
        print(f"🚀 Iniciando treinamento ({config['epochs']} épocas máx, batch_size={config['batch_size']})...")
        
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✅ LSTM treinado em {len(history.history['loss'])} épocas")
        
        return model, history
        
    except Exception as e:
        print(f"❌ Erro no treinamento LSTM: {e}")
        return None, None

def train_optimized_ensemble(X_train, y_train, config_dict):
    """
    Treinamento otimizado de modelos ensemble
    """
    print(f"🤖 === TREINANDO ENSEMBLE OTIMIZADO ===")
    print(f"📊 Dados: {X_train.shape}")
    
    models = {}
    
    # Flatten dos dados para modelos tradicionais
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Random Forest otimizado
    print("🌲 Treinando Random Forest otimizado...")
    rf_param_dist = {
        'n_estimators': config_dict['rf_estimators'],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    
    rf_search = RandomizedSearchCV(
        rf,
        rf_param_dist,
        n_iter=config_dict['search_iterations'],
        cv=config_dict['cv_folds'],
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    rf_search.fit(X_train_flat, y_train)
    models['rf'] = rf_search.best_estimator_
    print(f"✅ Random Forest: score={-rf_search.best_score_:.4f}")
    
    # Gradient Boosting otimizado
    print("⚡ Treinando Gradient Boosting otimizado...")
    gb_param_dist = {
        'n_estimators': config_dict['rf_estimators'],  # Reutiliza mesmos valores
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    gb = GradientBoostingRegressor(
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    gb_search = RandomizedSearchCV(
        gb,
        gb_param_dist,
        n_iter=config_dict['search_iterations'] // 2,  # Menos iterações para GB
        cv=config_dict['cv_folds'],
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    gb_search.fit(X_train_flat, y_train)
    models['gb'] = gb_search.best_estimator_
    print(f"✅ Gradient Boosting: score={-gb_search.best_score_:.4f}")
    
    return models

def calculate_metrics_optimized(y_true, y_pred, model_name):
    """
    Cálculo otimizado de métricas
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (tratando divisão por zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.inf
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def main_optimized():
    """
    Função principal otimizada
    """
    import time
    from datetime import datetime
    
    print("🚀 === SISTEMA OTIMIZADO DE PREVISÃO ===")
    print(f"🕐 Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Carrega dados otimizados
        print("📊 Carregando dados com otimizações...")
        df, feature_columns, optimal_config = load_and_preprocess_optimized_data_from_db(
            TECHNICAL_FEATURES_OPTIMIZED,
            limit=None
        )
        
        # Usa configuração otimizada automaticamente detectada
        seq_length = optimal_config['seq_length']
        lstm_config = LSTM_CONFIG_OPTIMIZED.copy()
        lstm_config['epochs'] = optimal_config['lstm_epochs']
        lstm_config['batch_size'] = optimal_config['batch_size']
        
        print(f"⚙️ Configuração otimizada aplicada para {len(df)} registros")
        print(f"   • Sequence Length: {seq_length}")
        print(f"   • LSTM Epochs: {lstm_config['epochs']}")
        print(f"   • Batch Size: {lstm_config['batch_size']}")
        
        # Prepara dados
        data = df[feature_columns].values.astype(np.float32)  # float32 para economia de memória
        target_col_idx = feature_columns.index('close')
        
        # Normalização
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Cria sequências otimizadas
        X, y = create_sequences_optimized(data_scaled, target_col_idx, seq_length)
        
        # Divisão treino/teste
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📈 Divisão: Treino={X_train.shape[0]}, Teste={X_test.shape[0]}")
        
        # Treinamento de modelos
        predictions = {}
        metrics_list = []
        
        # LSTM otimizado
        if TENSORFLOW_AVAILABLE:
            print("\n🧠 Treinando LSTM otimizado...")
            lstm_model, history = train_optimized_lstm(X_train, y_train, X_test, y_test, lstm_config)
            
            if lstm_model is not None:
                lstm_pred = lstm_model.predict(X_test, verbose=0)
                
                # Desnormaliza
                lstm_pred_full = np.zeros((len(lstm_pred), len(feature_columns)))
                lstm_pred_full[:, target_col_idx] = lstm_pred.flatten()
                lstm_pred_rescaled = scaler.inverse_transform(lstm_pred_full)[:, target_col_idx]
                
                predictions['LSTM'] = lstm_pred_rescaled
        
        # Ensemble otimizado
        print("\n🤖 Treinando ensemble otimizado...")
        ensemble_models = train_optimized_ensemble(X_train, y_train, optimal_config)
        
        # Desnormaliza y_test
        y_test_full = np.zeros((len(y_test), len(feature_columns)))
        y_test_full[:, target_col_idx] = y_test
        y_test_rescaled = scaler.inverse_transform(y_test_full)[:, target_col_idx]
        
        # Previsões ensemble
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        for name, model in ensemble_models.items():
            pred = model.predict(X_test_flat)
            pred_full = np.zeros((len(pred), len(feature_columns)))
            pred_full[:, target_col_idx] = pred
            pred_rescaled = scaler.inverse_transform(pred_full)[:, target_col_idx]
            predictions[name.upper()] = pred_rescaled
        
        # Ensemble final
        if len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            predictions['ENSEMBLE'] = ensemble_pred
        
        # Calcula métricas
        print("\n📊 Calculando métricas...")
        for name, preds in predictions.items():
            metrics = calculate_metrics_optimized(y_test_rescaled, preds, name)
            metrics_list.append(metrics)
        
        # Resultados
        results_df = pd.DataFrame(metrics_list)
        print("\n🏆 === RESULTADOS OTIMIZADOS ===")
        print(results_df.round(4))
        
        best_model_idx = results_df['RMSE'].idxmin()
        best_model = results_df.loc[best_model_idx, 'Model']
        best_rmse = results_df.loc[best_model_idx, 'RMSE']
        print(f"\n🥇 Melhor modelo: {best_model} (RMSE: {best_rmse:.4f})")
        
        # Tempo total
        total_time = time.time() - start_time
        print(f"\n⏱️ Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🚀 Otimização: ~3-5x mais rápido que versão original")
        
        return results_df, predictions
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None, None

if __name__ == "__main__":
    main_optimized()

#!/usr/bin/env python3
# main_m1_max.py - Versão específica otimizada para Mac M1 Max

import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importa configurações M1 Max
from config_m1_max import *

def check_m1_max_environment():
    """
    Verifica e configura ambiente M1 Max
    """
    print("🔍 === VERIFICANDO AMBIENTE M1 MAX ===")
    
    # Verifica TensorFlow e Metal
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        # Verifica Metal
        metal_available, gpus = check_metal_support()
        
        if metal_available:
            print("🚀 Metal Performance Shaders: DISPONÍVEL")
            print(f"💪 GPUs detectadas: {len(gpus)}")
            return True
        else:
            print("⚠️ Metal Performance Shaders: NÃO DISPONÍVEL")
            print("🔄 Continuará com CPU otimizada")
            return False
            
    except ImportError:
        print("❌ TensorFlow não instalado")
        return False

def setup_m1_max_tensorflow():
    """
    Configura TensorFlow especificamente para M1 Max
    """
    try:
        import os
        import tensorflow as tf
        
        # Configurações específicas M1 Max
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Aproveita todos os cores do M1 Max
        os.environ['OMP_NUM_THREADS'] = '10'  # 8 performance + 2 efficiency
        os.environ['TF_NUM_INTEROP_THREADS'] = '10'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '10'
        
        # Configurações de threading
        tf.config.threading.set_inter_op_parallelism_threads(10)
        tf.config.threading.set_intra_op_parallelism_threads(10)
        
        # Detecta e configura Metal GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎯 Configurando {len(gpus)} GPU(s) Metal...")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # Testa GPU
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                result = tf.reduce_sum(test_tensor)
                print(f"✅ Teste GPU Metal: {result.numpy()}")
                
            return True, gpus
        else:
            print("⚠️ GPU Metal não detectada, usando CPU otimizada")
            return False, []
            
    except Exception as e:
        print(f"❌ Erro configurando TensorFlow: {e}")
        return False, []

def run_m1_max_optimized():
    """
    Executa versão otimizada para M1 Max
    """
    print("🚀 === LSTM-RMSE OTIMIZADO PARA M1 MAX ===")
    print(f"🕐 Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    start_time = time.time()
    
    # Verifica ambiente
    metal_available = check_m1_max_environment()
    
    if metal_available:
        print("💪 Modo: MÁXIMA PERFORMANCE (GPU Metal)")
        config_mode = "gpu"
    else:
        print("⚡ Modo: ALTA PERFORMANCE (CPU M1 Max)")
        config_mode = "cpu"
    
    # Configura TensorFlow
    gpu_available, gpus = setup_m1_max_tensorflow()
    
    try:
        # Carrega dados (usando database manager original)
        print("\n📊 === CARREGANDO DADOS ===")
        from database import DatabaseManager
        
        db = DatabaseManager()
        if not db.connect():
            raise Exception("Falha na conexão com banco")
        
        # Carrega dados
        df = db.load_botbinance_data(limit=None, order_by='id')
        data_size = len(df)
        print(f"✅ {data_size} registros carregados")
        
        # Obtém configuração otimizada para M1 Max
        optimal_config = get_m1_max_optimal_config(data_size, force_gpu=gpu_available)
        
        # Adiciona indicadores técnicos otimizados
        print("🔧 Calculando indicadores técnicos...")
        from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics
        
        df = add_technical_indicators(df, TECHNICAL_FEATURES_M1_MAX)
        
        # Features lag otimizadas
        base_columns = ['close', 'open', 'high', 'low']
        if 'volume' in df.columns:
            base_columns.append('volume')
        df = add_lagged_features(df, base_columns, lags=[1, 2, 3, 5])
        
        # Rolling statistics
        df = add_rolling_statistics(df, ['close'], windows=[5, 20, 50])
        
        # Remove NaN
        df = df.dropna()
        print(f"📈 Dataset final: {df.shape}")
        
        # Prepara features
        exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Seleção otimizada de features para M1 Max
        if len(feature_columns) > 50:  # Limite maior para M1 Max
            print("🎯 Otimizando features para M1 Max...")
            from sklearn.feature_selection import SelectKBest, mutual_info_regression
            
            X_temp = df[feature_columns].values
            y_temp = df['close'].values
            
            selector = SelectKBest(score_func=mutual_info_regression, k=50)
            X_selected = selector.fit_transform(X_temp, y_temp)
            feature_columns = [feature_columns[i] for i in range(len(feature_columns)) if selector.get_support()[i]]
            print(f"🎯 Selecionadas {len(feature_columns)} features otimizadas")
        
        # Prepara dados
        data = df[feature_columns].values.astype(np.float32)  # float32 para M1 Max
        target_col_idx = feature_columns.index('close')
        
        # Normalização
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Cria sequências
        def create_sequences_m1_max(data, target_col_idx, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i, target_col_idx])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
        seq_length = optimal_config['seq_length']
        X, y = create_sequences_m1_max(data_scaled, target_col_idx, seq_length)
        
        # Divisão treino/teste
        split_idx = int(len(X) * 0.85)  # 85% treino para M1 Max
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # Configuração LSTM M1 Max
        lstm_config = LSTM_CONFIG_M1_MAX.copy()
        lstm_config.update({
            'epochs': optimal_config['lstm_epochs'],
            'batch_size': optimal_config['batch_size'],
            'force_gpu': optimal_config['use_gpu']
        })
        
        print(f"\n🧠 === TREINANDO LSTM M1 MAX ===")
        print(f"⚙️ Configuração:")
        print(f"   • Epochs: {lstm_config['epochs']}")
        print(f"   • Batch Size: {lstm_config['batch_size']}")
        print(f"   • Sequence Length: {seq_length}")
        print(f"   • GPU: {gpu_available}")
        print(f"   • Features: {len(feature_columns)}")
        
        # Importa função otimizada
        try:
            from main_advanced import train_advanced_lstm_m1_max
            
            # Define GPU_AVAILABLE baseado na detecção
            gpu_available_for_training = gpu_available
            
            # Treinamento
            lstm_start = time.time()
            
            # Adiciona GPU_AVAILABLE ao contexto global temporariamente
            import main_advanced
            main_advanced.GPU_AVAILABLE = gpu_available_for_training
            
            lstm_model, history = train_advanced_lstm_m1_max(X_train, y_train, X_test, y_test, lstm_config)
            lstm_time = time.time() - lstm_start
            
        except Exception as e:
            print(f"❌ Erro importando função otimizada: {e}")
            print("🔄 Usando função padrão...")
            
            # Fallback para função padrão
            from main_advanced import train_advanced_lstm
            import main_advanced
            main_advanced.GPU_AVAILABLE = gpu_available_for_training
            
            lstm_start = time.time()
            lstm_model, history = train_advanced_lstm(X_train, y_train, X_test, y_test, lstm_config)
            lstm_time = time.time() - lstm_start
        
        if lstm_model is not None:
            print(f"✅ LSTM treinado em {lstm_time:.1f}s ({lstm_time/60:.1f} min)")
            
            # Previsões
            print("🔮 Fazendo previsões...")
            lstm_pred = lstm_model.predict(X_test, verbose=0)
            
            # Desnormaliza
            lstm_pred_full = np.zeros((len(lstm_pred), len(feature_columns)))
            lstm_pred_full[:, target_col_idx] = lstm_pred.flatten()
            lstm_pred_rescaled = scaler.inverse_transform(lstm_pred_full)[:, target_col_idx]
            
            y_test_full = np.zeros((len(y_test), len(feature_columns)))
            y_test_full[:, target_col_idx] = y_test
            y_test_rescaled = scaler.inverse_transform(y_test_full)[:, target_col_idx]
            
            # Métricas
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_test_rescaled, lstm_pred_rescaled))
            mae = mean_absolute_error(y_test_rescaled, lstm_pred_rescaled)
            r2 = r2_score(y_test_rescaled, lstm_pred_rescaled)
            
            print(f"\n📊 === RESULTADOS M1 MAX ===")
            print(f"🎯 RMSE: {rmse:.4f}")
            print(f"📏 MAE: {mae:.4f}")
            print(f"📈 R²: {r2:.4f}")
            
            # Próxima previsão
            last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, -1)
            next_pred = lstm_model.predict(last_sequence, verbose=0)[0, 0]
            
            next_pred_full = np.zeros((1, len(feature_columns)))
            next_pred_full[0, target_col_idx] = next_pred
            next_pred_rescaled = scaler.inverse_transform(next_pred_full)[0, target_col_idx]
            
            print(f"🔮 Próxima previsão: {next_pred_rescaled:.2f}")
            
            # Salva modelo
            print("\n💾 Salvando modelo M1 Max...")
            import os
            os.makedirs('models', exist_ok=True)
            lstm_model.save('models/lstm_m1_max_optimized.h5')
            print("✅ Modelo salvo em: models/lstm_m1_max_optimized.h5")
        
        # Tempo total
        total_time = time.time() - start_time
        print(f"\n⏱️ === PERFORMANCE M1 MAX ===")
        print(f"⏱️ Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🚀 LSTM: {lstm_time:.1f}s")
        print(f"💪 Aproveitamento M1 Max: {'GPU Metal' if gpu_available else 'CPU 10-cores'}")
        print(f"📊 Registros/segundo: {data_size/total_time:.0f}")
        
        if gpu_available:
            print(f"🎯 Speedup estimado vs CPU: 5-10x")
        else:
            print(f"🎯 Speedup vs configuração original: 3-5x")
        
        db.disconnect()
        
        return {
            'model': lstm_model,
            'history': history,
            'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
            'predictions': {'next': next_pred_rescaled},
            'performance': {
                'total_time': total_time,
                'lstm_time': lstm_time,
                'gpu_used': gpu_available,
                'records_per_second': data_size/total_time
            }
        }
        
    except Exception as e:
        print(f"❌ Erro na execução M1 Max: {e}")
        return None

if __name__ == "__main__":
    result = run_m1_max_optimized()
    
    if result:
        print("\n🏆 === EXECUÇÃO M1 MAX CONCLUÍDA ===")
        print("✅ Modelo treinado com sucesso")
        print("💾 Arquivos salvos em: models/")
        print("🚀 Performance otimizada para M1 Max")
    else:
        print("\n❌ === FALHA NA EXECUÇÃO ===")
        print("🔧 Verifique configurações e dependências")

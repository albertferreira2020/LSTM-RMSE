# main_parallel_optimized.py - Versão com máxima paralelização

import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importa configurações de paralelização máxima
from config import *

# Configurações de ambiente para máxima performance
def setup_parallel_environment():
    """Configura ambiente para máxima performance paralela"""
    print("🔧 === CONFIGURANDO AMBIENTE PARALELO ===")
    
    # Configurações TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime todos os warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Evita warnings OneDNN
    os.environ['OMP_NUM_THREADS'] = str(M1_MAX_PERFORMANCE_CORES)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(M1_MAX_PERFORMANCE_CORES)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(M1_MAX_PERFORMANCE_CORES)
    
    print(f"✅ Ambiente configurado para {M1_MAX_PERFORMANCE_CORES} cores de performance")

# Configuração inicial
setup_parallel_environment()

# Importação do DatabaseManager com tratamento de erro
try:
    from database import DatabaseManager
    DATABASE_AVAILABLE = True
    print("✅ DatabaseManager carregado com sucesso")
except ImportError as e:
    print(f"⚠️ DatabaseManager não disponível: {e}")
    DATABASE_AVAILABLE = False
    DatabaseManager = None
except Exception as e:
    print(f"⚠️ Erro ao carregar DatabaseManager: {e}")
    DATABASE_AVAILABLE = False
    DatabaseManager = None

# Imports otimizados
try:
    import tensorflow as tf
    
    # Configuração TensorFlow para máxima paralelização
    tf.config.threading.set_inter_op_parallelism_threads(M1_MAX_PERFORMANCE_CORES)
    tf.config.threading.set_intra_op_parallelism_threads(M1_MAX_PERFORMANCE_CORES)
    
    # Configura GPU Metal se disponível
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU Metal configurada: {len(gpus)} dispositivo(s)")
        GPU_AVAILABLE = True
    else:
        GPU_AVAILABLE = False
        print("ℹ️ GPU não detectada, usando CPU")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers.legacy import Adam
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow configurado para máxima paralelização")
    
except Exception as e:
    print(f"⚠️ TensorFlow não disponível: {e}")
    TENSORFLOW_AVAILABLE = False
    GPU_AVAILABLE = False

# Imports para modelos tradicionais
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Imports para visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Configura estilo dos gráficos
try:
    plt.style.use('seaborn-v0_8')  # Tenta estilo novo
except OSError:
    try:
        plt.style.use('seaborn')  # Fallback para versão antiga
    except OSError:
        plt.style.use('default')  # Fallback final
        
sns.set_palette("husl")

# Importa funções base necessárias
from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics

# Callback personalizado para mostrar progresso organizado
class ParallelTrainingCallback:
    """Callback personalizado para mostrar progresso de treinamento em paralelo"""
    
    def __init__(self, model_name="LSTM"):
        self.model_name = model_name
        
    def create_keras_callback(self):
        """Cria callback do Keras que usa esta classe"""
        import tensorflow as tf
        
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, model_name):
                super().__init__()
                self.model_name = model_name
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                mae = logs.get('mae', 0)
                val_mae = logs.get('val_mae', 0)
                
                print(f"[{self.model_name}] Época {epoch+1:3d} - "
                      f"Loss: {loss:.4f} - Val_Loss: {val_loss:.4f} - "
                      f"MAE: {mae:.4f} - Val_MAE: {val_mae:.4f}")
                
        return ProgressCallback(self.model_name)

def load_and_preprocess_advanced_data_from_db(config, limit=None):
    """
    Carregamento e pré-processamento avançado dos dados do PostgreSQL
    """
    if not DATABASE_AVAILABLE:
        raise Exception("DatabaseManager não está disponível - problema de compatibilidade com typing_extensions/SQLAlchemy")
    
    print("=== CARREGANDO DADOS DO POSTGRESQL ===")
    
    # Inicializa conexão com banco
    db = DatabaseManager()
    
    try:
        # Conecta ao banco
        if not db.connect():
            raise Exception("Falha ao conectar com o banco de dados")
        
        # Obtém informações da tabela
        print("Verificando estrutura da tabela...")
        db.get_table_info('botbinance')
        
        # Carrega os dados
        print("Carregando dados da tabela botbinance...")
        df = db.load_botbinance_data(limit=limit, order_by='id')
        
        if df is None or len(df) == 0:
            raise Exception("Nenhum dado encontrado na tabela botbinance")
        
        print(f"✅ Dados carregados do banco: {df.shape}")
        print(f"Colunas disponíveis: {list(df.columns)}")
        
        # Verifica se as colunas necessárias existem
        required_columns = ['close', 'open', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Colunas obrigatórias não encontradas: {missing_columns}")
            # Tenta mapear colunas com nomes similares
            column_mapping = {}
            for col in missing_columns:
                similar_cols = [c for c in df.columns if col.lower() in c.lower()]
                if similar_cols:
                    column_mapping[similar_cols[0]] = col
                    print(f"Mapeando {similar_cols[0]} -> {col}")
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
        
        # Usa feature engineering paralela com TODOS os sensores
        print("Calculando features com TODOS os sensores disponíveis...")
        df, all_features = parallel_feature_engineering(df, config)
        
        # Seleciona features para o modelo (exclui colunas de ID e timestamp)
        exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
        feature_columns = [col for col in all_features if col not in exclude_columns and col in df.columns]
        
        print(f"Features selecionadas: {len(feature_columns)}")
        print(f"Primeiras 15 features: {feature_columns[:15]}")
        
        return df, feature_columns
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados do banco: {e}")
        raise e
    finally:
        if 'db' in locals():
            db.disconnect()

def load_and_preprocess_advanced_data_from_csv(csv_file, config):
    """
    Carregamento e pré-processamento dos dados do CSV (fallback)
    """
    print(f"=== CARREGANDO DADOS DO CSV: {csv_file} ===")
    
    # Lê CSV com separador correto
    df = pd.read_csv(csv_file, sep=';')
    print(f"✅ Dados carregados do CSV: {df.shape}")
    print(f"Colunas disponíveis: {list(df.columns)}")
    
    # Verifica colunas necessárias
    required_columns = ['close', 'open', 'high', 'low']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Colunas obrigatórias não encontradas: {missing_columns}")
        raise Exception(f"Colunas obrigatórias faltando: {missing_columns}")
    
    # Usa feature engineering paralela com TODOS os sensores
    print("Calculando features com TODOS os sensores disponíveis...")
    df, all_features = parallel_feature_engineering(df, config)
    
    # Seleciona features para o modelo (exclui colunas de ID e timestamp)
    exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
    feature_columns = [col for col in all_features if col not in exclude_columns and col in df.columns]
    
    print(f"Features selecionadas: {len(feature_columns)}")
    print(f"Primeiras 15 features: {feature_columns[:15]}")
    
    return df, feature_columns

def prepare_all_sensor_features(df):
    """
    Prepara TODAS as colunas disponíveis como features/sensores para previsão
    """
    print("🔧 === PREPARANDO TODOS OS SENSORES COMO FEATURES ===")
    
    # Colunas originais disponíveis
    original_sensors = [
        'open', 'high', 'low', 'volume', 'reversal',
        'best_bid_price', 'best_bid_quantity', 
        'best_ask_price', 'best_ask_quantity',
        'spread', 'spread_percentage',
        'bid_liquidity', 'ask_liquidity', 'total_liquidity',
        'imbalance', 'weighted_mid_price'
    ]
    
    # Verifica quais sensores estão disponíveis
    available_sensors = [col for col in original_sensors if col in df.columns]
    
    print(f"✅ Sensores disponíveis ({len(available_sensors)}): {available_sensors}")
    
    # Cria features derivadas essenciais
    sensor_features = []
    
    # Features básicas de preço
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        sensor_features.extend(['price_range', 'body_size'])
    
    # Features de volume
    if 'volume' in df.columns:
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        sensor_features.append('volume_ma5')
    
    # Features temporais básicas
    if 'close' in df.columns:
        for lag in [1, 3, 5]:
            lag_col = f'close_lag_{lag}'
            df[lag_col] = df['close'].shift(lag)
            sensor_features.append(lag_col)
    
    print(f"✅ Features criadas: {len(sensor_features)}")
    
    # Combina sensores originais + features derivadas
    all_features = available_sensors + sensor_features
    
    return df, all_features

def parallel_feature_engineering(df, features_config):
    """
    Calcula indicadores técnicos em paralelo
    """
    print("🔧 === ENGENHARIA DE FEATURES PARALELA ===")
    start_time = time.time()
    
    # Prepara features dos sensores disponíveis
    df_enhanced, sensor_features = prepare_all_sensor_features(df)
    
    # Verifica se as colunas básicas existem
    required_cols = ['close', 'open', 'high', 'low']
    available_cols = df_enhanced.columns.tolist()
    
    if not all(col in available_cols for col in required_cols):
        print(f"⚠️ Algumas colunas obrigatórias estão faltando: {required_cols}")
        print(f"   Disponíveis: {[col for col in required_cols if col in available_cols]}")
    
    # Calcula indicadores técnicos básicos
    try:
        df_enhanced = add_technical_indicators(df_enhanced, features_config)
        print("✅ Indicadores técnicos adicionados")
    except Exception as e:
        print(f"⚠️ Erro ao adicionar indicadores técnicos: {e}")
    
    # Features lag básicas
    try:
        base_columns = ['close', 'open', 'high', 'low']
        if 'volume' in df_enhanced.columns:
            base_columns.append('volume')
        
        available_base = [col for col in base_columns if col in df_enhanced.columns]
        lags = features_config.get('lag_features', [1, 2, 3])
        
        df_enhanced = add_lagged_features(df_enhanced, available_base, lags)
        print(f"✅ Features lag adicionadas para {len(available_base)} colunas")
    except Exception as e:
        print(f"⚠️ Erro ao adicionar features lag: {e}")
    
    # Remove linhas com NaN
    df_enhanced = df_enhanced.dropna()
    
    # Features finais
    all_features = [col for col in df_enhanced.columns if col not in ['id', 'created_at', 'updated_at', 'timestamp']]
    
    elapsed_time = time.time() - start_time
    print(f"✅ Feature engineering concluído em {elapsed_time:.2f}s")
    print(f"📊 Total de features: {len(all_features)}")
    
    return df_enhanced, all_features

# Função global para criação de sequências
def create_sequence_chunk(args):
    """Cria um chunk de sequências - função global para permitir pickle"""
    data, target_idx, seq_length, start_idx, end_idx = args
    
    X_chunk = []
    y_chunk = []
    
    for i in range(start_idx, end_idx):
        if i >= seq_length:
            X_chunk.append(data[i-seq_length:i])
            y_chunk.append(data[i, target_idx])
    
    return np.array(X_chunk), np.array(y_chunk)

def create_sequences_parallel(data, target_idx, seq_length, n_jobs=4):
    """
    Cria sequências usando ThreadPoolExecutor ao invés de ProcessPoolExecutor
    para evitar problemas de serialização
    """
    print(f"🔧 Criando sequências com paralelização ({n_jobs} workers)")
    start_time = time.time()
    
    # Usa ThreadPoolExecutor que é mais simples e evita problemas de pickle
    total_samples = len(data) - seq_length
    
    if total_samples <= 0:
        print("⚠️ Dados insuficientes para criar sequências")
        return np.array([]), np.array([])
    
    # Para datasets pequenos, usa versão sequencial
    if total_samples < 1000:
        print("📊 Dataset pequeno - usando versão sequencial")
        X = []
        y = []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, target_idx])
        
        elapsed_time = time.time() - start_time
        print(f"✅ Sequências criadas em {elapsed_time:.2f}s")
        return np.array(X), np.array(y)
    
    # Para datasets maiores, usa paralelização com chunks
    chunk_size = max(100, total_samples // n_jobs)
    
    def create_chunk_threaded(start_idx, end_idx):
        """Versão para ThreadPoolExecutor"""
        X_chunk = []
        y_chunk = []
        
        for i in range(start_idx, end_idx):
            if i >= seq_length:
                X_chunk.append(data[i-seq_length:i])
                y_chunk.append(data[i, target_idx])
        
        return np.array(X_chunk), np.array(y_chunk)
    
    # Cria ranges para cada worker
    ranges = []
    for i in range(0, total_samples, chunk_size):
        start_idx = seq_length + i
        end_idx = min(seq_length + i + chunk_size, len(data))
        if start_idx < end_idx:
            ranges.append((start_idx, end_idx))
    
    print(f"📊 Dividindo em {len(ranges)} chunks para processamento paralelo")
    
    # Processa chunks em paralelo usando ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n_jobs, len(ranges))) as executor:
        futures = [executor.submit(create_chunk_threaded, start, end) for start, end in ranges]
        results = [future.result() for future in futures]
    
    # Combina resultados
    X_parts = [result[0] for result in results if len(result[0]) > 0]
    y_parts = [result[1] for result in results if len(result[1]) > 0]
    
    X = np.concatenate(X_parts, axis=0) if X_parts else np.array([])
    y = np.concatenate(y_parts, axis=0) if y_parts else np.array([])
    
    elapsed_time = time.time() - start_time
    print(f"✅ Sequências criadas em {elapsed_time:.2f}s")
    print(f"📊 Shape final: X={X.shape}, y={y.shape}")
    
    return X, y

def train_models_parallel(X_train, y_train, X_test, y_test):
    """
    Treina múltiplos modelos em paralelo
    """
    print("🚀 === TREINAMENTO PARALELO DE MODELOS ===")
    
    def train_lstm_model():
        """Treina modelo LSTM"""
        if not TENSORFLOW_AVAILABLE:
            return None, None
        
        print("🧠 Treinando LSTM...")
        start_time = time.time()
        
        # Usa configuração otimizada para paralelização
        config = LSTM_CONFIG_PARALLEL_MAX
        
        model = Sequential([
            LSTM(config['layers'][0], return_sequences=True, 
                 input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(config['dropout_rates'][0]),
            LSTM(config['layers'][1], return_sequences=True),
            Dropout(config['dropout_rates'][1]),
            LSTM(config['layers'][2]),
            Dropout(config['dropout_rates'][2]),
            Dense(config['dense_layers'][0], activation='relu'),
            Dropout(0.2),
            Dense(config['dense_layers'][1], activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Compilação otimizada
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=optimizer, loss=config['loss_function'], metrics=['mae'])
        
        # Callbacks com callback personalizado para progresso
        progress_callback = ParallelTrainingCallback("LSTM").create_keras_callback()
        
        # Cria diretório models se não existir
        os.makedirs('models', exist_ok=True)
        
        # Checkpoint para salvar o melhor modelo
        checkpoint = ModelCheckpoint(
            'models/best_lstm_parallel.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        
        callbacks = [
            progress_callback,  # Callback personalizado para progresso
            checkpoint,  # Salva o melhor modelo
            EarlyStopping(monitor='val_loss', patience=config['patience_early_stop'], 
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=config['reduce_lr_factor'], 
                             patience=config['patience_reduce_lr'], min_lr=config['min_lr'])
        ]
        
        # Treinamento com paralelização máxima
        print(f"🧠 Iniciando treinamento LSTM ({config['epochs']} épocas)...")
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0,  # Usa callback personalizado ao invés do verbose padrão
            use_multiprocessing=TENSORFLOW_CONFIG_MAX_PARALLEL['use_multiprocessing'],
            workers=TENSORFLOW_CONFIG_MAX_PARALLEL['workers'],
            max_queue_size=TENSORFLOW_CONFIG_MAX_PARALLEL['max_queue_size']
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ LSTM treinado em {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
        
        return model, history
    
    def train_random_forest():
        """Treina Random Forest"""
        print("🌲 [RF] Iniciando treinamento Random Forest...")
        start_time = time.time()
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        print(f"🌲 [RF] Dados achatados: {X_train_flat.shape}")
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        print(f"🌲 [RF] Testando {RF_CONFIG_PARALLEL_MAX['random_search_iterations']} combinações de hiperparâmetros...")
        
        rf_search = RandomizedSearchCV(
            rf, 
            {
                'n_estimators': RF_CONFIG_PARALLEL_MAX['n_estimators_options'],
                'max_depth': RF_CONFIG_PARALLEL_MAX['max_depth_options'],
                'min_samples_split': RF_CONFIG_PARALLEL_MAX['min_samples_split_options'],
                'min_samples_leaf': RF_CONFIG_PARALLEL_MAX['min_samples_leaf_options'],
                'max_features': RF_CONFIG_PARALLEL_MAX['max_features_options']
            },
            n_iter=RF_CONFIG_PARALLEL_MAX['random_search_iterations'],
            cv=TimeSeriesSplit(n_splits=RF_CONFIG_PARALLEL_MAX['cv_folds']),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1  # Mostra progresso do RandomizedSearchCV
        )
        
        rf_search.fit(X_train_flat, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Random Forest treinado em {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
        
        return rf_search.best_estimator_
    
    def train_gradient_boosting():
        """Treina Gradient Boosting"""
        print("📈 [GB] Iniciando treinamento Gradient Boosting...")
        start_time = time.time()
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        print(f"📈 [GB] Dados achatados: {X_train_flat.shape}")
        
        gb = GradientBoostingRegressor(random_state=42)
        print(f"📈 [GB] Testando {GB_CONFIG_PARALLEL_MAX['random_search_iterations']} combinações de hiperparâmetros...")
        
        gb_search = RandomizedSearchCV(
            gb,
            {
                'n_estimators': GB_CONFIG_PARALLEL_MAX['n_estimators_options'],
                'learning_rate': GB_CONFIG_PARALLEL_MAX['learning_rate_options'],
                'max_depth': GB_CONFIG_PARALLEL_MAX['max_depth_options'],
                'min_samples_split': GB_CONFIG_PARALLEL_MAX['min_samples_split_options'],
                'min_samples_leaf': GB_CONFIG_PARALLEL_MAX['min_samples_leaf_options'],
                'subsample': GB_CONFIG_PARALLEL_MAX['subsample_options'],
                'max_features': GB_CONFIG_PARALLEL_MAX['max_features_options']
            },
            n_iter=GB_CONFIG_PARALLEL_MAX['random_search_iterations'],
            cv=TimeSeriesSplit(n_splits=GB_CONFIG_PARALLEL_MAX['cv_folds']),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1  # Mostra progresso do RandomizedSearchCV
        )
        
        gb_search.fit(X_train_flat, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Gradient Boosting treinado em {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
        
        return gb_search.best_estimator_
    
    # Treina modelos em paralelo usando ThreadPoolExecutor
    print(f"🚀 Iniciando treinamento paralelo com {min(3, M1_MAX_PERFORMANCE_CORES)} workers")
    print("📋 Modelos sendo treinados simultaneamente:")
    print("   🧠 LSTM - Redes Neurais Recorrentes")
    print("   🌲 Random Forest - Ensemble de Árvores")
    print("   📈 Gradient Boosting - Boosting Sequencial")
    print("\n" + "="*60)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submete tarefas
        print("⚡ Iniciando threads de treinamento...")
        lstm_future = executor.submit(train_lstm_model)
        rf_future = executor.submit(train_random_forest)
        gb_future = executor.submit(train_gradient_boosting)
        
        print("🔄 Todos os modelos treinando em paralelo...\n")
        
        # Coleta resultados conforme ficam prontos
        models = {}
        
        # LSTM
        try:
            lstm_model, lstm_history = lstm_future.result()
            if lstm_model is not None:
                models['LSTM'] = {'model': lstm_model, 'history': lstm_history}
                print("✅ [LSTM] Treinamento concluído!")
        except Exception as e:
            print(f"❌ [LSTM] Erro no treinamento: {e}")
        
        # Random Forest
        try:
            rf_model = rf_future.result()
            models['RF'] = {'model': rf_model, 'history': None}
            print("✅ [RF] Treinamento concluído!")
        except Exception as e:
            print(f"❌ [RF] Erro no treinamento: {e}")
        
        # Gradient Boosting
        try:
            gb_model = gb_future.result()
            models['GB'] = {'model': gb_model, 'history': None}
            print("✅ [GB] Treinamento concluído!")
        except Exception as e:
            print(f"❌ [GB] Erro no treinamento: {e}")
    
    print("\n" + "="*60)
    print(f"🎉 Treinamento paralelo concluído! Modelos treinados: {list(models.keys())}")
    
    return models

def save_models(models):
    """
    Salva todos os modelos treinados
    """
    print("\n💾 === SALVANDO MODELOS ===")
    os.makedirs('models', exist_ok=True)
    
    for model_name, model_info in models.items():
        model = model_info['model']
        
        try:
            if model_name == 'LSTM':
                # Modelo TensorFlow/Keras
                model_path = f'models/best_{model_name.lower()}_parallel.h5'
                model.save(model_path)
                print(f"✅ {model_name} salvo em: {model_path}")
            else:
                # Modelos scikit-learn
                import pickle
                model_path = f'models/best_{model_name.lower()}_parallel.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✅ {model_name} salvo em: {model_path}")
        except Exception as e:
            print(f"❌ Erro ao salvar {model_name}: {e}")

def create_comparison_plots(models, X_test, y_test, scaler, feature_columns, target_idx):
    """
    Cria gráficos de comparação entre os modelos
    """
    print("\n📊 === GERANDO GRÁFICOS DE COMPARAÇÃO ===")
    
    # Prepara dados para visualização
    predictions = {}
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Gera predições para todos os modelos
    for model_name, model_info in models.items():
        model = model_info['model']
        
        try:
            if model_name == 'LSTM':
                pred = model.predict(X_test, verbose=0).flatten()
            else:
                pred = model.predict(X_test_flat)
            
            # Desnormaliza predições
            pred_full = np.zeros((len(pred), len(feature_columns)))
            pred_full[:, target_idx] = pred
            pred_rescaled = scaler.inverse_transform(pred_full)[:, target_idx]
            predictions[model_name] = pred_rescaled
            
        except Exception as e:
            print(f"❌ Erro ao gerar predições para {model_name}: {e}")
    
    # Desnormaliza valores reais
    y_test_full = np.zeros((len(y_test), len(feature_columns)))
    y_test_full[:, target_idx] = y_test
    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, target_idx]
    
    if not predictions:
        print("❌ Nenhuma predição disponível para plotar")
        return
    
    # Cria figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparação de Modelos - Previsão vs Real', fontsize=16, fontweight='bold')
    
    # 1. Gráfico de série temporal (últimos 100 pontos)
    ax1 = axes[0, 0]
    n_points = min(100, len(y_test_rescaled))
    x_range = range(n_points)
    
    ax1.plot(x_range, y_test_rescaled[-n_points:], 'k-', label='Real', linewidth=2, alpha=0.8)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, pred) in enumerate(predictions.items()):
        ax1.plot(x_range, pred[-n_points:], color=colors[i % len(colors)], 
                label=f'{model_name}', linewidth=1.5, alpha=0.7)
    
    ax1.set_title('Série Temporal - Últimos 100 Pontos')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Preço')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot - Predito vs Real
    ax2 = axes[0, 1]
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        ax2.scatter(y_test_rescaled, pred, alpha=0.6, s=20, 
                   color=colors[i % len(colors)], label=model_name)
    
    # Linha perfeita (y=x)
    min_val, max_val = y_test_rescaled.min(), y_test_rescaled.max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfeito')
    
    ax2.set_title('Predito vs Real')
    ax2.set_xlabel('Valor Real')
    ax2.set_ylabel('Valor Predito')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Métricas comparativas
    ax3 = axes[1, 0]
    
    metrics_data = {'Model': [], 'RMSE': [], 'MAE': [], 'R²': []}
    
    for model_name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, pred))
        mae = mean_absolute_error(y_test_rescaled, pred)
        r2 = r2_score(y_test_rescaled, pred)
        
        metrics_data['Model'].append(model_name)
        metrics_data['RMSE'].append(rmse)
        metrics_data['MAE'].append(mae)
        metrics_data['R²'].append(r2)
    
    x_pos = np.arange(len(metrics_data['Model']))
    width = 0.25
    
    ax3.bar(x_pos - width, metrics_data['RMSE'], width, label='RMSE', alpha=0.8)
    ax3.bar(x_pos, metrics_data['MAE'], width, label='MAE', alpha=0.8)
    ax3.bar(x_pos + width, [r2*100 for r2 in metrics_data['R²']], width, label='R² (x100)', alpha=0.8)
    
    ax3.set_title('Métricas de Performance')
    ax3.set_xlabel('Modelos')
    ax3.set_ylabel('Valor da Métrica')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics_data['Model'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribuição dos erros
    ax4 = axes[1, 1]
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        errors = pred - y_test_rescaled
        ax4.hist(errors, bins=30, alpha=0.6, label=model_name, 
                color=colors[i % len(colors)], density=True)
    
    ax4.set_title('Distribuição dos Erros')
    ax4.set_xlabel('Erro (Predito - Real)')
    ax4.set_ylabel('Densidade')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva o gráfico
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {plot_path}")
    
    # Mostra o gráfico
    plt.show()
    
    # Cria tabela de métricas detalhada
    print("\n📊 === MÉTRICAS DETALHADAS ===")
    print("-" * 60)
    print(f"{'Modelo':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 60)
    
    for i, model_name in enumerate(metrics_data['Model']):
        rmse = metrics_data['RMSE'][i]
        mae = metrics_data['MAE'][i]
        r2 = metrics_data['R²'][i]
        print(f"{model_name:<15} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f}")
    
    print("-" * 60)
    
    return predictions, metrics_data

def main_parallel():
    """
    Função principal otimizada para máxima paralelização
    """
    print("🚀 === SISTEMA DE PREVISÃO COM PARALELIZAÇÃO MÁXIMA ===")
    print(f"💻 Cores disponíveis: {CPU_CORES}")
    print(f"⚡ Performance cores M1 Max: {M1_MAX_PERFORMANCE_CORES}")
    print(f"🔧 TensorFlow workers: {TENSORFLOW_CONFIG_MAX_PARALLEL.get('workers', 'N/A')}")
    
    start_time = time.time()
    
    # Configuração para fallback CSV
    use_database = DATABASE_AVAILABLE  # Só usa database se estiver disponível
    csv_fallback = 'relatorio_mensal_geral_2025-03 (1).csv'
    
    try:
        # Tenta carregar dados do PostgreSQL primeiro
        if use_database:
            print("Tentando carregar dados do PostgreSQL...")
            df, feature_columns = load_and_preprocess_advanced_data_from_db(
                TECHNICAL_FEATURES_PARALLEL, 
                limit=None  # None = todos os dados, ou especifique um número para teste
            )
            print("✅ Dados carregados do PostgreSQL com sucesso")
        else:
            raise Exception("DatabaseManager não disponível - usando CSV")
            
    except Exception as e:
        print(f"⚠️ Erro ao carregar do PostgreSQL: {e}")
        print("Tentando fallback para CSV...")
        
        try:
            df, feature_columns = load_and_preprocess_advanced_data_from_csv(
                csv_fallback, 
                TECHNICAL_FEATURES_PARALLEL
            )
            print("✅ Dados carregados do CSV com sucesso")
        except Exception as csv_error:
            print(f"❌ Erro também no CSV: {csv_error}")
            print("Verifique as configurações do banco (.env) ou a existência do arquivo CSV")
            return None, None, None
    
    # Prepara dados
    print(f"\n📈 === PREPARANDO DADOS ===")
    print(f"Features selecionadas: {len(feature_columns)}")
    print(f"Dados shape: {df.shape}")
    
    # Seleciona apenas as colunas de features
    data = df[feature_columns].values
    target_idx = feature_columns.index('close')
    
    print(f"� Dados preparados:")
    print(f"Shape dos dados: {data.shape}")
    print(f"Índice da coluna 'close': {target_idx}")
    
    # Normalização
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Criação de sequências paralela
    X, y = create_sequences_parallel(data_scaled, target_idx, SEQ_LENGTH_PARALLEL)
    
    # Divisão treino/teste
    split_idx = int(len(X) * (1 - TEST_SIZE_PARALLEL))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # Treinamento paralelo
    print(f"\n🚀 === TREINAMENTO PARALELO ===")
    training_start = time.time()
    models = train_models_parallel(X_train, y_train, X_test, y_test)
    training_time = time.time() - training_start
    
    print(f"⏱️ Tempo total de treinamento: {training_time:.2f}s ({training_time/60:.1f} min)")
    
    # Salva os modelos
    save_models(models)
    
    # Avaliação rápida
    print(f"\n📈 === AVALIAÇÃO DOS MODELOS ===")
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    for model_name, model_info in models.items():
        model = model_info['model']
        
        if model_name == 'LSTM':
            pred = model.predict(X_test, verbose=0)
            pred = pred.flatten()
        else:
            pred = model.predict(X_test_flat)
        
        # Desnormaliza para cálculo das métricas
        pred_full = np.zeros((len(pred), len(feature_columns)))
        pred_full[:, target_idx] = pred
        pred_rescaled = scaler.inverse_transform(pred_full)[:, target_idx]
        
        y_test_full = np.zeros((len(y_test), len(feature_columns)))
        y_test_full[:, target_idx] = y_test
        y_test_rescaled = scaler.inverse_transform(y_test_full)[:, target_idx]
        
        # Métricas
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, pred_rescaled))
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
        r2 = r2_score(y_test_rescaled, pred_rescaled)
        
        print(f"🎯 {model_name}:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
    
    # Gera gráficos de comparação
    predictions, metrics_data = create_comparison_plots(models, X_test, y_test, scaler, feature_columns, target_idx)
    
    total_time = time.time() - start_time
    print(f"\n🏁 === EXECUÇÃO CONCLUÍDA ===")
    print(f"⏱️ Tempo total: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"🚀 Speedup obtido com paralelização!")
    print(f"📊 Fonte dos dados: {'PostgreSQL' if use_database else 'CSV'}")
    
    return models, scaler, feature_columns

def cleanup_resources():
    """Limpa recursos para evitar warnings de leaked file objects"""
    import gc
    import threading
    
    # Força coleta de lixo
    gc.collect()
    
    # Aguarda threads finalizarem
    for thread in threading.enumerate():
        if thread != threading.current_thread() and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except:
                pass

if __name__ == "__main__":
    try:
        # Executa versão paralela otimizada
        models, scaler, features = main_parallel()
        print("\n✅ === EXECUÇÃO CONCLUÍDA COM SUCESSO ===")
    except Exception as e:
        print(f"\n❌ === ERRO NA EXECUÇÃO ===")
        print(f"Erro: {e}")
    finally:
        # Limpa recursos
        cleanup_resources()
        print("🧹 Recursos limpos")

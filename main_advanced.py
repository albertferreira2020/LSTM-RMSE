# main_advanced.py - Versão avançada com conexão PostgreSQL e logs melhorados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurações
from config import *
from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics

# Função utilitária para monitoramento
def show_system_stats():
    """
    Mostra estatísticas básicas do sistema durante processamento
    """
    import os
    import gc
    
    # Força coleta de lixo
    gc.collect()
    
    # Informações básicas
    print(f"   🐍 PID do processo: {os.getpid()}")
    print(f"   📊 Status: Processando...")

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

# Imports do TensorFlow (otimizado para Mac M1 Max com GPU Metal)
try:
    import os
    # Configurações otimizadas para Mac M1 Max
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita oneDNN 
    # Configurações para aproveitar múltiplos cores do M1 Max
    os.environ['OMP_NUM_THREADS'] = '8'  # Aumentado para aproveitar M1 Max (8 performance cores)
    os.environ['TF_NUM_INTEROP_THREADS'] = '8'  # Otimizado para M1 Max
    os.environ['TF_NUM_INTRAOP_THREADS'] = '8'  # Otimizado para M1 Max
    
    import tensorflow as tf
    
    # Configuração otimizada para M1 Max
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    
    # Detecta e configura Metal Performance Shaders (GPU) no Mac M1 Max
    print("🔍 Detectando dispositivos disponíveis...")
    
    # Lista todos os dispositivos físicos
    physical_devices = tf.config.list_physical_devices()
    print(f"📱 Dispositivos físicos encontrados: {len(physical_devices)}")
    for device in physical_devices:
        print(f"   • {device}")
    
    # Configuração específica para GPU Metal (M1 Max)
    try:
        # Verifica se Metal Performance Shaders está disponível
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU Metal detectada: {len(gpus)} dispositivo(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
                # Habilita crescimento dinâmico de memória para evitar erros
                tf.config.experimental.set_memory_growth(gpu, True)
                # Define GPU como preferencial para operações
                tf.config.set_visible_devices(gpu, 'GPU')
            
            # Testa se a GPU está funcionando
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.reduce_sum(test_tensor)
                print(f"✅ Teste GPU bem-sucedido: {result.numpy()}")
                
            print("🎯 TensorFlow configurado para usar GPU Metal (M1 Max)")
            GPU_AVAILABLE = True
            
        else:
            print("⚠️ GPU Metal não detectada, usando CPU otimizada")
            # Otimiza CPU para M1 Max
            tf.config.set_soft_device_placement(True)
            GPU_AVAILABLE = False
            
    except Exception as gpu_error:
        print(f"⚠️ Erro na configuração GPU: {gpu_error}")
        print("🔄 Continuando com CPU otimizada para M1 Max")
        GPU_AVAILABLE = False
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    
    if GPU_AVAILABLE:
        print("✅ TensorFlow carregado com GPU Metal (M1 Max) - MÁXIMA PERFORMANCE")
    else:
        print("✅ TensorFlow carregado com CPU otimizada (M1 Max)")
        
except ImportError as e:
    print(f"TensorFlow não disponível: {e}. Apenas Random Forest será usado.")
    TENSORFLOW_AVAILABLE = False
    GPU_AVAILABLE = False
except ImportError as e:
    print(f"TensorFlow não disponível: {e}. Apenas Random Forest será usado.")
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    print(f"Erro ao configurar TensorFlow: {e}. Tentando modo simplificado...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.optimizers import Adam
        TENSORFLOW_AVAILABLE = True
        print("✅ TensorFlow carregado em modo simplificado")
    except:
        print("❌ TensorFlow completamente indisponível. Apenas Random Forest será usado.")
        TENSORFLOW_AVAILABLE = False

# Imports do sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

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
        
        # Adiciona indicadores técnicos
        print("Calculando indicadores técnicos...")
        df = add_technical_indicators(df, TECHNICAL_FEATURES)
        
        # Adiciona features com lag
        print("Adicionando features com lag...")
        base_columns = ['close', 'open', 'high', 'low']
        if 'volume' in df.columns:
            base_columns.append('volume')
        df = add_lagged_features(df, base_columns, lags=[1, 2, 3, 5])
        
        # Adiciona estatísticas rolantes
        print("Calculando estatísticas rolantes...")
        df = add_rolling_statistics(df, ['close'], windows=[5, 10, 20])
        
        # Remove NaN
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        print(f"Removidas {removed_rows} linhas com NaN")
        print(f"Dataset final: {df.shape}")
        
        # Seleciona features para o modelo (exclui colunas de ID e timestamp)
        exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        print(f"Features selecionadas: {len(feature_columns)}")
        print(f"Primeiras 10 features: {feature_columns[:10]}")
        
        return df, feature_columns
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados do banco: {e}")
        raise
    finally:
        # Sempre fecha a conexão
        db.disconnect()

def load_and_preprocess_advanced_data_from_csv(file_path, config):
    """
    Função original para carregar dados do CSV (mantida como fallback)
    """
    print("=== CARREGANDO DADOS DO CSV (FALLBACK) ===")
    df = pd.read_csv(file_path, delimiter=';')
    
    print(f"Dataset original: {df.shape}")
    print(f"Colunas disponíveis: {list(df.columns)}")
    
    # Adiciona indicadores técnicos
    print("Calculando indicadores técnicos...")
    df = add_technical_indicators(df, TECHNICAL_FEATURES)
    
    # Adiciona features com lag
    print("Adicionando features com lag...")
    base_columns = ['close', 'open', 'high', 'low']
    if 'volume' in df.columns:
        base_columns.append('volume')
    df = add_lagged_features(df, base_columns, lags=[1, 2, 3, 5])
    
    # Adiciona estatísticas rolantes
    print("Calculando estatísticas rolantes...")
    df = add_rolling_statistics(df, ['close'], windows=[5, 10, 20])
    
    # Remove NaN
    df = df.dropna()
    print(f"Dataset após processamento: {df.shape}")
    
    # Seleciona features para o modelo
    feature_columns = [col for col in df.columns if col not in ['id', 'created_at']]
    
    print(f"Features selecionadas: {len(feature_columns)}")
    
    return df, feature_columns

def create_sequences_advanced(data, target_col_idx, seq_length):
    """
    Cria sequências para modelos de séries temporais
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, target_col_idx])
    
    return np.array(X), np.array(y)

def train_advanced_lstm_m1_max(X_train, y_train, X_test, y_test, config):
    """
    Treina modelo LSTM otimizado para Mac M1 Max com GPU Metal
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow não disponível, pulando LSTM...")
        return None, None
    
    print("🚀 Construindo modelo LSTM otimizado para M1 Max...")
    
    # Força uso da GPU se disponível
    device_name = "/GPU:0" if GPU_AVAILABLE else "/CPU:0"
    print(f"🎯 Usando dispositivo: {device_name}")
    
    # Habilita precisão mista se configurado
    if config.get('use_mixed_precision', False) and GPU_AVAILABLE:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ Precisão mista habilitada (mixed_float16)")
        except Exception as e:
            print(f"⚠️ Não foi possível habilitar precisão mista: {e}")
    
    # Constrói modelo dentro do contexto do dispositivo
    with tf.device(device_name):
        model = Sequential()
        
        # Primeira camada LSTM otimizada
        model.add(LSTM(
            config['layers'][0], 
            return_sequences=len(config['layers']) > 1,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            # Otimizações para Metal
            activation='tanh',  # Otimizado para Metal
            recurrent_activation='sigmoid',
            dropout=0.0,  # Dropout manual para melhor controle
            recurrent_dropout=0.0,
            implementation=2  # Implementação otimizada
        ))
        model.add(Dropout(config['dropout_rates'][0]))
        
        # Camadas LSTM intermediárias
        for i in range(1, len(config['layers'])):
            return_seq = i < len(config['layers']) - 1
            model.add(LSTM(
                config['layers'][i], 
                return_sequences=return_seq,
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=0.0,
                recurrent_dropout=0.0,
                implementation=2
            ))
            model.add(Dropout(config['dropout_rates'][i]))
        
        # Camadas densas otimizadas
        for j, dense_size in enumerate(config['dense_layers']):
            model.add(Dense(
                dense_size, 
                activation='relu',
                kernel_initializer='he_normal'  # Melhor para ReLU
            ))
            model.add(Dropout(0.2))
        
        # Camada de saída
        if config.get('use_mixed_precision', False):
            # Para precisão mista, usa float32 na saída
            model.add(Dense(1, dtype='float32'))
        else:
            model.add(Dense(1))
    
    print(f"🏗️ Modelo construído com {model.count_params():,} parâmetros")
    
    # Compilação otimizada para M1 Max
    with tf.device(device_name):
        optimizer = Adam(
            learning_rate=config['learning_rate'],
            clipnorm=config.get('clipnorm', 1.0),
            # Otimizações específicas para Metal
            amsgrad=False,  # Desabilita AMSGrad para velocidade
        )
        
        # Loss function otimizada
        loss_function = config.get('loss_function', 'mse')
        if config.get('use_mixed_precision', False):
            # Para precisão mista, usa loss scaling
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer, 
            loss=loss_function, 
            metrics=['mae']
        )
    
    print("🎯 Modelo compilado para M1 Max")
    print("📊 Arquitetura do modelo:")
    model.summary()
    
    # Callbacks otimizados para M1 Max
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
    
    # Adiciona ModelCheckpoint se especificado
    if config.get('save_best_model', True):
        callbacks.append(ModelCheckpoint(
            'models/best_lstm_m1_max.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))
    
    print(f"🚀 Iniciando treinamento M1 Max:")
    print(f"   • Épocas: {config['epochs']}")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"   • Dispositivo: {device_name}")
    print(f"   • Precisão Mista: {config.get('use_mixed_precision', False)}")
    
    # Treinamento otimizado
    try:
        with tf.device(device_name):
            history = model.fit(
                X_train, y_train,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1,
                # Otimizações para M1 Max
                use_multiprocessing=True,
                workers=4,  # Aproveita cores do M1 Max
                max_queue_size=20  # Buffer maior para GPU
            )
        
        print(f"✅ Treinamento M1 Max concluído em {len(history.history['loss'])} épocas")
        print(f"📈 Loss final: {history.history['loss'][-1]:.6f}")
        print(f"📉 Val Loss final: {history.history['val_loss'][-1]:.6f}")
        
        return model, history
        
    except Exception as e:
        print(f"❌ Erro no treinamento M1 Max: {e}")
        print("🔄 Tentando fallback para CPU...")
        
        # Fallback para CPU
        return train_advanced_lstm(X_train, y_train, X_test, y_test, config)

def train_advanced_lstm(X_train, y_train, X_test, y_test, config):
    """
    Treina modelo LSTM com arquitetura avançada (versão original mantida)
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow não disponível, pulando LSTM...")
        return None, None
    
    # Se M1 Max detectado e GPU disponível, usa versão otimizada
    if GPU_AVAILABLE and hasattr(config, 'get') and config.get('force_gpu', False):
        return train_advanced_lstm_m1_max(X_train, y_train, X_test, y_test, config)
    
    print("Construindo modelo LSTM avançado...")
    
    model = Sequential()
    
    # Primeira camada LSTM
    model.add(LSTM(
        config['layers'][0], 
        return_sequences=True, 
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(BatchNormalization())
    model.add(Dropout(config['dropout_rates'][0]))
    
    # Camadas LSTM intermediárias
    for i in range(1, len(config['layers'])-1):
        model.add(LSTM(config['layers'][i], return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(config['dropout_rates'][i]))
    
    # Última camada LSTM
    model.add(LSTM(config['layers'][-1]))
    model.add(BatchNormalization())
    model.add(Dropout(config['dropout_rates'][-1]))
    
    # Camadas densas
    for dense_size in config['dense_layers']:
        model.add(Dense(dense_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
    # Camada de saída
    model.add(Dense(1))
    
    # Compilação
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss=config['loss_function'], metrics=['mae'])
    
    print("Arquitetura do modelo:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['patience_early_stop'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['patience_reduce_lr'],
            min_lr=0.0001,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_lstm_model.h5',  # Mudando para formato .h5 para compatibilidade
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Treinamento
    print("Iniciando treinamento do LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def train_ensemble_models(X_train, y_train, quick_mode=False):
    """
    Treina múltiplos modelos para ensemble com logs detalhados
    """
    import time
    import threading
    import sys
    from datetime import datetime
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    print(f"📊 Dados achatados para modelos tradicionais: {X_train_flat.shape}")
    
    models = {}
    
    # Configura parâmetros baseado no modo
    if quick_mode:
        print("⚡ MODO RÁPIDO ATIVADO - Menos iterações para teste")
        rf_iterations = 10  # Reduzido de 200
        rf_cv_folds = 3     # Reduzido de 7
        gb_iterations = 5   # Reduzido de 30
        gb_cv_folds = 2     # Reduzido de 3
    else:
        print("🎯 MODO COMPLETO - Busca completa de hiperparâmetros")
        rf_iterations = RF_CONFIG['random_search_iterations']
        rf_cv_folds = RF_CONFIG['cv_folds']
        gb_iterations = 30
        gb_cv_folds = 3
    
    # Random Forest
    print(f"\n🌲 === INICIANDO TREINAMENTO RANDOM FOREST ===")
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🔧 Modo: {'RÁPIDO' if quick_mode else 'COMPLETO'}")
    
    rf_param_dist = {
        'n_estimators': RF_CONFIG['n_estimators_options'],
        'max_depth': RF_CONFIG['max_depth_options'],
        'min_samples_split': RF_CONFIG['min_samples_split_options'],
        'min_samples_leaf': RF_CONFIG['min_samples_leaf_options'],
        'max_features': RF_CONFIG['max_features_options']
    }
    
    print(f"🔧 Parâmetros para busca:")
    for param, values in rf_param_dist.items():
        print(f"   {param}: {values}")
    
    total_combinations = rf_iterations * rf_cv_folds
    print(f"🔍 Testando {rf_iterations} combinações com {rf_cv_folds} folds")
    print(f"📈 Total de fits: {total_combinations}")
    estimated_time = total_combinations * 2  # Estimativa de 2 segundos por fit
    print(f"⏱️  Tempo estimado: {estimated_time/60:.1f} minutos")
    print("💡 Aguarde o processamento...")
    
    start_time = time.time()
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf, rf_param_dist, 
        n_iter=rf_iterations,
        cv=TimeSeriesSplit(n_splits=rf_cv_folds),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2  # Aumentado para mais detalhes
    )
    
    print("🚀 Iniciando busca de hiperparâmetros...")
    
    # Monitor de progresso para RF
    def progress_monitor():
        start = time.time()
        counter = 0
        while not hasattr(progress_monitor, 'stop'):
            time.sleep(45)  # Aguarda 45 segundos
            if not hasattr(progress_monitor, 'stop'):
                counter += 1
                elapsed = time.time() - start
                print(f"   🔄 Random Forest ainda processando... {elapsed:.0f}s ({elapsed/60:.1f} min) - Update #{counter}")
                show_system_stats()
                sys.stdout.flush()
    
    # Inicia monitor em thread separada
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    try:
        rf_search.fit(X_train_flat, y_train)
    finally:
        # Para o monitor
        progress_monitor.stop = True
    
    rf_time = time.time() - start_time
    print(f"✅ Random Forest concluído!")
    print(f"⏱️  Tempo decorrido: {rf_time:.1f} segundos ({rf_time/60:.1f} minutos)")
    print(f"🏆 Melhor score: {rf_search.best_score_:.6f}")
    print(f"🔧 Melhores parâmetros:")
    for param, value in rf_search.best_params_.items():
        print(f"   {param}: {value}")
    
    models['rf'] = rf_search.best_estimator_
    
    # Gradient Boosting
    print(f"\n🚀 === INICIANDO TREINAMENTO GRADIENT BOOSTING ===")
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    
    gb_param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print(f"🔧 Parâmetros para busca:")
    for param, values in gb_param_dist.items():
        print(f"   {param}: {values}")
    
    gb_total_combinations = gb_iterations * gb_cv_folds
    print(f"🔍 Testando {gb_iterations} combinações com {gb_cv_folds} folds")
    print(f"📈 Total de fits: {gb_total_combinations}")
    gb_estimated_time = gb_total_combinations * 3  # Estimativa de 3 segundos por fit
    print(f"⏱️  Tempo estimado: {gb_estimated_time/60:.1f} minutos")
    print("💡 Processando Gradient Boosting...")
    
    start_time = time.time()
    
    gb = GradientBoostingRegressor(random_state=42)
    gb_search = RandomizedSearchCV(
        gb, gb_param_dist,
        n_iter=gb_iterations,
        cv=TimeSeriesSplit(n_splits=gb_cv_folds),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2  # Aumentado para mais detalhes
    )
    
    print("🚀 Iniciando busca de hiperparâmetros GB...")
    
    # Monitor de progresso para GB
    def progress_monitor_gb():
        start = time.time()
        counter = 0
        while not hasattr(progress_monitor_gb, 'stop'):
            time.sleep(30)  # Aguarda 30 segundos
            if not hasattr(progress_monitor_gb, 'stop'):
                counter += 1
                elapsed = time.time() - start
                print(f"   🔄 Gradient Boosting ainda processando... {elapsed:.0f}s ({elapsed/60:.1f} min) - Update #{counter}")
                show_system_stats()
                sys.stdout.flush()
    
    # Inicia monitor em thread separada
    monitor_thread_gb = threading.Thread(target=progress_monitor_gb, daemon=True)
    monitor_thread_gb.start()
    
    try:
        gb_search.fit(X_train_flat, y_train)
    finally:
        # Para o monitor
        progress_monitor_gb.stop = True
    
    gb_time = time.time() - start_time
    print(f"✅ Gradient Boosting concluído!")
    print(f"⏱️  Tempo decorrido: {gb_time:.1f} segundos ({gb_time/60:.1f} minutos)")
    print(f"🏆 Melhor score: {gb_search.best_score_:.6f}")
    print(f"🔧 Melhores parâmetros:")
    for param, value in gb_search.best_params_.items():
        print(f"   {param}: {value}")
    
    models['gb'] = gb_search.best_estimator_
    
    print(f"\n🎉 === ENSEMBLE TREINAMENTO CONCLUÍDO ===")
    print(f"✅ Modelos treinados: {list(models.keys())}")
    total_time = rf_time + gb_time
    print(f"⏱️  Tempo total do ensemble: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
    
    return models

def calculate_comprehensive_metrics(y_true, y_pred, model_name):
    """
    Calcula métricas abrangentes
    """
    metrics = {
        'Model': model_name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics

def plot_comprehensive_results(y_test, predictions_dict, history=None):
    """
    Plota resultados abrangentes
    """
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    # Plot 1: Comparação de previsões
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(y_test, label='Real', color='blue', linewidth=2)
    for name, preds in predictions_dict.items():
        plt.plot(preds, label=name, alpha=0.8)
    plt.title('Comparação de Previsões')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Erro por modelo
    plt.subplot(2, 3, 2)
    errors = {}
    for name, preds in predictions_dict.items():
        errors[name] = np.abs(y_test.flatten() - preds.flatten())
    
    plt.boxplot(errors.values(), labels=errors.keys())
    plt.title('Distribuição dos Erros')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # Plot 3: Scatter plot - Real vs Previsto
    plt.subplot(2, 3, 3)
    for name, preds in predictions_dict.items():
        plt.scatter(y_test, preds, alpha=0.6, label=name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title('Real vs Previsto')
    plt.legend()
    
    # Plot 4: Histórico de treinamento (se disponível)
    if history is not None:
        plt.subplot(2, 3, 4)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title('Histórico de Perda')
        plt.legend()
        plt.yscale('log')
    
    # Plot 5: Residuos
    plt.subplot(2, 3, 5)
    for name, preds in predictions_dict.items():
        residuals = y_test.flatten() - preds.flatten()
        plt.scatter(preds, residuals, alpha=0.6, label=name)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduos')
    plt.title('Análise de Resíduos')
    plt.legend()
    
    # Plot 6: Últimas previsões (zoom)
    plt.subplot(2, 3, 6)
    last_points = min(50, len(y_test))
    plt.plot(y_test[-last_points:], label='Real', color='blue', linewidth=2)
    for name, preds in predictions_dict.items():
        plt.plot(preds[-last_points:], label=name, alpha=0.8)
    plt.title('Últimas 50 Previsões')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_all_models(lstm_model, ensemble_models, scaler, feature_columns, model_dir='models'):
    """
    Salva todos os modelos treinados e metadados necessários
    """
    import os
    import joblib
    import json
    from datetime import datetime
    
    # Cria diretório se não existir
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n💾 === SALVANDO MODELOS ===")
    print(f"📁 Diretório: {model_dir}")
    saved_files = []
    
    try:
        # 1. Salva o scaler (ESSENCIAL para desnormalização)
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        saved_files.append(scaler_path)
        print(f"✅ Scaler salvo: {scaler_path}")
        
        # 2. Salva modelos ensemble (Random Forest e Gradient Boosting)
        for name, model in ensemble_models.items():
            model_path = os.path.join(model_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)
            saved_files.append(model_path)
            print(f"✅ {name.upper()} salvo: {model_path}")
        
        # 3. Modelo LSTM já é salvo automaticamente pelo ModelCheckpoint
        lstm_path = os.path.join(model_dir, 'best_lstm_model.h5')
        if os.path.exists(lstm_path):
            saved_files.append(lstm_path)
            print(f"✅ LSTM já salvo: {lstm_path}")
        elif lstm_model is not None:
            # Salva manualmente se não foi salvo pelo callback
            lstm_model.save(lstm_path)
            saved_files.append(lstm_path)
            print(f"✅ LSTM salvo manualmente: {lstm_path}")
        
        # 4. Salva metadados importantes
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'feature_columns': feature_columns,
            'seq_length': SEQ_LENGTH,
            'test_size': TEST_SIZE,
            'models_available': list(ensemble_models.keys()) + (['lstm'] if lstm_model else []),
            'total_features': len(feature_columns),
            'target_column': 'close'
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files.append(metadata_path)
        print(f"✅ Metadados salvos: {metadata_path}")
        
        # 5. Cria script de carregamento
        loading_script = f"""# Script para carregar os modelos salvos
import joblib
import json
import numpy as np
from tensorflow.keras.models import load_model

# Carrega metadados
with open('{metadata_path}', 'r') as f:
    metadata = json.load(f)

# Carrega scaler
scaler = joblib.load('{scaler_path}')

# Carrega modelos ensemble
ensemble_models = {{}}
"""
        
        for name in ensemble_models.keys():
            loading_script += f"ensemble_models['{name}'] = joblib.load('{model_dir}/{name}_model.pkl')\n"
        
        if lstm_model:
            loading_script += f"""
# Carrega modelo LSTM
lstm_model = load_model('{lstm_path}')

print("✅ Todos os modelos carregados com sucesso!")
print(f"Features disponíveis: {{len(metadata['feature_columns'])}}")
print(f"Modelos carregados: {{metadata['models_available']}}")
"""
        
        script_path = os.path.join(model_dir, 'load_models.py')
        with open(script_path, 'w') as f:
            f.write(loading_script)
        saved_files.append(script_path)
        print(f"✅ Script de carregamento criado: {script_path}")
        
        print(f"\n🎉 Salvamento concluído!")
        print(f"📊 Total de arquivos salvos: {len(saved_files)}")
        
        # Calcula tamanho total
        total_size = 0
        for file_path in saved_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                print(f"   📄 {os.path.basename(file_path)}: {size/1024/1024:.2f} MB")
        
        print(f"💽 Tamanho total: {total_size/1024/1024:.2f} MB")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ Erro ao salvar modelos: {e}")
        return []

# Função para carregar modelos salvos
def load_saved_models(model_dir='models'):
    """
    Carrega todos os modelos salvos
    """
    import os
    import joblib
    import json
    
    try:
        # Carrega metadados
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Carrega scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        
        # Carrega modelos ensemble
        ensemble_models = {}
        for name in ['rf', 'gb']:
            model_path = os.path.join(model_dir, f'{name}_model.pkl')
            if os.path.exists(model_path):
                ensemble_models[name] = joblib.load(model_path)
        
        # Carrega LSTM se disponível
        lstm_model = None
        lstm_path = os.path.join(model_dir, 'best_lstm_model.h5')
        if os.path.exists(lstm_path):
            try:
                from tensorflow.keras.models import load_model
                lstm_model = load_model(lstm_path)
            except:
                print("⚠️ TensorFlow não disponível, LSTM não carregado")
        
        print(f"✅ Modelos carregados de {model_dir}")
        print(f"📊 Features: {len(metadata['feature_columns'])}")
        print(f"🤖 Modelos: {metadata['models_available']}")
        
        return {
            'metadata': metadata,
            'scaler': scaler,
            'ensemble_models': ensemble_models,
            'lstm_model': lstm_model
        }
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return None

def main():
    """
    Função principal - Versão com PostgreSQL e logs melhorados
    """
    import time
    from datetime import datetime
    
    print("=== SISTEMA AVANÇADO DE PREVISÃO COM POSTGRESQL ===")
    print(f"🕐 Início da execução: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Configuração para fallback CSV
    use_database = DATABASE_AVAILABLE  # Só usa database se estiver disponível
    csv_fallback = 'relatorio_mensal_geral_2025-03 (1).csv'
    
    try:
        # Tenta carregar dados do PostgreSQL
        if use_database:
            print("Tentando carregar dados do PostgreSQL...")
            df, feature_columns = load_and_preprocess_advanced_data_from_db(
                TECHNICAL_FEATURES, 
                limit=None  # None = todos os dados, ou especifique um número para teste
            )
        else:
            raise Exception("DatabaseManager não disponível - usando CSV")
            
    except Exception as e:
        print(f"⚠️ Erro ao carregar do PostgreSQL: {e}")
        print("Tentando fallback para CSV...")
        
        try:
            df, feature_columns = load_and_preprocess_advanced_data_from_csv(
                csv_fallback, 
                TECHNICAL_FEATURES
            )
            print("✅ Dados carregados do CSV com sucesso")
        except Exception as csv_error:
            print(f"❌ Erro também no CSV: {csv_error}")
            print("Verifique as configurações do banco (.env) ou a existência do arquivo CSV")
            return
    
    # Prepara dados
    data = df[feature_columns].values
    target_col_idx = feature_columns.index('close')
    
    print(f"\nDados preparados:")
    print(f"Shape dos dados: {data.shape}")
    print(f"Índice da coluna 'close': {target_col_idx}")
    
    # Normalização
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Cria sequências
    X, y = create_sequences_advanced(data_scaled, target_col_idx, SEQ_LENGTH)
    
    # Divisão treino/teste
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # Treina modelos
    predictions = {}
    metrics_list = []
    
    # LSTM
    if TENSORFLOW_AVAILABLE:
        print("\n=== TREINANDO LSTM ===")
        lstm_model, history = train_advanced_lstm(X_train, y_train, X_test, y_test, LSTM_CONFIG)
        if lstm_model is not None:
            lstm_pred = lstm_model.predict(X_test)
            
            # Desnormaliza LSTM
            lstm_pred_full = np.zeros((len(lstm_pred), len(feature_columns)))
            lstm_pred_full[:, target_col_idx] = lstm_pred.flatten()
            lstm_pred_rescaled = scaler.inverse_transform(lstm_pred_full)[:, target_col_idx]
            
            predictions['LSTM'] = lstm_pred_rescaled
    else:
        history = None
        print("⚠️ TensorFlow não disponível, LSTM será ignorado")
    
    # Ensemble de modelos tradicionais
    print("\n=== TREINANDO MODELOS ENSEMBLE ===")
    print("🤖 Iniciando treinamento de modelos tradicionais (Random Forest + Gradient Boosting)")
    print("⚠️  IMPORTANTE: Este processo pode levar vários minutos dependendo do tamanho dos dados")
    print(f"📊 Dados de treino: {X_train.shape[0]} amostras com {X_train.shape[1]} timesteps e {X_train.shape[2]} features")
    
    # Pergunta sobre modo rápido
    print("\n🤔 Escolha o modo de treinamento:")
    print("   1️⃣  COMPLETO: Busca completa de hiperparâmetros (mais lento, melhor precisão)")
    print("   2️⃣  RÁPIDO: Busca reduzida para teste (mais rápido, precisão OK)")
    
    # Por padrão, usar modo rápido se muitos dados
    use_quick_mode = len(X_train) > 5000  # Modo rápido para datasets grandes
    
    if use_quick_mode:
        print("🚀 Dataset grande detectado - Usando MODO RÁPIDO automaticamente")
        print("   (Para forçar modo completo, modifique o código)")
    else:
        print("📊 Dataset pequeno/médio - Usando MODO COMPLETO")
    
    ensemble_start_time = time.time()
    ensemble_models = train_ensemble_models(X_train, y_train, quick_mode=use_quick_mode)
    ensemble_total_time = time.time() - ensemble_start_time
    
    print(f"🎯 Ensemble concluído em {ensemble_total_time:.1f} segundos ({ensemble_total_time/60:.1f} minutos)")
    
    # Desnormaliza y_test
    print("\n📈 === PREPARANDO DADOS PARA AVALIAÇÃO ===")
    print("🔄 Desnormalizando dados de teste...")
    y_test_full = np.zeros((len(y_test), len(feature_columns)))
    y_test_full[:, target_col_idx] = y_test
    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, target_col_idx]
    
    # Previsões dos modelos tradicionais
    print("\n🔮 === FAZENDO PREVISÕES ===")
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    print(f"📊 Dados de teste achatados: {X_test_flat.shape}")
    
    for name, model in ensemble_models.items():
        print(f"🎯 Fazendo previsões com {name.upper()}...")
        pred_start = time.time()
        pred = model.predict(X_test_flat)
        pred_time = time.time() - pred_start
        
        pred_full = np.zeros((len(pred), len(feature_columns)))
        pred_full[:, target_col_idx] = pred
        pred_rescaled = scaler.inverse_transform(pred_full)[:, target_col_idx]
        predictions[name.upper()] = pred_rescaled
        
        print(f"   ✅ {name.upper()} concluído em {pred_time:.2f}s")
        print(f"   📈 Previsão média: {pred_rescaled.mean():.2f}")
        print(f"   📊 Min: {pred_rescaled.min():.2f}, Max: {pred_rescaled.max():.2f}")
    
    # Ensemble final
    if len(predictions) > 1:
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        predictions['ENSEMBLE'] = ensemble_pred
    
    # Calcula métricas
    print("\n=== CALCULANDO MÉTRICAS ===")
    for name, preds in predictions.items():
        metrics = calculate_comprehensive_metrics(y_test_rescaled, preds, name)
        metrics_list.append(metrics)
    
    # Mostra resultados
    results_df = pd.DataFrame(metrics_list)
    print("\n=== RESULTADOS FINAIS ===")
    print(results_df.round(4))
    
    # Identifica o melhor modelo
    best_model_idx = results_df['RMSE'].idxmin()
    best_model = results_df.loc[best_model_idx, 'Model']
    best_rmse = results_df.loc[best_model_idx, 'RMSE']
    print(f"\n🏆 Melhor modelo: {best_model} (RMSE: {best_rmse:.4f})")
    
    # Previsão do próximo valor
    print("\n=== PREVISÃO DO PRÓXIMO VALOR ===")
    last_sequence = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, -1)
    
    next_predictions = {}
    
    for name, model in ensemble_models.items():
        if name == 'rf' or name == 'gb':
            next_pred = model.predict(last_sequence.reshape(1, -1))[0]
            next_pred_full = np.zeros((1, len(feature_columns)))
            next_pred_full[0, target_col_idx] = next_pred
            next_pred_rescaled = scaler.inverse_transform(next_pred_full)[0, target_col_idx]
            next_predictions[name.upper()] = next_pred_rescaled
            print(f"{name.upper()}: {next_pred_rescaled:.2f}")
    
    if TENSORFLOW_AVAILABLE and 'lstm_model' in locals():
        lstm_next = lstm_model.predict(last_sequence)[0, 0]
        lstm_next_full = np.zeros((1, len(feature_columns)))
        lstm_next_full[0, target_col_idx] = lstm_next
        lstm_next_rescaled = scaler.inverse_transform(lstm_next_full)[0, target_col_idx]
        next_predictions['LSTM'] = lstm_next_rescaled
        print(f"LSTM: {lstm_next_rescaled:.2f}")
    
    # Ensemble da próxima previsão
    if len(next_predictions) > 1:
        ensemble_next = np.mean(list(next_predictions.values()))
        print(f"ENSEMBLE: {ensemble_next:.2f}")
    
    # Salva todos os modelos
    print("\n=== SALVANDO MODELOS ===")
    lstm_model_to_save = lstm_model if TENSORFLOW_AVAILABLE and 'lstm_model' in locals() else None
    saved_files = save_all_models(
        lstm_model=lstm_model_to_save,
        ensemble_models=ensemble_models,
        scaler=scaler,
        feature_columns=feature_columns
    )
    
    if saved_files:
        print(f"🎯 Modelos salvos com sucesso em: models/")
        print("📖 Para usar os modelos salvos:")
        print("   1. Execute: from main_advanced import load_saved_models")
        print("   2. models = load_saved_models()")
        print("   3. Ou execute o script: python models/load_models.py")
    
    # Plots
    print("\n=== GERANDO GRÁFICOS ===")
    plot_comprehensive_results(y_test_rescaled.reshape(-1, 1), predictions, history)
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    print(f"🕐 Fim da execução: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Fonte dos dados: {'PostgreSQL' if use_database else 'CSV'}")
    print(f"Total de features utilizadas: {len(feature_columns)}")
    print(f"Modelos treinados: {list(predictions.keys())}")
    print(f"📁 Modelos salvos em: models/ ({len(saved_files)} arquivos)")
    
    return results_df, predictions, next_predictions

# Função para fazer previsões usando modelos salvos
def predict_with_saved_models(new_data, model_dir='models', target_column='close'):
    """
    Faz previsões usando modelos salvos
    
    Args:
        new_data: DataFrame com os mesmos indicadores técnicos do treinamento
        model_dir: Diretório onde estão os modelos salvos
        target_column: Nome da coluna target
    
    Returns:
        dict: Previsões de cada modelo
    """
    import os
    
    # Carrega modelos
    loaded = load_saved_models(model_dir)
    if loaded is None:
        print("❌ Não foi possível carregar os modelos")
        return None
    
    metadata = loaded['metadata']
    scaler = loaded['scaler']
    ensemble_models = loaded['ensemble_models']
    lstm_model = loaded['lstm_model']
    
    print(f"🔮 === FAZENDO PREVISÕES COM MODELOS SALVOS ===")
    print(f"📊 Dados de entrada: {new_data.shape}")
    
    try:
        # Prepara os dados
        feature_columns = metadata['feature_columns']
        seq_length = metadata['seq_length']
        
        # Verifica se todas as features estão presentes
        missing_features = [col for col in feature_columns if col not in new_data.columns]
        if missing_features:
            print(f"⚠️ Features faltando: {missing_features[:5]}...")
            print("Adicionando indicadores técnicos...")
            # Adiciona indicadores técnicos se necessário
            new_data = add_technical_indicators(new_data, feature_columns)
            new_data = add_lagged_features(new_data, ['close', 'open', 'high', 'low'], lags=[1, 2, 3, 5])
            new_data = add_rolling_statistics(new_data, ['close'], windows=[5, 10, 20])
        
        # Seleciona apenas as features do modelo
        data_features = new_data[feature_columns].values
        
        # Normaliza
        data_scaled = scaler.transform(data_features)
        
        predictions = {}
        
        # Previsões com modelos ensemble
        if len(ensemble_models) > 0:
            # Para modelos tradicionais, usa apenas o último ponto
            if len(data_scaled) >= seq_length:
                last_sequence = data_scaled[-seq_length:].reshape(1, -1)
                target_idx = feature_columns.index(target_column)
                
                for name, model in ensemble_models.items():
                    pred = model.predict(last_sequence)[0]
                    
                    # Desnormaliza
                    pred_full = np.zeros((1, len(feature_columns)))
                    pred_full[0, target_idx] = pred
                    pred_rescaled = scaler.inverse_transform(pred_full)[0, target_idx]
                    
                    predictions[name.upper()] = pred_rescaled
                    print(f"🎯 {name.upper()}: {pred_rescaled:.2f}")
        
        # Previsão com LSTM
        if lstm_model is not None and len(data_scaled) >= seq_length:
            lstm_sequence = data_scaled[-seq_length:].reshape(1, seq_length, -1)
            lstm_pred = lstm_model.predict(lstm_sequence, verbose=0)[0, 0]
            
            # Desnormaliza
            target_idx = feature_columns.index(target_column)
            lstm_pred_full = np.zeros((1, len(feature_columns)))
            lstm_pred_full[0, target_idx] = lstm_pred
            lstm_pred_rescaled = scaler.inverse_transform(lstm_pred_full)[0, target_idx]
            
            predictions['LSTM'] = lstm_pred_rescaled
            print(f"🧠 LSTM: {lstm_pred_rescaled:.2f}")
        
        # Ensemble final
        if len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()))
            predictions['ENSEMBLE'] = ensemble_pred
            print(f"🎯 ENSEMBLE: {ensemble_pred:.2f}")
        
        print(f"✅ Previsões concluídas!")
        return predictions
        
    except Exception as e:
        print(f"❌ Erro ao fazer previsões: {e}")
        return None

if __name__ == "__main__":
    main()

# main_advanced.py - Versão avançada com conexão PostgreSQL

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
from database import DatabaseManager

# Imports do TensorFlow (com tratamento de erro)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow não disponível. Apenas Random Forest será usado.")
    TENSORFLOW_AVAILABLE = False

# Imports do sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

def load_and_preprocess_advanced_data_from_db(config, limit=None):
    """
    Carregamento e pré-processamento avançado dos dados do PostgreSQL
    """
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

def train_advanced_lstm(X_train, y_train, X_test, y_test, config):
    """
    Treina modelo LSTM com arquitetura avançada
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow não disponível, pulando LSTM...")
        return None, None
    
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
            'best_lstm_model.h5',
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

def train_ensemble_models(X_train, y_train):
    """
    Treina múltiplos modelos para ensemble
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    models = {}
    
    # Random Forest
    print("Treinando Random Forest...")
    rf_param_dist = {
        'n_estimators': RF_CONFIG['n_estimators_options'],
        'max_depth': RF_CONFIG['max_depth_options'],
        'min_samples_split': RF_CONFIG['min_samples_split_options'],
        'min_samples_leaf': RF_CONFIG['min_samples_leaf_options'],
        'max_features': RF_CONFIG['max_features_options']
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf, rf_param_dist, 
        n_iter=RF_CONFIG['random_search_iterations'],
        cv=TimeSeriesSplit(n_splits=RF_CONFIG['cv_folds']),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    rf_search.fit(X_train_flat, y_train)
    models['rf'] = rf_search.best_estimator_
    
    # Gradient Boosting
    print("Treinando Gradient Boosting...")
    gb_param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    gb_search = RandomizedSearchCV(
        gb, gb_param_dist,
        n_iter=30,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    gb_search.fit(X_train_flat, y_train)
    models['gb'] = gb_search.best_estimator_
    
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

def main():
    """
    Função principal - Versão com PostgreSQL
    """
    print("=== SISTEMA AVANÇADO DE PREVISÃO COM POSTGRESQL ===")
    
    # Configuração para fallback CSV
    use_database = True
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
            raise Exception("Modo CSV selecionado")
            
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
    ensemble_models = train_ensemble_models(X_train, y_train)
    
    # Desnormaliza y_test
    y_test_full = np.zeros((len(y_test), len(feature_columns)))
    y_test_full[:, target_col_idx] = y_test
    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, target_col_idx]
    
    # Previsões dos modelos tradicionais
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
    
    # Plots
    print("\n=== GERANDO GRÁFICOS ===")
    plot_comprehensive_results(y_test_rescaled.reshape(-1, 1), predictions, history)
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    print(f"Fonte dos dados: {'PostgreSQL' if use_database else 'CSV'}")
    print(f"Total de features utilizadas: {len(feature_columns)}")
    print(f"Modelos treinados: {list(predictions.keys())}")
    
    return results_df, predictions, next_predictions

if __name__ == "__main__":
    main()

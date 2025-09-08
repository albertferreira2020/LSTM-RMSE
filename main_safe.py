# main_safe.py - Versão segura sem TensorFlow para evitar problemas de mutex

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Configurações
from config import *
from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics
from database import DatabaseManager

def load_and_preprocess_safe_data_from_db(config, limit=None):
    """
    Carregamento e pré-processamento dos dados do PostgreSQL (versão segura)
    """
    print("=== CARREGANDO DADOS DO POSTGRESQL (MODO SEGURO) ===")
    
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

def load_and_preprocess_safe_data_from_csv(file_path, config):
    """
    Função para carregar dados do CSV (fallback)
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

def create_sequences_for_ml(data, target_col_idx, seq_length):
    """
    Cria sequências para modelos de machine learning tradicionais
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        # Flatten a janela em uma única linha
        sequence_flat = data[i-seq_length:i].flatten()
        X.append(sequence_flat)
        y.append(data[i, target_col_idx])
    
    return np.array(X), np.array(y)

def train_safe_ensemble_models(X_train, y_train):
    """
    Treina múltiplos modelos sem usar TensorFlow
    """
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
    rf_search.fit(X_train, y_train)
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
    gb_search.fit(X_train, y_train)
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

def plot_safe_results(y_test, predictions_dict):
    """
    Plota resultados (versão segura)
    """
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    # Plot principal
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(y_test, label='Real', color='blue', linewidth=2)
    for name, preds in predictions_dict.items():
        plt.plot(preds, label=name, alpha=0.8)
    plt.title('Comparação de Previsões')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot de erro
    plt.subplot(2, 3, 2)
    errors = {}
    for name, preds in predictions_dict.items():
        errors[name] = np.abs(y_test.flatten() - preds.flatten())
    
    plt.boxplot(errors.values(), labels=errors.keys())
    plt.title('Distribuição dos Erros')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # Scatter plot
    plt.subplot(2, 3, 3)
    for name, preds in predictions_dict.items():
        plt.scatter(y_test, preds, alpha=0.6, label=name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title('Real vs Previsto')
    plt.legend()
    
    # Residuos
    plt.subplot(2, 3, 4)
    for name, preds in predictions_dict.items():
        residuals = y_test.flatten() - preds.flatten()
        plt.scatter(preds, residuals, alpha=0.6, label=name)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduos')
    plt.title('Análise de Resíduos')
    plt.legend()
    
    # Últimas previsões
    plt.subplot(2, 3, 5)
    last_points = min(50, len(y_test))
    plt.plot(y_test[-last_points:], label='Real', color='blue', linewidth=2)
    for name, preds in predictions_dict.items():
        plt.plot(preds[-last_points:], label=name, alpha=0.8)
    plt.title('Últimas 50 Previsões')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Métricas por modelo
    plt.subplot(2, 3, 6)
    rmse_values = []
    model_names = []
    for name, preds in predictions_dict.items():
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmse_values.append(rmse)
        model_names.append(name)
    
    plt.bar(model_names, rmse_values)
    plt.title('RMSE por Modelo')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Função principal - Versão segura sem TensorFlow
    """
    print("=== SISTEMA SEGURO DE PREVISÃO (SEM TENSORFLOW) ===")
    
    # Configuração para fallback CSV
    use_database = True
    csv_fallback = 'relatorio_mensal_geral_2025-03 (1).csv'
    
    try:
        # Tenta carregar dados do PostgreSQL
        if use_database:
            print("Tentando carregar dados do PostgreSQL...")
            df, feature_columns = load_and_preprocess_safe_data_from_db(
                TECHNICAL_FEATURES, 
                limit=None  # None = todos os dados, ou especifique um número para teste
            )
        else:
            raise Exception("Modo CSV selecionado")
            
    except Exception as e:
        print(f"⚠️ Erro ao carregar do PostgreSQL: {e}")
        print("Tentando fallback para CSV...")
        
        try:
            df, feature_columns = load_and_preprocess_safe_data_from_csv(
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
    
    # Cria sequências para ML tradicional
    X, y = create_sequences_for_ml(data_scaled, target_col_idx, SEQ_LENGTH)
    
    # Divisão treino/teste
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # Treina modelos tradicionais
    print("\n=== TREINANDO MODELOS ENSEMBLE (SEGURO) ===")
    models = train_safe_ensemble_models(X_train, y_train)
    
    # Desnormaliza y_test
    y_test_full = np.zeros((len(y_test), len(feature_columns)))
    y_test_full[:, target_col_idx] = y_test
    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, target_col_idx]
    
    # Previsões dos modelos
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_test)
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
    metrics_list = []
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
    last_sequence = data_scaled[-SEQ_LENGTH:].flatten().reshape(1, -1)
    
    next_predictions = {}
    
    for name, model in models.items():
        next_pred = model.predict(last_sequence)[0]
        next_pred_full = np.zeros((1, len(feature_columns)))
        next_pred_full[0, target_col_idx] = next_pred
        next_pred_rescaled = scaler.inverse_transform(next_pred_full)[0, target_col_idx]
        next_predictions[name.upper()] = next_pred_rescaled
        print(f"{name.upper()}: {next_pred_rescaled:.2f}")
    
    # Ensemble da próxima previsão
    if len(next_predictions) > 1:
        ensemble_next = np.mean(list(next_predictions.values()))
        print(f"ENSEMBLE: {ensemble_next:.2f}")
    
    # Plots
    print("\n=== GERANDO GRÁFICOS ===")
    plot_safe_results(y_test_rescaled.reshape(-1, 1), predictions)
    
    print("\n=== ANÁLISE CONCLUÍDA (MODO SEGURO) ===")
    print(f"Fonte dos dados: {'PostgreSQL' if use_database else 'CSV'}")
    print(f"Total de features utilizadas: {len(feature_columns)}")
    print(f"Modelos treinados: {list(predictions.keys())}")
    print("ℹ️ Esta versão não usa TensorFlow/LSTM para evitar problemas de threading")
    
    return results_df, predictions, next_predictions

if __name__ == "__main__":
    main()

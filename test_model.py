#!/usr/bin/env python3
"""
Script para testar o modelo Random Forest salvo
Carrega o modelo e faz previsões em novos dados
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from config import *
from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics
from database import DatabaseManager
import warnings
warnings.filterwarnings('ignore')

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
    
    print(f"📊 Total de features: {len(all_features)}")
    
    return df_enhanced, all_features

def load_saved_model(model_path):
    """Carrega o modelo salvo"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Modelo carregado de: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None

def prepare_test_data(limit=1000):
    """Prepara dados de teste"""
    print("📊 Preparando dados de teste...")
    
    # Conecta ao banco
    db = DatabaseManager()
    
    # Carrega dados recentes
    df = db.load_botbinance_data(limit=limit, order_by='id DESC')  # Dados mais recentes
    
    if df is None or df.empty:
        print("❌ Erro ao carregar dados")
        return None, None, None

    print(f"📈 Dados carregados: {len(df)} registros")
    print(f"📅 Período: {df['created_at'].min()} até {df['created_at'].max()}")
    
    # Usa a mesma engenharia de features do treinamento
    config = TECHNICAL_FEATURES_PARALLEL
    df_enhanced, all_features = parallel_feature_engineering(df, config)
    
    # Seleciona features para o modelo (exclui colunas de ID e timestamp)
    exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
    feature_columns = [col for col in all_features if col not in exclude_columns and col in df_enhanced.columns]
    
    print(f"✅ Features selecionadas: {len(feature_columns)}")
    print(f"📝 Primeiras 15 features: {feature_columns[:15]}")
    
    # Prepara dados para normalização (features + target)
    data = df_enhanced[feature_columns + ['close']].values
    target_idx = len(feature_columns)  # Índice da coluna 'close'
    
    print(f"📊 Dados antes da normalização: {data.shape}")
    print(f"🎯 Target (close) - Min: {data[:, target_idx].min():.2f}, Max: {data[:, target_idx].max():.2f}")
    
    # ⚠️ IMPORTANTE: Aplica a mesma normalização do treinamento
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    print(f"✅ Dados normalizados aplicados")
    print(f"🎯 Target normalizado - Min: {data_scaled[:, target_idx].min():.2f}, Max: {data_scaled[:, target_idx].max():.2f}")
    
    seq_length = SEQ_LENGTH_PARALLEL  # 60 da configuração
    
    print(f"🔧 Criando sequências com seq_length={seq_length}")
    
    # Cria sequências com dados normalizados
    X_sequences = []
    y_sequences = []
    
    for i in range(seq_length, len(data_scaled)):
        X_sequences.append(data_scaled[i-seq_length:i, :-1])  # Todas as features exceto target
        y_sequences.append(data_scaled[i, target_idx])        # Target normalizado
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Achata as sequências para o Random Forest (igual ao treinamento)
    X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
    
    print(f"📊 Shape sequências: X_sequences={X_sequences.shape}, y={y_sequences.shape}")
    print(f"📊 Shape achatado: X_flat={X_flat.shape}")
    
    # Retorna também o scaler e dados originais para desnormalização
    return X_flat, y_sequences, df_enhanced, scaler, data[:, target_idx]

def test_model_predictions_normalized(model, X, y_normalized, y_original, df, n_predictions=20):
    """Testa o modelo no espaço normalizado (sem desnormalização)"""
    print(f"\n🔮 Fazendo {n_predictions} previsões...")
    
    # Faz previsões no espaço normalizado
    y_pred_normalized = model.predict(X[-n_predictions:])
    y_real_normalized = y_normalized[-n_predictions:]
    y_real_original = y_original[-n_predictions:]  # Para referência
    
    # Calcula métricas NO ESPAÇO NORMALIZADO
    rmse_norm = np.sqrt(mean_squared_error(y_real_normalized, y_pred_normalized))
    mae_norm = mean_absolute_error(y_real_normalized, y_pred_normalized)
    r2_norm = r2_score(y_real_normalized, y_pred_normalized)
    
    print(f"\n📊 === MÉTRICAS NO ESPAÇO NORMALIZADO ===")
    print(f"🎯 RMSE: {rmse_norm:.6f}")
    print(f"🎯 MAE: {mae_norm:.6f}")
    print(f"🎯 R²: {r2_norm:.6f}")
    
    # Mostra previsões no espaço normalizado
    print(f"\n🔍 === PREVISÕES NORMALIZADAS ===")
    timestamps = df['created_at'].iloc[-n_predictions:].values
    
    for i in range(min(10, n_predictions)):
        erro_norm = abs(y_pred_normalized[i] - y_real_normalized[i])
        erro_pct = (erro_norm / y_real_normalized[i]) * 100 if y_real_normalized[i] != 0 else 0
        print(f"[{i+1:2d}] Norm - Real: {y_real_normalized[i]:.6f} | Pred: {y_pred_normalized[i]:.6f} | Erro: {erro_norm:.6f} ({erro_pct:.2f}%)")
        print(f"     Orig - Real: {y_real_original[i]:8.2f}")
    
    return y_pred_normalized, y_real_normalized, timestamps

def plot_predictions(y_real, y_pred, timestamps, n_show=20):
    """Plota gráfico de previsões vs real"""
    plt.figure(figsize=(15, 8))
    
    # Últimas n previsões
    indices = range(len(y_real))
    
    plt.subplot(2, 1, 1)
    plt.plot(indices, y_real, 'b-', label='Preço Real', linewidth=2)
    plt.plot(indices, y_pred, 'r--', label='Previsão RF', linewidth=2)
    plt.title('Random Forest - Previsões vs Preços Reais')
    plt.ylabel('Preço')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de erros
    plt.subplot(2, 1, 2)
    erros = y_pred - y_real
    plt.plot(indices, erros, 'g-', label='Erro (Pred - Real)', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title('Erros de Previsão')
    plt.xlabel('Índice')
    plt.ylabel('Erro')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva o gráfico
    plot_path = 'plots/test_predictions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo em: {plot_path}")
    plt.show()

def main():
    """Função principal"""
    print("🤖 === TESTE DO MODELO RANDOM FOREST ===\n")
    
    # Caminho do modelo
    model_path = 'models/best_rf_parallel.pkl'
    
    # 1. Carrega o modelo
    model = load_saved_model(model_path)
    if model is None:
        return
    
    # Mostra informações do modelo
    print(f"📋 Tipo do modelo: {type(model).__name__}")
    if hasattr(model, 'n_estimators'):
        print(f"🌳 N° de árvores: {model.n_estimators}")
    if hasattr(model, 'max_depth'):
        print(f"📏 Profundidade máxima: {model.max_depth}")
    
    # 2. Prepara dados de teste
    X_test, y_test_normalized, df_test, scaler, y_test_original = prepare_test_data(limit=1000)
    if X_test is None:
        return
    
    # 3. Testa o modelo (no espaço normalizado)
    y_pred, y_real, timestamps = test_model_predictions_normalized(
        model, X_test, y_test_normalized, y_test_original, df_test, n_predictions=50
    )
    
    # 4. Plota resultados
    plot_predictions(y_real, y_pred, timestamps)
    
    # 5. Feature importance (se disponível)
    if hasattr(model, 'feature_importances_'):
        print(f"\n🏆 === TOP 10 FEATURES MAIS IMPORTANTES ===")
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    print(f"\n✅ Teste concluído! Modelo Random Forest funcionando perfeitamente.")

if __name__ == "__main__":
    main()

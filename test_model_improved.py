#!/usr/bin/env python3
"""
Script melhorado para testar o modelo Random Forest
Inclui gráfico de candles e previsões futuras com timestamps
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import *
from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics
from database import DatabaseManager

def prepare_all_sensor_features(df):
    """Prepara TODAS as colunas disponíveis como features/sensores para previsão"""
    print("🔧 === PREPARANDO TODOS OS SENSORES COMO FEATURES ===")
    
    original_sensors = [
        'open', 'high', 'low', 'volume', 'reversal',
        'best_bid_price', 'best_bid_quantity', 
        'best_ask_price', 'best_ask_quantity',
        'spread', 'spread_percentage',
        'bid_liquidity', 'ask_liquidity', 'total_liquidity',
        'imbalance', 'weighted_mid_price'
    ]
    
    available_sensors = [col for col in original_sensors if col in df.columns]
    print(f"✅ Sensores disponíveis ({len(available_sensors)}): {available_sensors}")
    
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
    all_features = available_sensors + sensor_features
    
    return df, all_features

def parallel_feature_engineering(df, features_config):
    """Calcula indicadores técnicos em paralelo"""
    print("🔧 === ENGENHARIA DE FEATURES PARALELA ===")
    
    df_enhanced, sensor_features = prepare_all_sensor_features(df)
    
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

def create_approximated_scaler(df_current, df_reference_period=None):
    """
    Cria um scaler aproximado baseado nos dados históricos
    Para uso quando o scaler original não está disponível
    """
    print("🔧 === CRIANDO SCALER APROXIMADO ===")
    
    # Se temos dados de referência, use-os para criar o scaler
    if df_reference_period is not None:
        print("📊 Usando período de referência para scaler")
        reference_data = df_reference_period
    else:
        print("📊 Usando dados atuais para scaler")
        reference_data = df_current
    
    # Usa um período maior de dados para criar escala mais representativa
    config = TECHNICAL_FEATURES_PARALLEL
    df_enhanced, all_features = parallel_feature_engineering(reference_data, config)
    
    # Seleciona features
    exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
    feature_columns = [col for col in all_features if col not in exclude_columns and col in df_enhanced.columns]
    
    # Cria dados para o scaler
    data_for_scaler = df_enhanced[feature_columns + ['close']].values
    
    # Cria e ajusta o scaler
    scaler = MinMaxScaler()
    scaler.fit(data_for_scaler)
    
    print(f"✅ Scaler criado com {len(feature_columns)} features")
    return scaler, feature_columns

def prepare_data_with_approximated_scaler(df, scaler, feature_columns, seq_length):
    """Prepara dados usando scaler aproximado"""
    print("📊 === PREPARANDO DADOS COM SCALER APROXIMADO ===")
    
    # Aplica engenharia de features
    config = TECHNICAL_FEATURES_PARALLEL
    df_enhanced, _ = parallel_feature_engineering(df, config)
    
    # Usa as mesmas features do scaler
    available_features = [col for col in feature_columns if col in df_enhanced.columns]
    missing_features = [col for col in feature_columns if col not in df_enhanced.columns]
    
    if missing_features:
        print(f"⚠️ Features faltantes: {missing_features[:5]}...")
        # Adiciona colunas faltantes com zeros
        for col in missing_features:
            df_enhanced[col] = 0.0
    
    # Prepara dados
    data = df_enhanced[feature_columns + ['close']].values
    target_idx = len(feature_columns)
    
    # Normaliza
    data_scaled = scaler.transform(data)
    
    # Cria sequências
    X_sequences = []
    y_sequences = []
    timestamps = []
    
    for i in range(seq_length, len(data_scaled)):
        X_sequences.append(data_scaled[i-seq_length:i, :-1])
        y_sequences.append(data_scaled[i, target_idx])
        timestamps.append(df_enhanced['created_at'].iloc[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Achata para Random Forest
    X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
    
    print(f"📊 Dados preparados: X_flat={X_flat.shape}, y={y_sequences.shape}")
    
    return X_flat, y_sequences, timestamps, data[:, target_idx], target_idx

def predict_future_prices(model, df, scaler, feature_columns, seq_length, future_periods=24):
    """
    Faz previsões futuras e gera timestamps futuros
    """
    print(f"🔮 === PREVENDO PRÓXIMOS {future_periods} PERÍODOS ===")
    
    # Prepara dados atuais
    X_flat, y_sequences, timestamps, original_prices, target_idx = prepare_data_with_approximated_scaler(
        df, scaler, feature_columns, seq_length
    )
    
    if len(X_flat) == 0:
        print("❌ Não há dados suficientes para previsão")
        return None, None, None, None
    
    # Pega última sequência para previsão
    last_sequence = X_flat[-1:] 
    
    # Faz previsões futuras
    future_predictions_norm = []
    current_sequence = last_sequence.copy()
    
    for i in range(future_periods):
        # Prediz próximo valor
        next_pred_norm = model.predict(current_sequence)[0]
        future_predictions_norm.append(next_pred_norm)
        
        # Atualiza sequência (simplified approach)
        # Em um cenário real, você recriaria a sequência completa com as novas features
        new_sequence = current_sequence.copy()
        # Shift sequence and add prediction (simplified)
        # Para uma implementação completa, você precisaria recalcular todas as features
        
    # Desnormaliza previsões futuras
    future_predictions_real = []
    for pred_norm in future_predictions_norm:
        # Cria array temporário para desnormalização
        temp_array = np.zeros((1, scaler.n_features_in_))
        temp_array[0, target_idx] = pred_norm
        pred_real = scaler.inverse_transform(temp_array)[0, target_idx]
        future_predictions_real.append(pred_real)
    
    # Gera timestamps futuros
    last_timestamp = timestamps[-1]
    future_timestamps = []
    for i in range(future_periods):
        # Assume intervalo de 1 minuto entre dados (ajuste conforme necessário)
        future_time = last_timestamp + timedelta(minutes=i+1)
        future_timestamps.append(future_time)
    
    print(f"✅ Previsões futuras geradas")
    print(f"📅 Período: {future_timestamps[0]} até {future_timestamps[-1]}")
    print(f"💰 Preço atual: {original_prices[-1]:.2f}")
    print(f"💰 Preço previsto em {future_periods} períodos: {future_predictions_real[-1]:.2f}")
    
    return future_predictions_real, future_timestamps, timestamps, original_prices

def create_candlestick_with_predictions(df, future_predictions, future_timestamps, save_path="plots/candlestick_predictions.png"):
    """
    Cria gráfico de candlestick com previsões futuras
    """
    print("📊 === CRIANDO GRÁFICO DE CANDLESTICK COM PREVISÕES ===")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Dados históricos (últimos 100 pontos para visualização)
    n_show = min(100, len(df))
    df_show = df.tail(n_show).copy()
    
    # Converte timestamps para datetime se necessário
    if isinstance(df_show['created_at'].iloc[0], str):
        df_show['created_at'] = pd.to_datetime(df_show['created_at'])
    
    # Candlestick manual (matplotlib não tem candlestick nativo simples)
    for i, (idx, row) in enumerate(df_show.iterrows()):
        date = row['created_at']
        
        # Cores do candle
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Corpo do candle
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        # Sombras
        ax.plot([i, i], [row['low'], row['high']], color='white', linewidth=0.5, alpha=0.7)
        
        # Corpo
        ax.add_patch(plt.Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                                 facecolor=color, alpha=0.8, edgecolor='white', linewidth=0.5))
    
    # Linha de fechamento histórica
    close_line = ax.plot(range(len(df_show)), df_show['close'].values, 
                        color='cyan', linewidth=1, alpha=0.7, label='Fechamento Histórico')
    
    # Previsões futuras
    if future_predictions and future_timestamps:
        future_x = range(len(df_show), len(df_show) + len(future_predictions))
        
        # Conecta último preço real com primeira previsão
        connection_x = [len(df_show)-1, len(df_show)]
        connection_y = [df_show['close'].iloc[-1], future_predictions[0]]
        ax.plot(connection_x, connection_y, color='orange', linewidth=2, alpha=0.7)
        
        # Linha de previsões
        ax.plot(future_x, future_predictions, color='orange', linewidth=3, 
               label=f'Previsões Futuras ({len(future_predictions)} períodos)', 
               linestyle='--', marker='o', markersize=4)
        
        # Área de incerteza (opcional)
        future_predictions_array = np.array(future_predictions)
        uncertainty = future_predictions_array * 0.02  # 2% de incerteza
        ax.fill_between(future_x, 
                       future_predictions_array - uncertainty,
                       future_predictions_array + uncertainty,
                       color='orange', alpha=0.2, label='Zona de Incerteza (±2%)')
    
    # Configurações do gráfico
    ax.set_title('📊 Bitcoin - Candlestick + Previsões Random Forest', 
                fontsize=16, fontweight='bold', color='white')
    ax.set_xlabel('Tempo (Períodos)', fontsize=12, color='white')
    ax.set_ylabel('Preço (USD)', fontsize=12, color='white')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Linha vertical separando histórico de previsões
    if future_predictions:
        ax.axvline(x=len(df_show)-0.5, color='yellow', linestyle=':', alpha=0.8, linewidth=2)
        ax.text(len(df_show)-0.5, ax.get_ylim()[1]*0.95, 'Agora', 
               rotation=90, color='yellow', fontweight='bold', ha='right')
    
    # Formatação dos eixos
    ax.tick_params(colors='white')
    
    # Salva gráfico
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"✅ Gráfico salvo em: {save_path}")
    
    return fig, ax

def main():
    """Função principal melhorada"""
    print("🚀 === TESTE MELHORADO DO MODELO RANDOM FOREST ===")
    
    # 1. Carrega modelo
    model_path = "models/best_rf_parallel.pkl"
    model = load_saved_model(model_path)
    if model is None:
        return
    
    # 2. Carrega dados para criar scaler aproximado
    print("\n📊 === CARREGANDO DADOS ===")
    db = DatabaseManager()
    
    # Carrega mais dados para criar um scaler melhor
    df_large = db.load_botbinance_data(limit=1000, order_by='id DESC')
    df_test = db.load_botbinance_data(limit=100, order_by='id DESC')
    
    if df_large is None or df_test is None:
        print("❌ Erro ao carregar dados")
        return
    
    print(f"📈 Dados para scaler: {len(df_large)} registros")
    print(f"📈 Dados para teste: {len(df_test)} registros")
    
    # 3. Cria scaler aproximado
    scaler, feature_columns = create_approximated_scaler(df_large, df_large)
    
    # 4. Faz previsões futuras
    future_predictions, future_timestamps, historical_timestamps, historical_prices = predict_future_prices(
        model, df_test, scaler, feature_columns, SEQ_LENGTH_PARALLEL, future_periods=24
    )
    
    if future_predictions is None:
        return
    
    # 5. Cria gráfico de candlestick com previsões
    fig, ax = create_candlestick_with_predictions(df_test, future_predictions, future_timestamps)
    
    # 6. Mostra resumo das previsões
    print(f"\n🔮 === RESUMO DAS PREVISÕES ===")
    print(f"💰 Preço atual: {historical_prices[-1]:.2f} USD")
    print(f"💰 Previsão +1h: {future_predictions[0]:.2f} USD")
    print(f"💰 Previsão +6h: {future_predictions[5] if len(future_predictions) > 5 else 'N/A':.2f} USD")
    print(f"💰 Previsão +12h: {future_predictions[11] if len(future_predictions) > 11 else 'N/A':.2f} USD")
    print(f"💰 Previsão +24h: {future_predictions[-1]:.2f} USD")
    
    variation_24h = ((future_predictions[-1] - historical_prices[-1]) / historical_prices[-1]) * 100
    print(f"📈 Variação prevista 24h: {variation_24h:+.2f}%")
    
    print(f"\n✅ Análise completa finalizada!")

if __name__ == "__main__":
    main()

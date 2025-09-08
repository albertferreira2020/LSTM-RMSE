#!/usr/bin/env python3
"""
Script para análise detalhada da precisão e confiança do modelo Random Forest
Inclui múltiplas métricas de avaliação e intervalos de confiança
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import *
from technical_indicators import add_technical_indicators, add_lagged_features, add_rolling_statistics
from database import DatabaseManager

def prepare_all_sensor_features(df):
    """Prepara TODAS as colunas disponíveis como features/sensores para previsão"""
    original_sensors = [
        'open', 'high', 'low', 'volume', 'reversal',
        'best_bid_price', 'best_bid_quantity', 
        'best_ask_price', 'best_ask_quantity',
        'spread', 'spread_percentage',
        'bid_liquidity', 'ask_liquidity', 'total_liquidity',
        'imbalance', 'weighted_mid_price'
    ]
    
    available_sensors = [col for col in original_sensors if col in df.columns]
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
    
    all_features = available_sensors + sensor_features
    return df, all_features

def parallel_feature_engineering(df, features_config):
    """Calcula indicadores técnicos em paralelo"""
    df_enhanced, sensor_features = prepare_all_sensor_features(df)
    
    # Calcula indicadores técnicos básicos
    try:
        df_enhanced = add_technical_indicators(df_enhanced, features_config)
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
    except Exception as e:
        print(f"⚠️ Erro ao adicionar features lag: {e}")
    
    # Remove linhas com NaN
    df_enhanced = df_enhanced.dropna()
    
    # Features finais
    all_features = [col for col in df_enhanced.columns if col not in ['id', 'created_at', 'updated_at', 'timestamp']]
    
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

def prepare_test_data_for_analysis(limit=2000):
    """Prepara dados de teste para análise de precisão"""
    print("📊 === PREPARANDO DADOS PARA ANÁLISE DE PRECISÃO ===")
    
    # Conecta ao banco
    db = DatabaseManager()
    
    # Carrega dados recentes para teste
    df = db.load_botbinance_data(limit=limit, order_by='id DESC')
    
    if df is None or df.empty:
        print("❌ Erro ao carregar dados")
        return None, None, None, None, None

    print(f"📈 Dados carregados: {len(df)} registros")
    
    # Usa a mesma engenharia de features do treinamento
    config = TECHNICAL_FEATURES_PARALLEL
    df_enhanced, all_features = parallel_feature_engineering(df, config)
    
    # Seleciona features para o modelo
    exclude_columns = ['id', 'created_at', 'updated_at', 'timestamp']
    feature_columns = [col for col in all_features if col not in exclude_columns and col in df_enhanced.columns]
    
    # Prepara dados para normalização
    data = df_enhanced[feature_columns + ['close']].values
    target_idx = len(feature_columns)
    
    # Normalização
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    seq_length = SEQ_LENGTH_PARALLEL
    
    # Cria sequências
    X_sequences = []
    y_sequences = []
    y_original = []
    
    for i in range(seq_length, len(data_scaled)):
        X_sequences.append(data_scaled[i-seq_length:i, :-1])
        y_sequences.append(data_scaled[i, target_idx])
        y_original.append(data[i, target_idx])  # Valores originais
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    y_original = np.array(y_original)
    
    # Achata as sequências para o Random Forest
    X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
    
    print(f"📊 Shape dos dados: X_flat={X_flat.shape}, y={y_sequences.shape}")
    
    return X_flat, y_sequences, y_original, scaler, target_idx

def calculate_confidence_intervals(model, X, n_estimators=None):
    """
    Calcula intervalos de confiança usando as previsões individuais das árvores
    """
    if hasattr(model, 'estimators_'):
        # Pega previsões de cada árvore individual
        tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
        
        # Calcula estatísticas
        mean_pred = np.mean(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)
        
        # Intervalos de confiança (95%)
        confidence_95_lower = mean_pred - 1.96 * std_pred
        confidence_95_upper = mean_pred + 1.96 * std_pred
        
        # Intervalos de confiança (68% - 1 desvio padrão)
        confidence_68_lower = mean_pred - std_pred
        confidence_68_upper = mean_pred + std_pred
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'confidence_95_lower': confidence_95_lower,
            'confidence_95_upper': confidence_95_upper,
            'confidence_68_lower': confidence_68_lower,
            'confidence_68_upper': confidence_68_upper
        }
    else:
        return None

def analyze_model_precision(model):
    """
    Análise completa da precisão do modelo Random Forest
    """
    print("🎯 === ANÁLISE DETALHADA DE PRECISÃO E CONFIANÇA ===")
    
    # 1. Prepara dados de teste
    X_test, y_test_norm, y_test_orig, scaler, target_idx = prepare_test_data_for_analysis(limit=2000)
    
    if X_test is None:
        return
    
    # 2. Faz previsões
    print("\n🔮 Fazendo previsões para análise...")
    y_pred_norm = model.predict(X_test)
    
    # 3. Desnormaliza previsões para valores reais
    print("🔄 Desnormalizando previsões...")
    
    # Cria arrays temporários para desnormalização
    temp_real = np.zeros((len(y_test_norm), scaler.n_features_in_))
    temp_pred = np.zeros((len(y_pred_norm), scaler.n_features_in_))
    
    temp_real[:, target_idx] = y_test_norm
    temp_pred[:, target_idx] = y_pred_norm
    
    # Desnormaliza (aproximação - pode ter pequenos erros devido ao scaler diferente)
    # Para análise, usaremos os valores originais diretamente
    y_real = y_test_orig
    
    # Aproximação da desnormalização das previsões
    # Calcula a escala aproximada
    real_min, real_max = y_real.min(), y_real.max()
    norm_min, norm_max = y_test_norm.min(), y_test_norm.max()
    
    # Desnormaliza as previsões usando escala aproximada
    y_pred_approx = y_pred_norm * (real_max - real_min) + real_min
    
    # 4. Calcula intervalos de confiança
    print("📊 Calculando intervalos de confiança...")
    confidence_data = calculate_confidence_intervals(model, X_test[-100:])  # Últimas 100 previsões
    
    # 5. Métricas de precisão no espaço normalizado
    print("\n📈 === MÉTRICAS NO ESPAÇO NORMALIZADO ===")
    rmse_norm = np.sqrt(mean_squared_error(y_test_norm, y_pred_norm))
    mae_norm = mean_absolute_error(y_test_norm, y_pred_norm)
    r2_norm = r2_score(y_test_norm, y_pred_norm)
    mape_norm = mean_absolute_percentage_error(y_test_norm, y_pred_norm) * 100
    
    print(f"🎯 RMSE (normalizado): {rmse_norm:.6f}")
    print(f"🎯 MAE (normalizado): {mae_norm:.6f}")
    print(f"🎯 R² (coef. determinação): {r2_norm:.4f} ({r2_norm*100:.2f}%)")
    print(f"🎯 MAPE (erro % médio): {mape_norm:.2f}%")
    
    # 6. Métricas aproximadas no espaço real
    print(f"\n💰 === MÉTRICAS APROXIMADAS (VALORES REAIS) ===")
    rmse_real = np.sqrt(mean_squared_error(y_real[-len(y_pred_approx):], y_pred_approx))
    mae_real = mean_absolute_error(y_real[-len(y_pred_approx):], y_pred_approx)
    r2_real = r2_score(y_real[-len(y_pred_approx):], y_pred_approx)
    
    print(f"💰 RMSE (USD): ${rmse_real:,.2f}")
    print(f"💰 MAE (USD): ${mae_real:,.2f}")
    print(f"💰 R² (real): {r2_real:.4f} ({r2_real*100:.2f}%)")
    
    # 7. Análise de direção (alta/baixa)
    print(f"\n📊 === ANÁLISE DE DIREÇÃO ===")
    
    # Calcula mudanças de direção
    real_changes = np.diff(y_real[-len(y_pred_approx):])
    pred_changes = np.diff(y_pred_approx)
    
    # Classifica como alta (1) ou baixa (-1)
    real_direction = np.sign(real_changes)
    pred_direction = np.sign(pred_changes)
    
    # Accuracy direcional
    direction_accuracy = np.mean(real_direction == pred_direction) * 100
    
    print(f"🎯 Precisão Direcional: {direction_accuracy:.2f}%")
    print(f"   (Acerta direção da variação em {direction_accuracy:.1f}% dos casos)")
    
    # 8. Distribuição dos erros
    errors_norm = np.abs(y_test_norm - y_pred_norm)
    errors_pct = (errors_norm / y_test_norm) * 100
    
    print(f"\n📊 === DISTRIBUIÇÃO DOS ERROS ===")
    print(f"🎯 Erro < 1%: {np.mean(errors_pct < 1)*100:.1f}% das previsões")
    print(f"🎯 Erro < 2%: {np.mean(errors_pct < 2)*100:.1f}% das previsões")
    print(f"🎯 Erro < 5%: {np.mean(errors_pct < 5)*100:.1f}% das previsões")
    print(f"🎯 Erro médio: {np.mean(errors_pct):.2f}%")
    print(f"🎯 Erro mediano: {np.median(errors_pct):.2f}%")
    
    # 9. Intervalo de confiança
    if confidence_data:
        print(f"\n🔒 === INTERVALOS DE CONFIANÇA ===")
        print(f"🎯 Desvio padrão médio entre árvores: {np.mean(confidence_data['std']):.6f}")
        print(f"🎯 Variabilidade das previsões: {np.mean(confidence_data['std'])/np.mean(confidence_data['mean'])*100:.2f}%")
        
        # Analisa quantas previsões reais caem dentro dos intervalos
        last_100_real = y_test_norm[-100:]
        in_95_interval = np.mean(
            (last_100_real >= confidence_data['confidence_95_lower']) & 
            (last_100_real <= confidence_data['confidence_95_upper'])
        ) * 100
        
        in_68_interval = np.mean(
            (last_100_real >= confidence_data['confidence_68_lower']) & 
            (last_100_real <= confidence_data['confidence_68_upper'])
        ) * 100
        
        print(f"🎯 Valores reais dentro do intervalo 95%: {in_95_interval:.1f}%")
        print(f"🎯 Valores reais dentro do intervalo 68%: {in_68_interval:.1f}%")
    
    # 10. Informações do modelo
    print(f"\n🌳 === INFORMAÇÕES DO MODELO ===")
    if hasattr(model, 'n_estimators'):
        print(f"🌳 Número de árvores: {model.n_estimators}")
    if hasattr(model, 'max_depth'):
        print(f"📏 Profundidade máxima: {model.max_depth}")
    if hasattr(model, 'oob_score_'):
        print(f"🎯 OOB Score: {model.oob_score_:.4f} ({model.oob_score_*100:.2f}%)")
    
    # 11. Resumo da confiança
    print(f"\n🏆 === RESUMO DE CONFIANÇA ===")
    
    confianca_geral = (r2_norm * 0.4 + (direction_accuracy/100) * 0.3 + 
                      (np.mean(errors_pct < 2)/100) * 0.3) * 100
    
    print(f"🎯 CONFIANÇA GERAL DO MODELO: {confianca_geral:.1f}%")
    print(f"   • R² (explicação da variância): {r2_norm*100:.1f}%")
    print(f"   • Precisão direcional: {direction_accuracy:.1f}%") 
    print(f"   • Previsões com erro < 2%: {np.mean(errors_pct < 2)*100:.1f}%")
    
    if confianca_geral >= 85:
        nivel = "🟢 EXCELENTE"
    elif confianca_geral >= 75:
        nivel = "🟡 BOM"
    elif confianca_geral >= 65:
        nivel = "🟠 RAZOÁVEL"
    else:
        nivel = "🔴 BAIXO"
    
    print(f"\n📊 NÍVEL DE CONFIANÇA: {nivel}")
    
    # 12. Recomendações de uso
    print(f"\n💡 === RECOMENDAÇÕES DE USO ===")
    
    if confianca_geral >= 80:
        print("✅ Modelo altamente confiável para:")
        print("   • Previsões de curto prazo (1-6 horas)")
        print("   • Identificação de tendências")
        print("   • Suporte a decisões de trading")
    elif confianca_geral >= 70:
        print("⚠️ Modelo moderadamente confiável para:")
        print("   • Análise de tendências gerais")
        print("   • Complemento a outras análises")
        print("   • Alertas de mudanças de padrão")
    else:
        print("❌ Modelo com baixa confiança:")
        print("   • Use apenas como referência")
        print("   • Combine com outras fontes")
        print("   • Considere retreinar com mais dados")
    
    return {
        'r2_score': r2_norm,
        'mae_normalized': mae_norm,
        'mape': mape_norm,
        'direction_accuracy': direction_accuracy,
        'confidence_score': confianca_geral,
        'error_distribution': {
            'under_1pct': np.mean(errors_pct < 1)*100,
            'under_2pct': np.mean(errors_pct < 2)*100,
            'under_5pct': np.mean(errors_pct < 5)*100
        }
    }

def main():
    """Função principal para análise de precisão"""
    print("🎯 === ANÁLISE DE PRECISÃO E CONFIANÇA DO MODELO ===")
    
    # Carrega modelo
    model_path = "models/best_rf_parallel.pkl"
    model = load_saved_model(model_path)
    if model is None:
        return
    
    # Executa análise completa
    results = analyze_model_precision(model)
    
    print(f"\n✅ Análise de precisão concluída!")

if __name__ == "__main__":
    main()

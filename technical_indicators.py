# technical_indicators.py - Indicadores técnicos avançados

import pandas as pd
import numpy as np

def add_technical_indicators(df, config):
    """
    Adiciona indicadores técnicos avançados ao DataFrame
    """
    # RSI (Relative Strength Index)
    if config['rsi']['enabled']:
        df['rsi'] = calculate_rsi(df['close'], config['rsi']['window'])
    
    # Bollinger Bands
    if config['bollinger_bands']['enabled']:
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            df['close'], 
            config['bollinger_bands']['window'],
            config['bollinger_bands']['std_dev']
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # MACD
    if config['macd']['enabled']:
        macd_line, macd_signal, macd_histogram = calculate_macd(
            df['close'],
            config['macd']['fast'],
            config['macd']['slow'],
            config['macd']['signal']
        )
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
    
    # Stochastic Oscillator
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
    
    # Williams %R
    df['williams_r'] = calculate_williams_r(df, 14)
    
    # Average True Range (ATR)
    df['atr'] = calculate_atr(df, 14)
    
    # Commodity Channel Index (CCI)
    df['cci'] = calculate_cci(df, 20)
    
    # Rate of Change (ROC)
    df['roc'] = calculate_roc(df['close'], 10)
    
    # Money Flow Index (MFI)
    if 'volume' in df.columns:
        df['mfi'] = calculate_mfi(df, 14)
    
    # Volume indicators
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
    
    return df

def calculate_rsi(prices, window=14):
    """Calcula o RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calcula as Bandas de Bollinger"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcula o MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calculate_stochastic(df, k_window=14, d_window=3):
    """Calcula o Oscilador Estocástico"""
    lowest_low = df['low'].rolling(window=k_window).min()
    highest_high = df['high'].rolling(window=k_window).max()
    stoch_k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def calculate_williams_r(df, window=14):
    """Calcula o Williams %R"""
    highest_high = df['high'].rolling(window=window).max()
    lowest_low = df['low'].rolling(window=window).min()
    williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    return williams_r

def calculate_atr(df, window=14):
    """Calcula o Average True Range"""
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift())
    low_close_prev = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_cci(df, window=20):
    """Calcula o Commodity Channel Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci

def calculate_roc(prices, window=10):
    """Calcula o Rate of Change"""
    roc = ((prices - prices.shift(window)) / prices.shift(window)) * 100
    return roc

def calculate_mfi(df, window=14):
    """Calcula o Money Flow Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    
    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))
    return mfi

def add_lagged_features(df, columns, lags=[1, 2, 3, 5]):
    """Adiciona features com lag temporal"""
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def add_rolling_statistics(df, columns, windows=[5, 10, 20]):
    """Adiciona estatísticas rolantes"""
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f"{col}_mean_{window}"] = df[col].rolling(window=window).mean()
                df[f"{col}_std_{window}"] = df[col].rolling(window=window).std()
                df[f"{col}_min_{window}"] = df[col].rolling(window=window).min()
                df[f"{col}_max_{window}"] = df[col].rolling(window=window).max()
    return df

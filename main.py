# projeto_previsao.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(file_path, seq_length, test_size):
    """
    Carrega e pré-processa os dados do CSV
    """
    # Carrega os dados
    df = pd.read_csv(file_path, delimiter=';')
    
    # Seleciona apenas a coluna 'close' para previsão
    data = df['close'].values.astype(float)
    
    # Normaliza os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    # Cria sequências para o modelo
    X, y = [], []
    for i in range(seq_length, len(data_scaled)):
        X.append(data_scaled[i-seq_length:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Redimensiona X para ser compatível com LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Divide os dados em treino e teste
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Pega a última sequência para fazer previsão do próximo valor
    last_sequence = data_scaled[-seq_length:]
    
    return X_train, X_test, y_train, y_test, scaler, last_sequence


def calculate_rmse(y_true, y_pred):
    """
    Calcula o Root Mean Square Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_predictions(y_true, y_pred, title="Previsões"):
    """
    Plota as previsões vs valores reais
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Valores Reais', color='blue')
    plt.plot(y_pred, label='Previsões', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Tempo')
    plt.ylabel('Valor Close')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Treina o modelo LSTM
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Arquitetura do modelo LSTM:")
    model.summary()
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test), 
        verbose=1
    )
    
    return model


def train_random_forest(X_train, y_train):
    """
    Treina o modelo Random Forest
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train.ravel())
    return rf


# Código principal
if __name__ == "__main__":
    SEQ_LENGTH = 5
    TEST_SIZE = 0.2
    FILE_PATH = 'relatorio_mensal_geral_2025-03 (1).csv'

    # Carrega e pré-processa os dados
    X_train, X_test, y_train, y_test, scaler, last_sequence = load_and_preprocess_data(FILE_PATH, SEQ_LENGTH, TEST_SIZE)

    print("Treinando modelo LSTM...")
    lstm_model = train_lstm(X_train, y_train, X_test, y_test)

    # Faz previsões com LSTM
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    print("Treinando modelo Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    # Faz previsões com Random Forest
    rf_predictions = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
    rf_predictions_rescaled = scaler.inverse_transform(rf_predictions.reshape(-1, 1))

    # Calcula métricas de desempenho
    print("\nMétricas de desempenho:")
    print(f"LSTM RMSE: {calculate_rmse(y_test_rescaled, lstm_predictions_rescaled):.4f}")
    print(f"Random Forest RMSE: {calculate_rmse(y_test_rescaled, rf_predictions_rescaled):.4f}")

    # Previsão do próximo valor
    next_input = last_sequence.reshape(1, SEQ_LENGTH, 1)
    next_pred_lstm = scaler.inverse_transform(lstm_model.predict(next_input))
    next_pred_rf = scaler.inverse_transform(rf_model.predict(last_sequence.reshape(1, -1)).reshape(-1, 1))

    print(f"\nPrevisão do próximo bloco:")
    print(f"LSTM: {next_pred_lstm[0][0]:.2f}")
    print(f"Random Forest: {next_pred_rf[0][0]:.2f}")

    # Gera gráficos
    plot_predictions(y_test_rescaled, lstm_predictions_rescaled, title="Previsão com LSTM")
    plot_predictions(y_test_rescaled, rf_predictions_rescaled, title="Previsão com Random Forest")
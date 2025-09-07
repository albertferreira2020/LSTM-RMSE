# projeto_previsao.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(file_path, seq_length, test_size, use_multiple_features=True):
    """
    Carrega e pré-processa os dados do CSV com recursos aprimorados
    """
    # Carrega os dados
    df = pd.read_csv(file_path, delimiter=';')
    
    if use_multiple_features:
        # Usa múltiplas features para melhor previsão
        feature_columns = ['close', 'open', 'high', 'low', 'volume']
        # Verifica se as colunas existem
        available_columns = [col for col in feature_columns if col in df.columns]
        print(f"Usando features: {available_columns}")
        
        # Cria features técnicas adicionais
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['high'] - df['low']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Médias móveis
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_ratio'] = df['ma_5'] / df['ma_10']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Adiciona features técnicas à lista
        technical_features = ['price_change', 'volatility', 'price_position', 'ma_ratio', 'rsi']
        all_features = available_columns + technical_features
        
        # Remove NaN e prepara dados
        df = df.dropna()
        data = df[all_features].values.astype(float)
        
        # Normaliza cada feature separadamente
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        n_features = len(all_features)
    else:
        # Usa apenas close (modo original)
        data = df['close'].values.astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data.reshape(-1, 1))
        n_features = 1
    
    # Cria sequências para o modelo
    X, y = [], []
    for i in range(seq_length, len(data_scaled)):
        if use_multiple_features:
            X.append(data_scaled[i-seq_length:i, :])  # Todas as features
            y.append(data_scaled[i, 0])  # Prediz apenas close (primeira coluna)
        else:
            X.append(data_scaled[i-seq_length:i, 0])
            y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Redimensiona X para ser compatível com LSTM
    if not use_multiple_features:
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Divide os dados em treino e teste
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Pega a última sequência para fazer previsão do próximo valor
    last_sequence = data_scaled[-seq_length:]
    
    return X_train, X_test, y_train, y_test, scaler, last_sequence, n_features


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


def train_lstm_advanced(X_train, y_train, X_test, y_test, n_features, epochs=100, batch_size=32):
    """
    Treina o modelo LSTM com arquitetura aprimorada
    """
    model = Sequential([
        # Primeira camada LSTM
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], n_features)),
        Dropout(0.3),
        
        # Segunda camada LSTM
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        
        # Terceira camada LSTM
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        
        # Quarta camada LSTM
        LSTM(50),
        Dropout(0.2),
        
        # Camadas densas
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    # Otimizador personalizado
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    
    print("Arquitetura do modelo LSTM aprimorado:")
    model.summary()
    
    # Callbacks para melhor treinamento
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.0001,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history


def train_random_forest_advanced(X_train, y_train):
    """
    Treina o modelo Random Forest com hiperparâmetros otimizados
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import scipy.stats as stats
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Hiperparâmetros para otimização
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # RandomizedSearchCV para encontrar melhores hiperparâmetros
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=50,  # Número de combinações para testar
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    print("Otimizando hiperparâmetros do Random Forest...")
    rf_random.fit(X_train_flat, y_train.ravel())
    
    print(f"Melhores parâmetros: {rf_random.best_params_}")
    
    return rf_random.best_estimator_


def ensemble_prediction(lstm_pred, rf_pred, weights=[0.7, 0.3]):
    """
    Combina previsões usando ensemble weighted
    """
    return weights[0] * lstm_pred + weights[1] * rf_pred


def plot_training_history(history):
    """
    Plota o histórico de treinamento do modelo
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Learning rate (se disponível)
    if 'lr' in history.history:
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_additional_metrics(y_true, y_pred):
    """
    Calcula métricas adicionais de avaliação
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    
    rmse = calculate_rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }


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
    SEQ_LENGTH = 10  # Aumentado de 5 para 10
    TEST_SIZE = 0.2
    FILE_PATH = 'relatorio_mensal_geral_2025-03 (1).csv'
    USE_MULTIPLE_FEATURES = True  # Nova opção para usar múltiplas features

    print("=== Projeto de Previsão Avançado ===")
    print(f"Sequência: {SEQ_LENGTH}, Test Size: {TEST_SIZE}")
    print(f"Múltiplas features: {USE_MULTIPLE_FEATURES}")

    # Carrega e pré-processa os dados
    X_train, X_test, y_train, y_test, scaler, last_sequence, n_features = load_and_preprocess_data(
        FILE_PATH, SEQ_LENGTH, TEST_SIZE, USE_MULTIPLE_FEATURES
    )

    print(f"\nDados carregados:")
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"Features: {n_features}")

    print("\n=== Treinando modelo LSTM avançado ===")
    lstm_model, history = train_lstm_advanced(X_train, y_train, X_test, y_test, n_features)

    # Faz previsões com LSTM
    lstm_predictions = lstm_model.predict(X_test)
    if USE_MULTIPLE_FEATURES:
        # Para múltiplas features, criamos um array com zeros e colocamos a previsão na primeira posição
        lstm_pred_full = np.zeros((len(lstm_predictions), n_features))
        lstm_pred_full[:, 0] = lstm_predictions.flatten()
        lstm_predictions_rescaled = scaler.inverse_transform(lstm_pred_full)[:, 0].reshape(-1, 1)
        
        y_test_full = np.zeros((len(y_test), n_features))
        y_test_full[:, 0] = y_test
        y_test_rescaled = scaler.inverse_transform(y_test_full)[:, 0].reshape(-1, 1)
    else:
        lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    print("\n=== Treinando modelo Random Forest avançado ===")
    rf_model = train_random_forest_advanced(X_train, y_train)

    # Faz previsões com Random Forest
    rf_predictions = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
    if USE_MULTIPLE_FEATURES:
        rf_pred_full = np.zeros((len(rf_predictions), n_features))
        rf_pred_full[:, 0] = rf_predictions
        rf_predictions_rescaled = scaler.inverse_transform(rf_pred_full)[:, 0].reshape(-1, 1)
    else:
        rf_predictions_rescaled = scaler.inverse_transform(rf_predictions.reshape(-1, 1))

    # Ensemble prediction
    ensemble_predictions = ensemble_prediction(
        lstm_predictions_rescaled.flatten(), 
        rf_predictions_rescaled.flatten()
    ).reshape(-1, 1)

    # Calcula métricas avançadas
    print("\n=== Métricas de Desempenho Detalhadas ===")
    
    lstm_metrics = calculate_additional_metrics(y_test_rescaled, lstm_predictions_rescaled)
    rf_metrics = calculate_additional_metrics(y_test_rescaled, rf_predictions_rescaled)
    ensemble_metrics = calculate_additional_metrics(y_test_rescaled, ensemble_predictions)
    
    print("LSTM:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nRandom Forest:")
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nEnsemble:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Previsão do próximo valor
    if USE_MULTIPLE_FEATURES:
        next_input = last_sequence.reshape(1, SEQ_LENGTH, n_features)
        next_pred_lstm_raw = lstm_model.predict(next_input)
        next_pred_lstm_full = np.zeros((1, n_features))
        next_pred_lstm_full[0, 0] = next_pred_lstm_raw[0, 0]
        next_pred_lstm = scaler.inverse_transform(next_pred_lstm_full)[0, 0]
        
        next_pred_rf_raw = rf_model.predict(last_sequence.reshape(1, -1))
        next_pred_rf_full = np.zeros((1, n_features))
        next_pred_rf_full[0, 0] = next_pred_rf_raw[0]
        next_pred_rf = scaler.inverse_transform(next_pred_rf_full)[0, 0]
    else:
        next_input = last_sequence.reshape(1, SEQ_LENGTH, 1)
        next_pred_lstm = scaler.inverse_transform(lstm_model.predict(next_input))[0, 0]
        next_pred_rf = scaler.inverse_transform(rf_model.predict(last_sequence.reshape(1, -1)).reshape(-1, 1))[0, 0]
    
    next_pred_ensemble = ensemble_prediction(next_pred_lstm, next_pred_rf, [0.7, 0.3])

    print(f"\n=== Previsão do Próximo Bloco ===")
    print(f"LSTM: {next_pred_lstm:.2f}")
    print(f"Random Forest: {next_pred_rf:.2f}")
    print(f"Ensemble (70% LSTM + 30% RF): {next_pred_ensemble:.2f}")

    # Gera gráficos avançados
    plot_training_history(history)
    plot_predictions(y_test_rescaled, lstm_predictions_rescaled, title="Previsão com LSTM Avançado")
    plot_predictions(y_test_rescaled, rf_predictions_rescaled, title="Previsão com Random Forest Avançado")
    plot_predictions(y_test_rescaled, ensemble_predictions, title="Previsão com Ensemble")
    
    # Gráfico comparativo
    plt.figure(figsize=(15, 8))
    plt.plot(y_test_rescaled, label='Valores Reais', color='blue', linewidth=2)
    plt.plot(lstm_predictions_rescaled, label='LSTM', color='red', alpha=0.8)
    plt.plot(rf_predictions_rescaled, label='Random Forest', color='green', alpha=0.8)
    plt.plot(ensemble_predictions, label='Ensemble', color='purple', alpha=0.8, linewidth=2)
    plt.title('Comparação de Todos os Modelos')
    plt.xlabel('Tempo')
    plt.ylabel('Valor Close')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
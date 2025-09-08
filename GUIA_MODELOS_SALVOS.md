# 📖 Guia de Uso dos Modelos Salvos

## 🎯 O que é salvo automaticamente?

Quando o treinamento termina, o sistema salva automaticamente:

### 📁 Arquivos Gerados na pasta `models/`:

1. **`best_lstm_model.h5`** - Modelo LSTM (rede neural) completo
2. **`rf_model.pkl`** - Modelo Random Forest otimizado
3. **`gb_model.pkl`** - Modelo Gradient Boosting otimizado
4. **`scaler.pkl`** - Normalizador (ESSENCIAL para previsões)
5. **`model_metadata.json`** - Metadados (features, configurações)
6. **`load_models.py`** - Script automático para carregar modelos

---

## 🚀 Como usar os modelos salvos

### Método 1: Usando as funções do main_advanced.py

```python
from main_advanced import load_saved_models, predict_with_saved_models
import pandas as pd

# Carrega todos os modelos
modelos = load_saved_models('models')

# Seus dados novos (mesmo formato do treinamento)
dados_novos = pd.DataFrame({
    'open': [100.0, 101.0],
    'high': [102.0, 103.0],
    'low': [99.0, 100.0],
    'close': [101.0, 102.0],
    'volume': [1000, 1100]
})

# Faz previsões
previsoes = predict_with_saved_models(dados_novos, 'models')
print(previsoes)
# Output: {'RF': 103.45, 'GB': 102.88, 'LSTM': 103.12, 'ENSEMBLE': 103.15}
```

### Método 2: Carregamento manual

```python
import joblib
import json
from tensorflow.keras.models import load_model

# Carrega metadados
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Carrega scaler (OBRIGATÓRIO)
scaler = joblib.load('models/scaler.pkl')

# Carrega modelos individuais
rf_model = joblib.load('models/rf_model.pkl')
gb_model = joblib.load('models/gb_model.pkl')
lstm_model = load_model('models/best_lstm_model.h5')
```

### Método 3: Script automático

```bash
# Execute o script gerado automaticamente
cd models
python load_models.py
```

---

## 📊 Exemplo Completo de Uso

```python
# exemplo_uso.py
from main_advanced import predict_with_saved_models
import pandas as pd

# Simula dados de mercado recentes
dados_mercado = pd.DataFrame({
    'open': [50100, 50200, 50150, 50250, 50300],
    'high': [50200, 50300, 50250, 50350, 50400],
    'low': [50000, 50100, 50050, 50150, 50200],
    'close': [50150, 50180, 50200, 50280, 50350],
    'volume': [1000000, 1100000, 950000, 1200000, 1300000]
})

print("📊 Dados de entrada:")
print(dados_mercado.tail())

# Faz previsão do próximo valor
previsoes = predict_with_saved_models(dados_mercado)

print("\n🔮 Previsões do próximo valor:")
ultimo_preco = dados_mercado['close'].iloc[-1]

for modelo, predicao in previsoes.items():
    variacao = ((predicao - ultimo_preco) / ultimo_preco) * 100
    print(f"{modelo}: {predicao:.2f} ({variacao:+.2f}%)")
```

---

## 🔄 Integração com Sistemas de Produção

### 1. API Flask Simples

```python
from flask import Flask, request, jsonify
from main_advanced import load_saved_models, predict_with_saved_models

app = Flask(__name__)
modelos = load_saved_models('models')  # Carrega uma vez

@app.route('/predict', methods=['POST'])
def predict():
    dados = request.json
    df = pd.DataFrame(dados)
    previsoes = predict_with_saved_models(df)
    return jsonify(previsoes)

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. Monitoramento Contínuo

```python
import schedule
import time

def monitorar_mercado():
    # Carrega dados mais recentes (API, banco, etc.)
    dados = carregar_dados_atuais()

    # Faz previsão
    previsoes = predict_with_saved_models(dados)

    # Processa resultados (alertas, emails, etc.)
    if previsoes['ENSEMBLE'] > limite_superior:
        enviar_alerta("Alta prevista!")

# Roda a cada hora
schedule.every().hour.do(monitorar_mercado)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## ⚙️ Requisitos para Usar os Modelos

### Estrutura dos Dados de Entrada:

Os dados devem ter **pelo menos** estas colunas:

- `open`, `high`, `low`, `close`
- `volume` (opcional)

### Indicadores Técnicos:

O sistema adiciona automaticamente:

- RSI, MACD, Bollinger Bands
- Médias móveis (7, 14, 21 períodos)
- Features com lag (1, 2, 3, 5 períodos)
- Estatísticas rolantes (5, 10, 20 janelas)

### Quantidade Mínima:

- Pelo menos **30 pontos** de dados históricos para calcular indicadores
- **SEQ_LENGTH** pontos para previsões LSTM (padrão: 60)

---

## 🚨 Troubleshooting

### Erro: "Features faltando"

**Solução**: O sistema adiciona automaticamente os indicadores técnicos

### Erro: "Scaler não encontrado"

**Solução**: Execute o treinamento completo primeiro

### Erro: "TensorFlow não disponível"

**Solução**: Instale TensorFlow ou use apenas RF/GB:

```bash
pip install tensorflow
```

### Erro: "Dados insuficientes"

**Solução**: Forneça pelo menos 60 pontos históricos

---

## 📈 Interpretação dos Resultados

```python
previsoes = {
    'RF': 103.45,        # Random Forest
    'GB': 102.88,        # Gradient Boosting
    'LSTM': 103.12,      # Rede Neural LSTM
    'ENSEMBLE': 103.15   # Média dos 3 modelos
}

# ENSEMBLE geralmente é a previsão mais confiável
proxima_predicao = previsoes['ENSEMBLE']
```

### Confiabilidade por Modelo:

- **ENSEMBLE**: ⭐⭐⭐⭐⭐ (Recomendado)
- **LSTM**: ⭐⭐⭐⭐ (Bom para tendências)
- **RF**: ⭐⭐⭐⭐ (Estável, interpretável)
- **GB**: ⭐⭐⭐ (Rápido, menos overfitting)

---

## 🔧 Personalizações

### Retreinar apenas um modelo:

```python
# Carrega modelos existentes
modelos = load_saved_models('models')

# Retreina apenas o Random Forest com novos dados
new_rf = RandomForestRegressor(**melhores_params)
new_rf.fit(X_novo, y_novo)

# Salva o modelo atualizado
joblib.dump(new_rf, 'models/rf_model.pkl')
```

### Adicionar novos modelos:

```python
# Treina novo modelo (ex: XGBoost)
import xgboost as xgb
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Adiciona ao ensemble
joblib.dump(xgb_model, 'models/xgb_model.pkl')
```

---

## 📞 Suporte

Para dúvidas ou problemas:

1. Verifique se todos os arquivos estão na pasta `models/`
2. Confirme que o treinamento foi concluído com sucesso
3. Teste com `use_saved_models.py` primeiro
4. Consulte os logs de erro para diagnóstico

**Dica**: Execute `python use_saved_models.py` para um exemplo completo
funcionando!

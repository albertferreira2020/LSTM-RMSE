# Instruções de Uso - LSTM Simplificado

## **Execução Rápida**

1. **Instalar dependências:**

   ```bash
   pip install -r requirements-stable.txt
   ```

2. **Executar o projeto:**
   ```bash
   python main_advanced.py
   ```

## **Arquivos Principais**

- 📊 `main_advanced.py` - **EXECUTAR ESTE ARQUIVO**
- ⚙️ `config.py` - Configurações do modelo
- 📈 `technical_indicators.py` - Indicadores técnicos
- 🗄️ `database.py` - Conexão PostgreSQL (opcional)

## **Testes**

- `python test_tensorflow_fix.py` - Testar TensorFlow
- `python test_database.py` - Testar banco PostgreSQL

## **Dados**

- O arquivo `relatorio_mensal_geral_2025-03 (1).csv` contém dados de exemplo
- O modelo `best_lstm_model.h5` é um modelo pré-treinado

## **Configuração PostgreSQL (Opcional)**

Se quiser usar banco PostgreSQL, configure no arquivo `.env`:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=seu_banco
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
```

**⚠️ IMPORTANTE:** O projeto funciona sem banco de dados, usando apenas arquivos
CSV.

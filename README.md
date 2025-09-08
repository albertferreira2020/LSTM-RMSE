# Previsão de Valor de Close com LSTM e Random Forest

Este projeto utiliza redes neurais LSTM (TensorFlow/Keras) e Random Forest
(Scikit-learn) para prever o valor de **close** do próximo bloco, com base em
dados históricos (blocos Renko).

## **Configuração do Banco PostgreSQL (Nova Funcionalidade)**

### **1. Configuração das Credenciais**

1. Copie o arquivo de exemplo:

   ```bash
   cp .env.example .env
   ```

2. Edite o arquivo `.env` com suas credenciais:
   ```bash
   # Exemplo de configuração
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=seu_banco
   DB_USER=seu_usuario
   DB_PASSWORD=sua_senha
   ```

### **2. Teste a Conexão**

Antes de executar o sistema principal, teste a conexão criando um script
simples:

```python
# test_connection.py
from database import DatabaseManager

db = DatabaseManager()
if db.connect():
    print("✅ Conexão com PostgreSQL bem-sucedida!")
    db.get_table_info('botbinance')
    db.disconnect()
else:
    print("❌ Falha na conexão com PostgreSQL")
```

Execute:

```bash
python test_connection.py
```

### **3. Estrutura da Tabela**

O sistema espera uma tabela `botbinance` com as colunas:

- `id` - Identificador único
- `created_at` - Timestamp
- `open` - Preço de abertura
- `close` - Preço de fechamento
- `high` - Preço máximo
- `low` - Preço mínimo
- `volume` - Volume (opcional)

## **Configuração do Ambiente**

### **1. Criação do Ambiente Virtual**

⚠️ **Importante: Este projeto requer Python 3.10 ou superior** devido às
dependências avançadas de análise técnica.

Para macOS/Linux:

```bash
# Verificar versão do Python (deve ser 3.10+)
python3 --version
python3.11 --version  # Se disponível

# Criar ambiente virtual com Python 3.11 (recomendado)
python3.11 -m venv env

# Ou com Python 3.10 (mínimo)
python3.10 -m venv env

# Ativar ambiente virtual
source env/bin/activate
```

Para Windows:

```bash
# Verificar versão do Python (deve ser 3.10+)
python3 --version

# Criar ambiente virtual
python3 -m venv env

# Ativar ambiente virtual
source env/bin/activate
```

### **2. Instalação das Dependências**

Com o ambiente virtual ativado, você tem duas opções:

**Opção 1: Instalação automática (recomendada)**

```bash
# Atualizar pip primeiro
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

**Opção 2: Instalação manual**

```bash
# Atualizar pip primeiro
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

**Em caso de erro no TensorFlow (OSError):**

```bash
# Limpar cache do pip
pip cache purge

# Instalar pacotes individualmente
pip install pandas numpy scikit-learn matplotlib
pip install tensorflow

# Ou usar conda (alternativa recomendada)
conda install pandas numpy scikit-learn matplotlib
conda install tensorflow
```

### **3. Verificação da Instalação**

Para verificar se tudo foi instalado corretamente:

```bash
# Verificação manual das dependências
python3 -c "import tensorflow, pandas, numpy, sklearn, matplotlib; print('Todas as bibliotecas foram instaladas com sucesso!')"

# Verificação individual
python3 -c "import tensorflow; print('TensorFlow OK')"
python3 -c "import pandas; print('Pandas OK')"
python3 -c "import numpy; print('NumPy OK')"
python3 -c "import sklearn; print('Scikit-learn OK')"
python3 -c "import matplotlib; print('Matplotlib OK')"
python3 -c "import ta; print('TA (Technical Analysis) OK')"
```

## **Execução**

### **Versão Básica:**

1. Certifique-se de que o ambiente virtual está ativado:

   ```bash
   source env/bin/activate  # macOS/Linux
   # ou
   env\Scripts\activate     # Windows
   ```

2. Execute o script básico:
   ```bash
   python3 main.py
   ```

### **Versão Avançada (Recomendada para Maior Precisão):**

**Com PostgreSQL (Recomendado):**

```bash
# 1. Configure o banco no arquivo .env
cp .env.example .env
# Edite .env com suas credenciais

# 2. Teste a conexão
python3 test_database.py

# 3. Execute o sistema
python3 main_advanced.py
```

**Com CSV (Fallback):**

```bash
python3 main_advanced.py
# O sistema tentará PostgreSQL primeiro, depois CSV se falhar
```

### **Arquivos Disponíveis:**

- `main.py` - Versão básica e funcional
- `main_advanced.py` - Versão com todas as melhorias + PostgreSQL
- `config.py` - Configurações centralizadas
- `technical_indicators.py` - Indicadores técnicos avançados
- `database.py` - Gerenciador de conexão PostgreSQL
- `test_database.py` - Script para testar conexão com banco

## **Estrutura do Projeto**

```
perceptronLTSM/
├── main.py                                    # Script básico original
├── main_advanced.py                           # Script avançado com PostgreSQL
├── config.py                                  # Configurações centralizadas
├── config_deep_training.py                   # Configurações de treino profundo
├── hyperparameters_optimized.py              # Hiperparâmetros otimizados
├── technical_indicators.py                   # Indicadores técnicos avançados
├── database.py                               # Gerenciador PostgreSQL
├── requirements.txt                           # Dependências do projeto
├── .env                                      # Credenciais do banco (criar) - NÃO commitado
├── .env.example                              # Exemplo de configuração - Commitado
├── .gitignore                                # Configuração Git - Commitado
├── README.md                                 # Este arquivo
├── models/                                   # Modelos treinados (.keras)
├── relatorio_mensal_geral_2025-03 (1).csv   # Dados de entrada (fallback)
└── env/                                      # Ambiente virtual - NÃO commitado
```

**Legenda:**

- ✅ **Commitado:** Arquivos versionados no Git
- ❌ **NÃO commitado:** Arquivos ignorados pelo .gitignore

## **O que o script faz:**

### **Versão Básica (main.py):**

✅ **Carregamento e Pré-processamento:**

- Carrega dados do arquivo CSV
- Normaliza os valores usando MinMaxScaler
- Cria sequências temporais para treino

✅ **Treinamento de Modelos:**

- **LSTM:** Rede neural recorrente para séries temporais
- **Random Forest:** Algoritmo de ensemble para comparação

✅ **Avaliação:**

- Calcula RMSE (Root Mean Square Error) para ambos os modelos
- Compara performance entre LSTM e Random Forest

### **Versão Avançada (main_advanced.py):**

🚀 **Recursos Aprimorados para Maior Precisão:**

✅ **Features Técnicas Avançadas:**

- RSI (Relative Strength Index)
- Bandas de Bollinger
- MACD (Moving Average Convergence Divergence)
- Oscilador Estocástico
- Williams %R
- ATR (Average True Range)
- CCI (Commodity Channel Index)
- MFI (Money Flow Index)
- Features com lag temporal
- Estatísticas rolantes

✅ **Arquitetura LSTM Aprimorada:**

- 4 camadas LSTM com 100→100→50→50 neurônios
- BatchNormalization para estabilidade
- Dropout adaptativo (0.3→0.3→0.2→0.2)
- Camadas densas adicionais (50→25)
- Early Stopping e ReduceLROnPlateau
- Checkpoint do melhor modelo

✅ **Ensemble de Modelos:**

- Random Forest otimizado com RandomizedSearchCV
- Gradient Boosting Regressor
- Ensemble weightedcombinando todos os modelos
- Cross-validation com TimeSeriesSplit

✅ **Métricas Abrangentes:**

- RMSE, MAE, R², MAPE
- Análise de resíduos
- Gráficos comparativos detalhados
- Distribuição de erros

✅ **Visualizações Avançadas:**

- Comparação de todos os modelos
- Histórico de treinamento
- Scatter plots Real vs Previsto
- Análise de resíduos
- Zoom nas últimas previsões

### **Configurações de Treino Profundo e Precisão Máxima**

#### **Novos Arquivos de Configuração**

O projeto agora inclui configurações otimizadas para máxima precisão:

1. **`config.py`** - Configurações gerais otimizadas
2. **`config_deep_training.py`** - Configurações específicas para treino
   profundo
3. **`hyperparameters_optimized.py`** - Hiperparâmetros otimizados por pesquisa
   bibliográfica

#### **Principais Otimizações Implementadas**

##### **🧠 LSTM Profundo**

- **Arquitetura**: 8 camadas com 512→384→256→192→128→96→64→32 neurônios
- **Bidirectional LSTM**: Captura dependências passadas e futuras
- **Attention Mechanism**: Foco nas partes importantes da sequência
- **Regularização**: L1/L2, Dropout progressivo, Batch Normalization
- **Sequência**: 120 timesteps (4 meses de dados diários)
- **Treino**: 1000 épocas com early stopping inteligente

##### **🌲 Random Forest Otimizado**

- **Estimadores**: 2000 árvores para máxima estabilidade
- **Profundidade**: 35 níveis para capturar complexidade
- **Features**: 80% das features por árvore
- **Amostragem**: 90% das amostras por árvore

##### **🚀 Gradient Boosting Avançado**

- **Estimadores**: 1500 com learning rate 0.01
- **Profundidade**: 12 níveis com regularização
- **Loss Function**: Huber loss (robusto a outliers)
- **Early Stopping**: 30 iterações de paciência

##### **⚡ XGBoost Otimizado**

- **Estimadores**: 1500 com regularização L1/L2
- **Tree Method**: Histogram para velocidade
- **Subsampling**: 85% para regularização
- **Profundidade**: 10 níveis otimizada

##### **🎯 Ensemble Stacking**

- **Pesos Dinâmicos**: LSTM 50%, RF 18%, GB 18%, XGB 14%
- **Meta-Learner**: Ridge Regression para stacking
- **Validação**: 7-fold cross-validation temporal
- **Adaptação**: Pesos adaptativos baseados em performance recente

#### **📈 Feature Engineering Avançado**

##### **Indicadores Técnicos (80+ features)**

- **Médias Móveis**: SMA, EMA, WMA, Hull, TEMA (10 períodos)
- **Momentum**: RSI, MACD, Stochastic, Williams %R, ROC
- **Volatilidade**: Bollinger Bands, ATR, Keltner, Donchian
- **Volume**: OBV, MFI, VWAP, A/D Line, CMF
- **Padrões**: Ichimoku, ADX, Parabolic SAR, Fibonacci

##### **Features Estatísticas**

- **Lags**: 1, 2, 3, 5, 8, 13, 21 períodos (Fibonacci)
- **Rolling Stats**: Média, Desvio, Assimetria, Curtose, Min, Max
- **Fourier**: 15 componentes para capturar ciclicidade
- **Wavelets**: Decomposição em 4 níveis

##### **Preprocessamento Robusto**

- **Scaling**: Robust Quantile (5-95 percentil)
- **Outliers**: Isolation Forest (3% contaminação)
- **Seleção**: Hybrid (Mutual Info + F-Regression + RFE)
- **Missing Values**: Iterative Imputer com Bayesian Ridge

#### **🔍 Validação Temporal Avançada**

##### **Cross-Validation Purged**

- **Método**: Time Series CV com embargo de 5 dias
- **Splits**: 7 folds para validação robusta
- **Gap**: 2 dias entre treino e teste
- **Teste**: 30 dias por fold

##### **Walk-Forward Analysis**

- **Janela**: 252 dias (1 ano) de treino
- **Step**: 21 dias (1 mês) por iteração
- **Refit**: Retreino mensal automático
- **Expanding**: Janela crescente de dados

#### **📊 Métricas Financeiras Especializadas**

##### **Métricas de Regressão**

- MSE, RMSE, MAE, MAPE, R²
- Explained Variance, Poisson/Gamma Deviance

##### **Métricas Financeiras**

- **Sharpe Ratio**: Retorno ajustado ao risco
- **Sortino Ratio**: Foco no downside risk
- **Calmar Ratio**: Retorno vs. max drawdown
- **Directional Accuracy**: Precisão da direção
- **Hit Ratio**: Taxa de acertos
- **Profit Factor**: Ganhos vs. perdas

##### **Testes Estatísticos**

- Ljung-Box (autocorrelação)
- ADF/KPSS (estacionaridade)
- Jarque-Bera (normalidade)
- Shapiro-Wilk, Anderson-Darling

#### **🎛️ Otimização Bayesiana**

##### **Optuna Integration**

- **Trials**: 500 tentativas de otimização
- **Timeout**: 4 horas de busca
- **Pruning**: Median pruner para eficiência
- **Search Space**: Log-uniform, categorical, uniform

##### **Hyperparameter Tuning**

- **LSTM**: Learning rate, batch size, layers, dropout
- **Tree Models**: Estimadores, profundidade, regularização
- **Ensemble**: Pesos, meta-learner, blending

#### **🖥️ Configuração de Hardware**

##### **GPU Acceleration**

- **Mixed Precision**: FP16 para velocidade
- **XLA Compilation**: Otimização de grafos
- **Memory Growth**: Alocação dinâmica
- **Multi-GPU**: Suporte para múltiplas GPUs

##### **CPU Optimization**

- **Parallelização**: Todos os cores disponíveis
- **Numba JIT**: Compilação just-in-time
- **Dask**: Processamento distribuído
- **Bottleneck**: NumPy acelerado

#### **Como Usar as Configurações Otimizadas**

1. **Configuração Básica (Rápida)**:

   ```bash
   python main_advanced.py
   ```

2. **Configuração Profunda (Máxima Precisão)**:

   ```python
   from config_deep_training import setup_tensorflow_config
   from hyperparameters_optimized import get_optimized_config

   # Configurar TensorFlow para performance
   setup_tensorflow_config()

   # Obter configuração otimizada
   lstm_config = get_optimized_config('lstm')
   ```

3. **Otimização Bayesiana**:

   ```python
   from hyperparameters_optimized import BAYESIAN_OPTIMIZATION_CONFIG

   # Ativar otimização automática
   config['OPTIMIZATION_CONFIG']['use_optuna'] = True
   ```

#### **Tempo de Treinamento Estimado**

| Configuração | CPU (8 cores) | GPU (RTX 3080) | Precisão Esperada |
| ------------ | ------------- | -------------- | ----------------- |
| Básica       | 2-4 horas     | 30-60 min      | R² > 0.85         |
| Profunda     | 8-12 horas    | 2-4 horas      | R² > 0.90         |
| Máxima       | 24-48 horas   | 6-12 horas     | R² > 0.95         |

#### **Monitoramento do Treinamento**

1. **TensorBoard**:

   ```bash
   tensorboard --logdir=logs/tensorboard
   ```

2. **Logs de Treinamento**:

   ```bash
   tail -f logs/training_log.csv
   ```

3. **Checkpoints**:
   - Modelos salvos automaticamente em `checkpoints/`
   - Melhor modelo restaurado ao final

## **Desativação do Ambiente**

Quando terminar de usar o projeto:

```bash
deactivate
```

## **Dependências**

- **pandas:** Manipulação de dados
- **numpy:** Operações numéricas
- **scikit-learn:** Algoritmos de machine learning
- **matplotlib:** Visualização de dados
- **tensorflow:** Deep learning (LSTM)
- **seaborn:** Visualizações avançadas
- **psycopg2-binary:** Conector PostgreSQL
- **sqlalchemy:** ORM para banco de dados
- **python-dotenv:** Gerenciamento de variáveis de ambiente

## **Formato dos Dados**

O arquivo CSV deve conter uma coluna 'close' com os valores de fechamento dos
blocos Renko. O script utiliza separador ';' por padrão.

## **Troubleshooting**

### **Erro de Instalação do TensorFlow**

Se você encontrar o erro `OSError: [Errno 2] No such file or directory`, tente:

1. **Solução rápida (para o erro específico que você encontrou):**

   ```bash
   pip cache purge
   pip install --no-cache-dir tensorflow==2.15.0
   ```

2. **Limpar e reinstalar:**

   ```bash
   pip cache purge
   pip uninstall tensorflow
   pip install tensorflow
   ```

3. **Usar arquivo de requirements estável:**

   ```bash
   pip install -r requirements-stable.txt
   ```

4. **Alternativa com conda:**
   ```bash
   # Criar ambiente com conda
   conda create -n lstm_env python=3.9
   conda activate lstm_env
   conda install pandas numpy scikit-learn matplotlib tensorflow
   ```

### **Problemas de Memória**

Se o modelo LSTM consumir muita memória:

- Reduza o `batch_size` no código (linha padrão: 32)
- Reduza o número de `epochs` (linha padrão: 50)
- Reduza o tamanho das camadas LSTM (linha padrão: 50)

### **Problemas com pandas-ta**

**pandas-ta atualmente não está disponível via pip:** A biblioteca pandas-ta não
está mais disponível nos repositórios PyPI padrão.

**Soluções disponíveis:**

1. **Usar bibliotecas alternativas (já incluídas):**

   ```bash
   # Já incluído no requirements.txt:
   # ta>=0.10.0 - Biblioteca de análise técnica alternativa
   # yfinance>=0.2.0 - Para dados financeiros
   # stockstats>=0.5.0 - Indicadores técnicos
   ```

2. **Instalar pandas-ta via git (opcional):**

   ```bash
   pip install git+https://github.com/twopirllc/pandas-ta.git
   ```

3. **Instalar TA-Lib manualmente (opcional):**

   ```bash
   # macOS
   brew install ta-lib
   pip install TA-Lib

   # Ubuntu/Debian
   sudo apt-get install libta-lib0-dev
   pip install TA-Lib
   ```

**Alternativas funcionais incluídas:**

- **ta:** Biblioteca leve de análise técnica compatível com pandas
- **stockstats:** Indicadores técnicos prontos para uso
- **yfinance:** Download de dados financeiros históricos

### **Problemas com PostgreSQL**

**Erro de conexão:**

```bash
# Verifica se PostgreSQL está rodando
sudo systemctl status postgresql  # Linux
brew services list | grep postgres  # macOS

# Testa conexão manual
psql -h localhost -U seu_usuario -d seu_banco
```

**Erro de permissão:**

```sql
-- No PostgreSQL, conceda permissões
GRANT SELECT ON botbinance TO seu_usuario;
GRANT USAGE ON SCHEMA public TO seu_usuario;
```

**Tabela não existe:**

```sql
-- Verifique se a tabela existe
\dt botbinance

-- Ou via SQL
SELECT table_name
FROM information_schema.tables
WHERE table_name = 'botbinance';
```

**Fallback automático:**

- Se PostgreSQL falhar, o sistema usa automaticamente o CSV
- Verifique os logs para identificar problemas
- Use `python3 test_database.py` para diagnóstico

### **Problemas com o CSV**

- Verifique se o arquivo CSV usa separador `;`
- Certifique-se de que existe uma coluna chamada `close`
- Remova linhas com valores NaN ou vazios

### **Problemas de Versão do Python**

**Se você receber erros sobre versões Python requeridas:**

1. **Verificar versão atual:**

   ```bash
   python3 --version
   python3.11 --version  # ou python3.10
   ```

2. **Instalar Python 3.11+ no macOS:**

   ```bash
   # Via Homebrew (recomendado)
   brew install python@3.11

   # Via pyenv
   curl https://pyenv.run | bash
   pyenv install 3.11.13
   pyenv global 3.11.13
   ```

3. **Recriar ambiente virtual com versão correta:**
   ```bash
   rm -rf env
   python3.11 -m venv env
   source env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**Versões testadas e compatíveis:**

- ✅ Python 3.11.13 (recomendado)
- ✅ Python 3.12.x
- ⚠️ Python 3.10.x (mínimo - algumas features limitadas)
- ❌ Python 3.9.x ou inferior (não compatível)

## **Versionamento com Git**

### **Inicialização do Repositório**

```bash
# Inicializar repositório Git
git init

# Adicionar arquivos (o .gitignore já está configurado)
git add .

# Primeiro commit
git commit -m "Initial commit: LSTM prediction system with PostgreSQL"

# Conectar com repositório remoto (opcional)
git remote add origin https://github.com/seu-usuario/seu-repositorio.git
git push -u origin main
```

### **Arquivos Versionados**

✅ **Incluídos no Git:**

- Código fonte Python
- Configurações do projeto
- Documentação
- Arquivo de exemplo (.env.example)
- Requirements e dependências

❌ **Ignorados pelo Git:**

- Credenciais reais (.env)
- Modelos treinados (_.h5, _.pkl)
- Dados CSV reais
- Gráficos gerados
- Cache Python
- Ambientes virtuais

### **Comandos Úteis**

```bash
# Ver status dos arquivos
git status

# Ver arquivos ignorados
git status --ignored

# Adicionar arquivo específico ignorado (se necessário)
git add -f arquivo_especifico.csv
```

### **Problemas de Permissão com Arquivos de Modelo**

**Se você receber erro "Permission denied" ao salvar modelos:**

```
PermissionError: [Errno 13] Unable to synchronously create file (unable to open file: name = 'best_lstm_model.h5', errno = 13, error message = 'Permission denied')
```

**Soluções:**

1. **Executar script de preparação do ambiente (recomendado):**

   ```bash
   python setup_environment.py
   ```

2. **Remover arquivos problemáticos manualmente:**

   ```bash
   rm -f best_lstm_model.h5
   # Se necessário:
   sudo rm -f best_lstm_model.h5
   ```

3. **Criar estrutura de diretórios:**

   ```bash
   mkdir -p models checkpoints logs plots
   ```

4. **Verificar permissões:**
   ```bash
   ls -la *.h5 *.keras  # Ver arquivos existentes
   chmod 644 *.h5       # Corrigir permissões se necessário
   ```

**Nota:** O projeto agora usa o formato `.keras` (mais moderno) em vez de `.h5`
(legado).

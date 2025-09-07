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

Antes de executar o sistema principal, teste a conexão:

```bash
python3 test_database.py
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

Para macOS/Linux:

```bash
# Criar ambiente virtual
python3 -m venv env

# Ativar ambiente virtual
source env/bin/activate
```

Para Windows:

```bash
# Criar ambiente virtual
python -m venv env

# Ativar ambiente virtual
env\Scripts\activate
```

### **2. Instalação das Dependências**

Com o ambiente virtual ativado, você tem duas opções:

**Opção 1: Instalação automática (recomendada)**

```bash
python setup_env.py
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
# Opção 1: Script de verificação simples (recomendado)
python3 check_install.py

# Opção 2: Para zsh (macOS padrão)
python3 -c "import tensorflow, pandas, numpy, sklearn, matplotlib; print(\"Todas as bibliotecas foram instaladas com sucesso!\")"

# Opção 3: Script de configuração completa
python3 setup_env.py

# Opção 4: Verificação individual
python3 -c "import tensorflow; print('TensorFlow OK')"
python3 -c "import pandas; print('Pandas OK')"
python3 -c "import numpy; print('NumPy OK')"
python3 -c "import sklearn; print('Scikit-learn OK')"
python3 -c "import matplotlib; print('Matplotlib OK')"
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
├── technical_indicators.py                    # Indicadores técnicos avançados
├── database.py                               # Gerenciador PostgreSQL
├── test_database.py                          # Teste de conexão com banco
├── setup_env.py                              # Script de configuração automática
├── check_install.py                          # Script de verificação simples
├── requirements.txt                           # Dependências do projeto
├── requirements-stable.txt                    # Versões estáveis (backup)
├── .env                                      # Credenciais do banco (criar) - NÃO commitado
├── .env.example                              # Exemplo de configuração - Commitado
├── .gitignore                                # Configuração Git - Commitado
├── GITIGNORE_EXPLANATION.md                  # Explicação do .gitignore
├── README.md                                 # Este arquivo
├── relatorio_mensal_geral_2025-03 (1).csv   # Dados de entrada (fallback) - NÃO commitado
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

### **Principais Melhorias para Aumentar Precisão:**

1. **Sequência Temporal Maior:** Aumentada de 5 para 10 pontos
2. **Múltiplas Features:** Usa OHLCV + 20+ indicadores técnicos
3. **Arquitetura Mais Profunda:** 4 camadas LSTM + BatchNorm
4. **Otimização de Hiperparâmetros:** RandomizedSearchCV
5. **Ensemble de Modelos:** Combina LSTM + RF + GB
6. **Callbacks Avançados:** Early stopping, reduce LR, checkpoints
7. **Loss Function Robusta:** Huber loss (mais resistente a outliers)
8. **Validação Temporal:** TimeSeriesSplit para séries temporais

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

### **Problemas com o CSV**

- Verifique se o arquivo CSV usa separador `;`
- Certifique-se de que existe uma coluna chamada `close`
- Remova linhas com valores NaN ou vazios

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

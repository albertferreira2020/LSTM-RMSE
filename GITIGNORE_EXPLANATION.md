# GITIGNORE_EXPLANATION.md - Explicação do arquivo .gitignore

## 📝 **Explicação do .gitignore**

Este arquivo `.gitignore` foi criado especificamente para o projeto de previsão
com LSTM e Random Forest. Aqui está o que cada seção ignora:

### 🔒 **Credenciais e Configurações Sensíveis**

- `.env` - Credenciais do banco PostgreSQL
- `*.key`, `*.pem` - Chaves de autenticação
- `secrets.json` - Arquivos com dados sensíveis

### 🐍 **Python**

- `__pycache__/` - Cache compilado do Python
- `*.pyc` - Arquivos compilados
- `venv/`, `env/` - Ambientes virtuais
- `*.egg-info/` - Metadados de pacotes

### 🤖 **Machine Learning**

- `*.h5`, `*.hdf5` - Modelos TensorFlow/Keras salvos
- `*.pkl`, `*.joblib` - Modelos scikit-learn salvos
- `best_lstm_model.h5` - Checkpoint do melhor modelo LSTM
- `logs/` - Logs de treinamento
- `tensorboard_logs/` - Logs do TensorBoard

### 📊 **Dados**

- `*.csv` - Arquivos de dados (exceto exemplos)
- `relatorio_mensal_geral_*.csv` - Dados específicos do projeto
- `botbinance_data_*.csv` - Dados do banco
- `processed_data/` - Dados processados

### 💻 **Sistema Operacional**

- `.DS_Store` - Metadados do macOS
- `Thumbs.db` - Cache de miniaturas Windows
- `*~` - Arquivos temporários Linux

### 🛠️ **IDEs e Editores**

- `.vscode/` - Configurações VS Code
- `.idea/` - Configurações PyCharm
- `*.sublime-*` - Configurações Sublime Text

### 📈 **Resultados e Gráficos**

- `plots/`, `figures/` - Gráficos gerados
- `*.png`, `*.jpg` - Imagens (exceto exemplos)
- `results/` - Resultados de experimentos

## ✅ **Arquivos que SERÃO commitados:**

- `main.py`, `main_advanced.py` - Código principal
- `config.py` - Configurações
- `database.py` - Gerenciador de banco
- `technical_indicators.py` - Indicadores técnicos
- `requirements.txt` - Dependências
- `.env.example` - Exemplo de configuração
- `README.md` - Documentação
- `.gitignore` - Este arquivo

## ❌ **Arquivos que NÃO serão commitados:**

- `.env` - Suas credenciais reais
- `best_lstm_model.h5` - Modelos treinados
- Dados CSV reais
- Gráficos gerados
- Cache Python
- Ambientes virtuais

## 🔧 **Como usar:**

1. **Primeiro commit:** O `.gitignore` já está configurado
2. **Dados sensíveis:** Nunca serão enviados ao Git
3. **Modelos treinados:** Podem ser grandes, são ignorados
4. **Resultados:** Gere localmente, não são commitados

## 💡 **Dicas:**

- Se precisar commitar um arquivo específico ignorado, use:
  `git add -f arquivo.csv`
- Para ver arquivos ignorados: `git status --ignored`
- Para limpar cache: `git rm -r --cached .` e depois `git add .`

Este `.gitignore` garante que apenas o código fonte e configurações necessárias
sejam versionadas, mantendo o repositório limpo e seguro.

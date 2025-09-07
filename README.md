# Previsão de Valor de Close com LSTM e Random Forest

Este projeto utiliza redes neurais LSTM (TensorFlow/Keras) e Random Forest
(Scikit-learn) para prever o valor de **close** do próximo bloco, com base em
dados históricos (blocos Renko).

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

1. Certifique-se de que o ambiente virtual está ativado:

   ```bash
   source env/bin/activate  # macOS/Linux
   # ou
   env\Scripts\activate     # Windows
   ```

2. Coloque o arquivo CSV no mesmo diretório e execute:
   ```bash
   python main.py
   ```

## **Estrutura do Projeto**

```
perceptronLTSM/
├── main.py                                    # Script principal
├── setup_env.py                              # Script de configuração automática
├── check_install.py                          # Script de verificação simples
├── requirements.txt                           # Dependências do projeto
├── requirements-stable.txt                    # Versões estáveis (backup)
├── README.md                                 # Este arquivo
├── relatorio_mensal_geral_2025-03 (1).csv   # Dados de entrada
└── env/                                      # Ambiente virtual (criado após setup)
```

## **O que o script faz:**

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

✅ **Visualização:**

- Gera gráficos comparativos das previsões
- Mostra valores reais vs previsões

✅ **Previsão:**

- Faz previsão do próximo valor de close
- Exibe resultados de ambos os modelos

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

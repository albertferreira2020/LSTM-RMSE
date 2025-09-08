#!/usr/bin/env python3
"""
fix_macos_tensorflow.py - Script para corrigir problemas do TensorFlow no macOS

Este script detecta e corrige os problemas mais comuns do TensorFlow no macOS:
1. Problemas de threading (mutex lock failed)
2. Incompatibilidade com LibreSSL
3. Problemas de versão do TensorFlow
4. Conflitos de dependências

Uso:
    python3 fix_macos_tensorflow.py
"""

import os
import sys
import subprocess
import platform
import warnings

def print_banner():
    """Imprime banner do script"""
    print("=" * 60)
    print("🔧 FIX TENSORFLOW PARA MACOS")
    print("=" * 60)
    print()

def check_system_info():
    """Verifica informações do sistema"""
    print("📋 INFORMAÇÕES DO SISTEMA:")
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Arquitetura: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Diretório do projeto: {os.getcwd()}")
    print()

def check_virtual_env():
    """Verifica se está em ambiente virtual"""
    is_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if is_venv:
        print("✅ Ambiente virtual ativo")
        print(f"   Caminho: {sys.prefix}")
    else:
        print("⚠️  AVISO: Não está em ambiente virtual!")
        print("   Recomendo executar: source env/bin/activate")
    print()
    
    return is_venv

def run_command(command, description, check=True):
    """Executa um comando e mostra o resultado"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=check
        )
        if result.stdout:
            print(f"   Saída: {result.stdout.strip()}")
        if result.stderr and result.returncode == 0:
            print(f"   Aviso: {result.stderr.strip()}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erro: {e}")
        if e.stderr:
            print(f"   Stderr: {e.stderr.strip()}")
        return False, None

def fix_urllib3_openssl():
    """Corrige problema do urllib3 com OpenSSL"""
    print("🔧 CORRIGINDO URLLIB3/OPENSSL:")
    
    # Verificar versão atual do urllib3
    success, output = run_command(
        "pip show urllib3", 
        "Verificando versão do urllib3",
        check=False
    )
    
    if success and "Version: 2." in output:
        print("   📦 Downgrading urllib3 para versão compatível...")
        run_command(
            "pip install 'urllib3<2.0'",
            "Instalando urllib3 v1.x (compatível com LibreSSL)"
        )
    
    # Instalar certificados do sistema se necessário
    if platform.system() == "Darwin":  # macOS
        run_command(
            "/Applications/Python\\ 3.*/Install\\ Certificates.command 2>/dev/null || true",
            "Atualizando certificados do sistema",
            check=False
        )

def fix_tensorflow_version():
    """Corrige versão do TensorFlow para compatibilidade com macOS"""
    print("🔧 CORRIGINDO TENSORFLOW:")
    
    # Desinstalar TensorFlow atual
    run_command(
        "pip uninstall -y tensorflow tensorflow-macos tensorflow-metal",
        "Removendo versões anteriores do TensorFlow"
    )
    
    # Limpar cache
    run_command("pip cache purge", "Limpando cache do pip")
    
    # Detectar se é Apple Silicon ou Intel
    is_apple_silicon = platform.machine() == "arm64"
    
    if is_apple_silicon:
        print("   🍎 Detectado Apple Silicon (M1/M2/M3)")
        print("   📦 Instalando TensorFlow otimizado para Apple Silicon...")
        
        # Instalar versão otimizada para Apple Silicon
        commands = [
            "pip install tensorflow-macos==2.12.0",
            "pip install tensorflow-metal==0.8.0"
        ]
        
        for cmd in commands:
            success, _ = run_command(cmd, f"Executando: {cmd}")
            if not success:
                print(f"   ⚠️ Falha em: {cmd}")
                print("   🔄 Tentando versão alternativa...")
                run_command(
                    "pip install tensorflow==2.13.0",
                    "Instalando TensorFlow padrão"
                )
                break
    else:
        print("   💻 Detectado Intel Mac")
        print("   📦 Instalando TensorFlow padrão...")
        run_command(
            "pip install tensorflow==2.13.0",
            "Instalando TensorFlow para Intel Mac"
        )

def create_tensorflow_wrapper():
    """Cria wrapper para configurações do TensorFlow"""
    print("🔧 CRIANDO WRAPPER DE CONFIGURAÇÃO:")
    
    wrapper_content = '''#!/usr/bin/env python3
"""
tensorflow_config.py - Configurações seguras para TensorFlow no macOS
"""

import os
import warnings

def configure_tensorflow_for_macos():
    """Configura TensorFlow para funcionar corretamente no macOS"""
    
    # Suprimir avisos desnecessários
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')
    
    # Configurações de threading para evitar mutex lock failed
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    
    # Desabilitar OneDNN para evitar conflitos
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Configurações de memória
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    try:
        import tensorflow as tf
        
        # Configurar threading
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # Configurar GPUs se disponíveis
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        return True
        
    except Exception as e:
        print(f"Erro ao configurar TensorFlow: {e}")
        return False

# Aplicar configurações automaticamente ao importar
configure_tensorflow_for_macos()
'''
    
    with open("tensorflow_config.py", "w") as f:
        f.write(wrapper_content)
    
    print("   ✅ Wrapper criado: tensorflow_config.py")

def test_tensorflow():
    """Testa se TensorFlow está funcionando"""
    print("🧪 TESTANDO TENSORFLOW:")
    
    test_script = '''
import sys
sys.path.insert(0, ".")

try:
    # Importar configurações
    import tensorflow_config
    
    # Importar TensorFlow
    import tensorflow as tf
    
    print(f"✅ TensorFlow {tf.__version__} importado com sucesso!")
    
    # Teste básico
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.add(a, b)
    
    print(f"✅ Teste de computação: {a.numpy()} + {b.numpy()} = {c.numpy()}")
    
    # Testar criação de modelo
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential([Dense(1, input_shape=(1,))])
    model.compile(optimizer='adam', loss='mse')
    
    print("✅ Modelo Keras criado com sucesso!")
    print("🎉 TENSORFLOW ESTÁ FUNCIONANDO CORRETAMENTE!")
    
except Exception as e:
    print(f"❌ Erro no teste: {e}")
    sys.exit(1)
'''
    
    with open("test_tf_temp.py", "w") as f:
        f.write(test_script)
    
    success, _ = run_command(
        "python3 test_tf_temp.py",
        "Executando teste do TensorFlow"
    )
    
    # Limpar arquivo temporário
    if os.path.exists("test_tf_temp.py"):
        os.remove("test_tf_temp.py")
    
    return success

def create_safe_alternative():
    """Cria versão segura sem TensorFlow"""
    print("🔧 CRIANDO ALTERNATIVA SEGURA:")
    
    safe_content = '''#!/usr/bin/env python3
"""
main_safe_no_tf.py - Versão do main_advanced.py sem TensorFlow

Esta versão usa apenas scikit-learn e XGBoost para máxima compatibilidade.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_sequences(data, n_steps):
    """Cria sequências para modelos não-LSTM"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        # Usar valores das últimas n_steps como features
        X.append(data[i-n_steps:i].flatten())
        y.append(data[i])
    return np.array(X), np.array(y)

def main():
    print("🚀 INICIANDO VERSÃO SEGURA (SEM TENSORFLOW)")
    
    # Carregar dados
    try:
        df = pd.read_csv('relatorio_mensal_geral_2025-03 (1).csv', sep=';')
        print(f"✅ Dados carregados: {len(df)} registros")
    except FileNotFoundError:
        print("❌ Arquivo CSV não encontrado!")
        return
    
    # Preparar dados
    close_data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)
    
    # Criar sequências
    n_steps = 60
    X, y = create_sequences(scaled_data, n_steps)
    
    # Dividir dados
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"📊 Dados de treino: {X_train.shape}")
    print(f"📊 Dados de teste: {X_test.shape}")
    
    # Modelos
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    # Treinar e avaliar modelos
    for name, model in models.items():
        print(f"\\n🔄 Treinando {name}...")
        
        model.fit(X_train, y_train.ravel())
        
        # Previsões
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Métricas
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'predictions': test_pred
        }
        
        print(f"   📊 RMSE treino: {train_rmse:.6f}")
        print(f"   📊 RMSE teste: {test_rmse:.6f}")
        print(f"   📊 R² teste: {test_r2:.4f}")
        print(f"   📊 MAE teste: {test_mae:.6f}")
    
    # Ensemble simples
    ensemble_pred = np.mean([
        results['Random Forest']['predictions'],
        results['Gradient Boosting']['predictions']
    ], axis=0)
    
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\\n🎯 ENSEMBLE:")
    print(f"   📊 RMSE: {ensemble_rmse:.6f}")
    print(f"   📊 R²: {ensemble_r2:.4f}")
    
    # Visualização
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(y_test[-100:], label='Real', alpha=0.8)
    plt.plot(results['Random Forest']['predictions'][-100:], label='Random Forest', alpha=0.8)
    plt.plot(results['Gradient Boosting']['predictions'][-100:], label='Gradient Boosting', alpha=0.8)
    plt.plot(ensemble_pred[-100:], label='Ensemble', alpha=0.8, linewidth=2)
    plt.title('Comparação dos Modelos (Últimas 100 previsões)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, ensemble_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões Ensemble')
    plt.title(f'Real vs Previsto (R² = {ensemble_r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    model_names = list(results.keys()) + ['Ensemble']
    rmse_values = [results[name]['test_rmse'] for name in results.keys()] + [ensemble_rmse]
    plt.bar(model_names, rmse_values)
    plt.title('Comparação RMSE')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    residuals = y_test.ravel() - ensemble_pred
    plt.scatter(ensemble_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduos')
    plt.title('Análise de Resíduos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados_safe_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\n✅ ANÁLISE COMPLETA!")
    print("📊 Gráfico salvo como: resultados_safe_model.png")

if __name__ == "__main__":
    main()
'''
    
    with open("main_safe_no_tf.py", "w") as f:
        f.write(safe_content)
    
    print("   ✅ Versão segura criada: main_safe_no_tf.py")

def main():
    """Função principal do script de correção"""
    print_banner()
    check_system_info()
    
    if not check_virtual_env():
        print("💡 Para ativar o ambiente virtual:")
        print("   source env/bin/activate")
        print()
    
    print("🔧 INICIANDO CORREÇÕES...")
    print()
    
    # Passo 1: Corrigir urllib3/OpenSSL
    fix_urllib3_openssl()
    print()
    
    # Passo 2: Corrigir TensorFlow
    fix_tensorflow_version()
    print()
    
    # Passo 3: Criar wrapper de configuração
    create_tensorflow_wrapper()
    print()
    
    # Passo 4: Testar TensorFlow
    if test_tensorflow():
        print("🎉 SUCESSO! TensorFlow está funcionando!")
        print()
        print("💡 Para usar o sistema principal:")
        print("   python3 main_advanced.py")
        print()
        print("💡 Para importar TensorFlow em outros scripts:")
        print("   import tensorflow_config  # Adicione esta linha primeiro")
        print("   import tensorflow as tf   # Depois importe normalmente")
    else:
        print("⚠️ TensorFlow ainda apresenta problemas.")
        print("📦 Criando versão alternativa...")
        create_safe_alternative()
        print()
        print("💡 Use a versão segura:")
        print("   python3 main_safe_no_tf.py")
    
    print()
    print("=" * 60)
    print("✅ CORREÇÃO CONCLUÍDA!")
    print("=" * 60)

if __name__ == "__main__":
    main()

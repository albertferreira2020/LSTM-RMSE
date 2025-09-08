#!/usr/bin/env python3
"""
Script de solução rápida para o erro "mutex lock failed" no TensorFlow/macOS
"""

import os
import sys
import subprocess

def quick_fix_mutex_error():
    """Aplica fix rápido para erro de mutex lock no macOS"""
    print("🔧 APLICANDO FIX RÁPIDO PARA MUTEX LOCK ERROR")
    print("=" * 50)
    
    # 1. Definir variáveis de ambiente críticas
    print("1️⃣ Configurando variáveis de ambiente...")
    env_vars = {
        'OMP_NUM_THREADS': '1',
        'TF_NUM_INTEROP_THREADS': '1', 
        'TF_NUM_INTRAOP_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': 'YES'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   {var} = {value}")
    
    # 2. Reinstalar TensorFlow com versão estável
    print("\n2️⃣ Reinstalando TensorFlow...")
    
    # Desinstalar versões existentes
    packages_to_remove = ['tensorflow', 'tensorflow-macos', 'tensorflow-metal']
    for package in packages_to_remove:
        subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', package, '-y'
        ], capture_output=True)
    
    # Limpar cache
    subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], 
                   capture_output=True)
    
    # Verificar arquitetura
    arch_check = subprocess.run(['uname', '-m'], capture_output=True, text=True)
    is_apple_silicon = 'arm64' in arch_check.stdout
    
    if is_apple_silicon:
        print("   🍎 Instalando para Apple Silicon...")
        # Instalar versão estável para Apple Silicon
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'tensorflow-macos==2.13.0'
        ])
    else:
        print("   💻 Instalando para Intel Mac...")
        # Instalar versão estável para Intel
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'tensorflow==2.13.0'
        ])
    
    # 3. Criar script de importação segura
    print("\n3️⃣ Criando importação segura...")
    safe_import_content = '''#!/usr/bin/env python3
"""
Importação segura do TensorFlow para macOS
Use: from safe_tensorflow import tf
"""

import os
import warnings

# Configurar ambiente ANTES de importar TensorFlow
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Suprimir warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

# Importar TensorFlow
try:
    import tensorflow as tf
    
    # Configurar threading após importação
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Configurar GPUs se disponíveis
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    print(f"✅ TensorFlow {tf.__version__} carregado com sucesso")
    
except Exception as e:
    print(f"❌ Erro ao carregar TensorFlow: {e}")
    tf = None

# Exportar tensorflow configurado
__all__ = ['tf']
'''
    
    with open('safe_tensorflow.py', 'w') as f:
        f.write(safe_import_content)
    
    print("   ✅ Arquivo safe_tensorflow.py criado")
    
    # 4. Testar a correção
    print("\n4️⃣ Testando correção...")
    test_content = '''
import sys
import os

# Aplicar configurações
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

try:
    print("Importando TensorFlow...")
    import tensorflow as tf
    
    print("Configurando threading...")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    print("Testando operação básica...")
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.add(a, b)
    print(f"Teste: {c.numpy()}")
    
    print("Testando Keras/LSTM...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    
    model = Sequential([
        LSTM(10, input_shape=(5, 1)),
        Dense(1)
    ])
    
    print("🎉 TODOS OS TESTES PASSARAM!")
    print(f"TensorFlow {tf.__version__} está funcionando!")
    
except Exception as e:
    print(f"❌ Teste falhou: {e}")
    sys.exit(1)
'''
    
    with open('test_quick_fix.py', 'w') as f:
        f.write(test_content)
    
    # Executar teste
    result = subprocess.run([sys.executable, 'test_quick_fix.py'], 
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ CORREÇÃO APLICADA COM SUCESSO!")
        print("\n📝 PRÓXIMOS PASSOS:")
        print("1. Reinicie o terminal")
        print("2. Ative o ambiente virtual: source env/bin/activate")
        print("3. Teste: python3 test_quick_fix.py")
        print("4. Se OK, execute: python3 main_advanced.py")
        print("\n💡 Para usar TensorFlow nos seus scripts:")
        print("   from safe_tensorflow import tf")
        
        # Limpar arquivo de teste
        os.remove('test_quick_fix.py')
        
    else:
        print("❌ Teste falhou. Saída:")
        print(result.stdout)
        print(result.stderr)
        print("\n🔄 Alternativas:")
        print("1. Reiniciar o Mac completamente")
        print("2. Usar: python3 main_safe.py (sem TensorFlow)")
        print("3. Tentar: conda install tensorflow")

if __name__ == "__main__":
    quick_fix_mutex_error()

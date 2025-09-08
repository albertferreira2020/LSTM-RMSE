#!/usr/bin/env python3
"""
Script para reinstalar TensorFlow com configurações otimizadas para macOS
Resolve problemas de mutex lock, OpenSSL warnings e instabilidades
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Verifica se a versão do Python é adequada"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 9:
        print("❌ Python 3.9+ é necessário")
        return False
    
    if version.minor >= 12:
        print("⚠️  Python 3.12+ pode ter problemas - recomendamos 3.11")
    
    return True

def check_apple_silicon():
    """Verifica se está em Apple Silicon"""
    try:
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
        is_arm = 'arm64' in result.stdout
        if is_arm:
            print("🔥 Apple Silicon (M1/M2/M3) detectado")
        else:
            print("💻 Intel Mac detectado")
        return is_arm
    except:
        return False

def uninstall_tensorflow():
    """Remove todas as versões do TensorFlow"""
    print("\n🗑️  Removendo instalações anteriores do TensorFlow...")
    
    packages_to_remove = [
        'tensorflow',
        'tensorflow-cpu',
        'tensorflow-gpu',
        'tensorflow-macos',
        'tensorflow-metal',
        'tf-nightly',
        'tf-estimator',
        'tensorboard'
    ]
    
    for package in packages_to_remove:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'uninstall', 
                package, '-y'
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {package} removido")
        except:
            pass

def clear_cache():
    """Limpa cache do pip e compilação"""
    print("\n🧹 Limpando cache...")
    
    # Limpar cache do pip
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], 
                      capture_output=True)
        print("   ✅ Cache do pip limpo")
    except:
        print("   ⚠️  Não foi possível limpar cache do pip")
    
    # Limpar cache Python
    import shutil
    try:
        cache_dirs = ['__pycache__', '.pytest_cache', '.coverage']
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
        print("   ✅ Cache Python limpo")
    except:
        print("   ⚠️  Não foi possível limpar cache Python")

def install_tensorflow_for_macos():
    """Instala TensorFlow otimizado para macOS"""
    print("\n📦 Instalando TensorFlow otimizado para macOS...")
    
    is_apple_silicon = check_apple_silicon()
    
    # Atualizar pip primeiro
    print("   Atualizando pip...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
    ], capture_output=True)
    
    if is_apple_silicon:
        # Para Apple Silicon (M1/M2/M3)
        print("   🍎 Instalando TensorFlow para Apple Silicon...")
        packages = [
            'tensorflow-macos==2.13.0',  # Versão estável para Apple Silicon
            'tensorflow-metal==1.0.1',   # Aceleração Metal
        ]
    else:
        # Para Intel Macs
        print("   🖥️  Instalando TensorFlow para Intel Mac...")
        packages = [
            'tensorflow==2.13.0',  # Versão estável para Intel
        ]
    
    # Instalar pacotes principais
    for package in packages:
        print(f"   Instalando {package}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            '--no-cache-dir', package
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ {package} instalado")
        else:
            print(f"   ❌ Erro ao instalar {package}")
            print(f"      {result.stderr}")
    
    # Instalar dependências adicionais
    print("   Instalando dependências complementares...")
    additional_packages = [
        'numpy==1.24.3',      # Versão compatível
        'h5py==3.9.0',        # Para salvar modelos
        'protobuf==3.20.3',   # Evita conflitos
    ]
    
    for package in additional_packages:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            '--no-cache-dir', package
        ], capture_output=True)

def create_tensorflow_config():
    """Cria arquivo de configuração para TensorFlow"""
    config_content = '''#!/usr/bin/env python3
"""
Configuração automática para TensorFlow no macOS
Use: from tensorflow_config_macos import setup_tensorflow
"""

import os
import warnings

def setup_tensorflow():
    """
    Configura TensorFlow para macOS antes da importação
    DEVE ser chamado antes de importar tensorflow
    """
    # Suprimir warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')
    warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*')
    
    # Configurações críticas
    env_config = {
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'OMP_NUM_THREADS': '1',
        'TF_NUM_INTEROP_THREADS': '1', 
        'TF_NUM_INTRAOP_THREADS': '1',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': 'YES',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
    }
    
    for key, value in env_config.items():
        os.environ[key] = value
    
    print("🔧 TensorFlow configurado para macOS")

def import_tensorflow_safe():
    """Importa TensorFlow com configuração segura"""
    setup_tensorflow()
    
    try:
        import tensorflow as tf
        
        # Configurações pós-importação
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
        return tf
        
    except Exception as e:
        print(f"❌ Erro ao carregar TensorFlow: {e}")
        return None

# Configuração automática quando importado
setup_tensorflow()
'''
    
    with open('tensorflow_config_macos.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Arquivo tensorflow_config_macos.py criado")

def test_installation():
    """Testa a instalação do TensorFlow"""
    print("\n🧪 Testando instalação do TensorFlow...")
    
    try:
        # Importar com configuração
        exec(open('tensorflow_config_macos.py').read())
        import tensorflow as tf
        
        print(f"✅ TensorFlow {tf.__version__} importado com sucesso")
        
        # Teste básico
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✅ Teste de operação: {c.numpy()}")
        
        # Teste Keras
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, LSTM
        
        model = Sequential([Dense(1, input_shape=(1,))])
        print("✅ Keras funcionando")
        
        lstm_model = Sequential([LSTM(10, input_shape=(5, 1)), Dense(1)])
        print("✅ LSTM funcionando")
        
        print("\n🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ Teste falhou: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 REINSTALAÇÃO COMPLETA DO TENSORFLOW PARA MACOS")
    print("=" * 55)
    
    # Verificações iniciais
    if not check_python_version():
        return False
    
    print(f"Sistema: macOS {platform.mac_ver()[0]}")
    
    # Processo de reinstalação
    try:
        uninstall_tensorflow()
        clear_cache() 
        install_tensorflow_for_macos()
        create_tensorflow_config()
        
        if test_installation():
            print("\n✅ TENSORFLOW INSTALADO E CONFIGURADO!")
            print("\nPróximos passos:")
            print("1. Reinicie o terminal")
            print("2. Execute: python3 test_tensorflow_fix.py")
            print("3. Se OK, execute: python3 main_advanced.py")
            
            print("\n💡 Para usar TensorFlow nos seus scripts:")
            print("   from tensorflow_config_macos import import_tensorflow_safe")
            print("   tf = import_tensorflow_safe()")
            
        else:
            print("\n⚠️  Instalação completada mas com problemas nos testes")
            print("Tente reiniciar o terminal e testar novamente")
            
    except KeyboardInterrupt:
        print("\n⚠️  Instalação interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante instalação: {e}")

if __name__ == "__main__":
    main()

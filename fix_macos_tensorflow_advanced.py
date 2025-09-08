#!/usr/bin/env python3
"""
Fix avançado para TensorFlow no macOS
Resolve problemas de mutex lock, OpenSSL warnings e instabilidades
"""

import os
import sys
import warnings
import platform
import subprocess

def detect_macos_version():
    """Detecta a versão do macOS"""
    try:
        version = platform.mac_ver()[0]
        major, minor = map(int, version.split('.')[:2])
        return major, minor
    except:
        return 0, 0

def check_apple_silicon():
    """Verifica se está rodando em Apple Silicon (M1/M2/M3)"""
    try:
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
        return 'arm64' in result.stdout
    except:
        return False

def apply_macos_tensorflow_fix():
    """
    Aplica fix abrangente para TensorFlow no macOS
    Deve ser chamado ANTES de importar tensorflow
    """
    print("🔧 Aplicando fix para TensorFlow no macOS...")
    
    # 1. Suprimir warnings específicos
    warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')
    warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*')
    warnings.filterwarnings('ignore', message='.*LibreSSL.*')
    
    # 2. Configurações críticas de ambiente
    env_vars = {
        # TensorFlow logging
        'TF_CPP_MIN_LOG_LEVEL': '2',  # Suprimir logs INFO e WARNING
        
        # Threading - CRÍTICO para resolver mutex lock
        'OMP_NUM_THREADS': '1',
        'TF_NUM_INTEROP_THREADS': '1',
        'TF_NUM_INTRAOP_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        
        # OneDNN/MKL-DNN - Pode causar problemas no macOS
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'TF_DISABLE_MKL': '1',
        
        # GPU/Metal - Desabilitar se causar problemas
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'CUDA_VISIBLE_DEVICES': '',  # Força CPU se necessário
        
        # Memory management
        'TF_GPU_ALLOCATOR': 'cuda_malloc_async',
        'TF_ENABLE_GPU_GARBAGE_COLLECTION': 'false',
        
        # Compatibilidade macOS
        'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': 'YES',
        'TF_DISABLE_SEGMENT_REDUCTION': '1',
    }
    
    # Configurações específicas para Apple Silicon
    if check_apple_silicon():
        print("📱 Detectado Apple Silicon - aplicando configurações específicas...")
        env_vars.update({
            'TF_ENABLE_MLIR_BRIDGE': '0',
            'TF_DISABLE_MKL': '1',
            'TF_ENABLE_ONEDNN_OPTS': '0',
        })
    
    # Detectar versão do macOS
    major, minor = detect_macos_version()
    if major >= 12:  # macOS Monterey ou superior
        print(f"🍎 Detectado macOS {major}.{minor} - aplicando configurações modernas...")
        env_vars.update({
            'TF_DISABLE_MKL': '1',
            'TF_ENABLE_ONEDNN_OPTS': '0',
        })
    
    # Aplicar todas as variáveis
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key} = {value}")
    
    print("✅ Configurações de ambiente aplicadas")

def configure_tensorflow_after_import():
    """
    Configurações que devem ser aplicadas APÓS importar tensorflow
    """
    try:
        import tensorflow as tf
        print("🔧 Configurando TensorFlow após importação...")
        
        # Threading configuration
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # GPU configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🖥️  {len(gpus)} GPU(s) encontrada(s)")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"   GPU configurada: {gpu}")
            except RuntimeError as e:
                print(f"⚠️  Aviso GPU: {e}")
        else:
            print("💻 Usando CPU (sem GPU detectada)")
        
        # Configurações experimentais para macOS
        try:
            tf.config.experimental.enable_op_determinism()
            print("✅ Determinismo habilitado")
        except:
            print("⚠️  Determinismo não disponível nesta versão")
        
        print(f"✅ TensorFlow {tf.__version__} configurado com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro na configuração pós-importação: {e}")
        return False

def test_tensorflow_installation():
    """Teste completo da instalação do TensorFlow"""
    print("\n" + "="*60)
    print("🧪 TESTE COMPLETO DO TENSORFLOW")
    print("="*60)
    
    try:
        # Aplicar fix antes da importação
        apply_macos_tensorflow_fix()
        
        print("\n1️⃣ Testando importação...")
        import tensorflow as tf
        print("✅ Importação bem-sucedida")
        
        # Configurar após importação
        configure_tensorflow_after_import()
        
        print("\n2️⃣ Testando operações básicas...")
        # Teste de tensores
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        print(f"   Soma: {a.numpy()} + {b.numpy()} = {c.numpy()}")
        
        # Teste de matriz
        matrix = tf.random.normal([3, 3])
        result = tf.linalg.det(matrix)
        print(f"   Determinante de matriz 3x3: {result.numpy():.4f}")
        
        print("\n3️⃣ Testando Keras...")
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        
        model = Sequential([
            Dense(10, activation='relu', input_shape=(5,)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        print("✅ Modelo Keras criado com sucesso")
        
        print("\n4️⃣ Testando LSTM...")
        from tensorflow.keras.layers import LSTM
        
        lstm_model = Sequential([
            LSTM(50, input_shape=(10, 1)),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        print("✅ Modelo LSTM criado com sucesso")
        
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ TensorFlow está funcionando perfeitamente no seu macOS")
        
        # Informações do sistema
        print("\n📊 INFORMAÇÕES DO SISTEMA:")
        print(f"   TensorFlow: {tf.__version__}")
        print(f"   Python: {sys.version}")
        print(f"   macOS: {platform.mac_ver()[0]}")
        print(f"   Arquitetura: {platform.machine()}")
        print(f"   Dispositivos: {len(tf.config.list_physical_devices())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ TensorFlow não instalado: {e}")
        print("\n💡 Solução:")
        print("   pip install --upgrade tensorflow")
        return False
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        print(f"   Tipo do erro: {type(e).__name__}")
        
        # Diagnóstico específico para mutex lock
        if 'mutex lock failed' in str(e).lower():
            print("\n🔍 DIAGNÓSTICO: Problema de mutex lock detectado")
            print("💡 Soluções recomendadas:")
            print("   1. Reiniciar o terminal")
            print("   2. Executar: export OMP_NUM_THREADS=1")
            print("   3. Executar: export TF_NUM_INTEROP_THREADS=1")
            print("   4. Considerar usar conda em vez de pip")
        
        return False

def main():
    """Função principal"""
    print("🚀 FIX AVANÇADO PARA TENSORFLOW NO MACOS")
    print("=" * 50)
    
    # Mostrar informações do sistema
    print(f"Sistema: macOS {platform.mac_ver()[0]}")
    print(f"Arquitetura: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    if check_apple_silicon():
        print("🔥 Apple Silicon detectado (M1/M2/M3)")
    
    # Executar teste
    success = test_tensorflow_installation()
    
    if success:
        print("\n✅ TENSORFLOW PRONTO PARA USO!")
        print("Você pode executar:")
        print("   python3 main_advanced.py")
    else:
        print("\n⚠️  PROBLEMAS DETECTADOS")
        print("Alternativas:")
        print("   python3 main_safe.py  # Sem TensorFlow")
        print("   python3 fix_macos_tensorflow_advanced.py  # Este script")

if __name__ == "__main__":
    main()

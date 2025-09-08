#!/usr/bin/env python3
# test_tensorflow_fix.py - Testa se o fix do TensorFlow está funcionando

import os
import sys

def test_tensorflow_import():
    """Testa importação do TensorFlow com as configurações de fix"""
    print("=== TESTE DE IMPORTAÇÃO DO TENSORFLOW ===")
    
    try:
        # Aplicar configurações específicas para macOS antes de importar
        print("Aplicando configurações de ambiente para macOS...")
        
        # Suprimir avisos do urllib3/OpenSSL
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')
        warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*')
        warnings.filterwarnings('ignore', message='.*LibreSSL.*')
        
        # Configurações críticas para macOS - RESOLVER MUTEX LOCK
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_DISABLE_MKL'] = '1'
        
        # Threading - CRÍTICO para mutex lock failed
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Configurações macOS específicas
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Força uso de CPU se houver problemas
        
        print("Tentando importar TensorFlow...")
        import tensorflow as tf
        
        print("Configurando threading...")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # Verificar GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs encontradas: {len(gpus)}")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU configurada: {gpu}")
                except RuntimeError as e:
                    print(f"Erro ao configurar GPU: {e}")
        else:
            print("Nenhuma GPU encontrada, usando CPU")
        
        print("Testando importações do Keras...")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        print("Testando criação de modelo simples...")
        model = Sequential()
        model.add(Dense(10, input_shape=(5,), activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        print("✅ TensorFlow está funcionando corretamente!")
        print(f"Versão do TensorFlow: {tf.__version__}")
        print(f"Dispositivos disponíveis: {tf.config.list_physical_devices()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ TensorFlow não está instalado: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao configurar TensorFlow: {e}")
        return False

def test_simple_computation():
    """Testa uma computação simples com TensorFlow"""
    try:
        import tensorflow as tf
        
        print("\n=== TESTE DE COMPUTAÇÃO SIMPLES ===")
        
        # Teste de operação básica
        a = tf.constant([1, 2, 3, 4])
        b = tf.constant([5, 6, 7, 8])
        result = tf.add(a, b)
        
        print(f"Teste de adição: {a.numpy()} + {b.numpy()} = {result.numpy()}")
        
        # Teste com matriz
        matrix_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        matrix_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
        matrix_result = tf.matmul(matrix_a, matrix_b)
        
        print(f"Teste de multiplicação de matrizes:")
        print(f"A:\n{matrix_a.numpy()}")
        print(f"B:\n{matrix_b.numpy()}")
        print(f"A @ B:\n{matrix_result.numpy()}")
        
        print("✅ Computações básicas funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ Erro em computações básicas: {e}")
        return False

def main():
    """Função principal de teste"""
    print("TESTE DE CONFIGURAÇÃO DO TENSORFLOW\n")
    
    # Teste 1: Importação
    import_ok = test_tensorflow_import()
    
    if import_ok:
        # Teste 2: Computação
        computation_ok = test_simple_computation()
        
        if computation_ok:
            print("\n🎉 TODOS OS TESTES PASSARAM!")
            print("Você pode usar o main_advanced.py com confiança.")
        else:
            print("\n⚠️ Importação OK, mas computação falhou.")
            print("Recomendo usar o main_safe.py (sem TensorFlow).")
    else:
        print("\n⚠️ TensorFlow não disponível.")
        print("Use o main_safe.py para análises sem TensorFlow.")
    
    print("\nArquivos disponíveis:")
    print("- main_advanced.py: Versão completa com LSTM (com fix para macOS)")
    print("- main_safe.py: Versão segura apenas com RandomForest e GradientBoosting")
    print("- python3 test_tensorflow_fix.py: Este teste")

if __name__ == "__main__":
    main()

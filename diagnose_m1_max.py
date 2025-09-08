#!/usr/bin/env python3
# diagnose_m1_max.py - Diagnóstico completo para Mac M1 Max GPU

import platform
import sys
import subprocess

def check_system_info():
    """
    Verifica informações do sistema
    """
    print("🖥️ === INFORMAÇÕES DO SISTEMA ===")
    print(f"Sistema: {platform.system()}")
    print(f"Versão: {platform.version()}")
    print(f"Arquitetura: {platform.machine()}")
    print(f"Processador: {platform.processor()}")
    print(f"Python: {sys.version}")
    
    # Verifica se é M1/M2
    if platform.machine() == 'arm64':
        print("✅ Chip Apple Silicon detectado (M1/M2)")
        return True
    else:
        print("⚠️ Não é chip Apple Silicon")
        return False

def check_tensorflow_metal():
    """
    Verifica TensorFlow e suporte Metal
    """
    print("\n🔧 === VERIFICANDO TENSORFLOW ===")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow instalado: {tf.__version__}")
        
        # Verifica build info
        print(f"Compilado com CUDA: {tf.test.is_built_with_cuda()}")
        print(f"GPU disponível: {tf.test.is_gpu_available()}")
        
        # Lista dispositivos
        print("\n📱 Dispositivos disponíveis:")
        devices = tf.config.list_physical_devices()
        for i, device in enumerate(devices):
            print(f"   {i+1}. {device}")
        
        # Testa GPU especificamente
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n🚀 GPUs detectadas: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            
            # Testa operação na GPU
            try:
                with tf.device('/GPU:0'):
                    # Operação simples
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                    result = c.numpy()
                
                print("✅ Teste GPU bem-sucedido!")
                print(f"   Resultado teste: {result}")
                
                # Teste de performance
                print("\n⚡ Teste de performance GPU...")
                import time
                
                # Operação intensiva na GPU
                start_time = time.time()
                with tf.device('/GPU:0'):
                    large_matrix = tf.random.normal((1000, 1000))
                    for _ in range(100):
                        result = tf.matmul(large_matrix, large_matrix)
                gpu_time = time.time() - start_time
                
                # Mesma operação na CPU
                start_time = time.time()
                with tf.device('/CPU:0'):
                    large_matrix = tf.random.normal((1000, 1000))
                    for _ in range(100):
                        result = tf.matmul(large_matrix, large_matrix)
                cpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time
                print(f"   GPU: {gpu_time:.3f}s")
                print(f"   CPU: {cpu_time:.3f}s")
                print(f"   Speedup: {speedup:.1f}x")
                
                if speedup > 1.5:
                    print("✅ GPU está acelerando computações!")
                else:
                    print("⚠️ GPU pode não estar sendo usada efetivamente")
                
                return True, gpus
                
            except Exception as e:
                print(f"❌ Erro testando GPU: {e}")
                return False, []
        else:
            print("❌ Nenhuma GPU detectada")
            return False, []
            
    except ImportError:
        print("❌ TensorFlow não instalado")
        print("💡 Instale com: pip install tensorflow-macos tensorflow-metal")
        return False, []
    except Exception as e:
        print(f"❌ Erro verificando TensorFlow: {e}")
        return False, []

def check_metal_support():
    """
    Verifica suporte específico do Metal
    """
    print("\n🛠️ === VERIFICANDO METAL PERFORMANCE SHADERS ===")
    
    try:
        # Verifica se tensorflow-metal está instalado
        import subprocess
        result = subprocess.run([sys.executable, '-c', 'import tensorflow_metal'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ tensorflow-metal instalado")
        else:
            print("❌ tensorflow-metal não encontrado")
            print("💡 Instale com: pip install tensorflow-metal")
            return False
        
        # Testa funcionalidade Metal específica
        import tensorflow as tf
        
        # Verifica se operações Metal estão disponíveis
        try:
            # Force uma operação que usa Metal
            with tf.device('/GPU:0'):
                # Operação que se beneficia do Metal
                x = tf.random.normal((500, 500))
                y = tf.nn.relu(x)
                z = tf.reduce_sum(y)
                
            print("✅ Operações Metal funcionando")
            return True
            
        except Exception as e:
            print(f"⚠️ Problema com operações Metal: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Erro verificando Metal: {e}")
        return False

def check_memory_bandwidth():
    """
    Testa bandwidth de memória (importante para M1 Max)
    """
    print("\n💾 === TESTE DE BANDWIDTH DE MEMÓRIA ===")
    
    try:
        import tensorflow as tf
        import time
        
        # Teste de transferência CPU->GPU->CPU
        print("🔄 Testando transferência de dados...")
        
        # Cria dados grandes na CPU
        cpu_data = tf.random.normal((2000, 2000))  # ~32MB
        
        # Transfere para GPU e processa
        start_time = time.time()
        with tf.device('/GPU:0'):
            gpu_data = tf.identity(cpu_data)  # Transfere para GPU
            processed = tf.matmul(gpu_data, gpu_data)  # Operação na GPU
            result = tf.identity(processed)  # Mantém na GPU
        transfer_time = time.time() - start_time
        
        # Transfere de volta para CPU
        start_time = time.time()
        cpu_result = result.numpy()  # Transfere de volta
        back_transfer_time = time.time() - start_time
        
        print(f"   CPU->GPU + processamento: {transfer_time:.3f}s")
        print(f"   GPU->CPU: {back_transfer_time:.3f}s")
        
        # Calcula bandwidth aproximado
        data_size_mb = (2000 * 2000 * 4) / (1024 * 1024)  # float32 = 4 bytes
        bandwidth_gbps = (data_size_mb / 1024) / transfer_time
        
        print(f"   Tamanho dados: {data_size_mb:.1f}MB")
        print(f"   Bandwidth estimado: {bandwidth_gbps:.2f}GB/s")
        
        # M1 Max tem bandwidth de ~400GB/s na memória unificada
        if bandwidth_gbps > 50:
            print("✅ Bandwidth excelente (memória unificada)")
        elif bandwidth_gbps > 20:
            print("✅ Bandwidth boa")
        else:
            print("⚠️ Bandwidth baixo - pode haver gargalo")
            
        return True
        
    except Exception as e:
        print(f"❌ Erro testando bandwidth: {e}")
        return False

def recommend_optimizations():
    """
    Recomenda otimizações baseadas nos testes
    """
    print("\n💡 === RECOMENDAÇÕES DE OTIMIZAÇÃO ===")
    
    print("✅ Para M1 Max, recomendo:")
    print("   1. Use batch_size entre 32-128 (aproveita memória unificada)")
    print("   2. Use float16 para economizar memória e acelerar")
    print("   3. Aproveite todos os 10 cores (8+2) para CPU tasks")
    print("   4. Configure TensorFlow para usar todos os cores:")
    print("      os.environ['TF_NUM_INTEROP_THREADS'] = '10'")
    print("      os.environ['TF_NUM_INTRAOP_THREADS'] = '10'")
    
    print("\n⚡ Para máxima performance LSTM:")
    print("   1. Use implementation=2 nas camadas LSTM")
    print("   2. Evite bidirectional LSTM (dobra tempo)")
    print("   3. Use dropout manual ao invés de recurrent_dropout")
    print("   4. Configure memory_growth=True para GPU")
    
    print("\n🎯 Para seu dataset de 1500 registros:")
    print("   1. Batch size recomendado: 64")
    print("   2. Sequence length: 60-80")
    print("   3. LSTM layers: 2-3 camadas")
    print("   4. Epochs: 300-500 (GPU aguenta mais)")

def run_comprehensive_diagnosis():
    """
    Executa diagnóstico completo
    """
    print("🔍 === DIAGNÓSTICO COMPLETO M1 MAX ===")
    print("🎯 Verificando otimizações para GPU Metal\n")
    
    # Verifica sistema
    is_apple_silicon = check_system_info()
    
    if not is_apple_silicon:
        print("\n⚠️ Este diagnóstico é específico para M1/M2")
        return
    
    # Verifica TensorFlow
    tf_works, gpus = check_tensorflow_metal()
    
    # Verifica Metal
    metal_works = check_metal_support()
    
    # Testa performance de memória
    memory_ok = check_memory_bandwidth()
    
    # Resultado final
    print("\n🏆 === RESULTADO FINAL ===")
    
    if tf_works and metal_works and len(gpus) > 0:
        print("✅ TUDO OK! M1 Max GPU está funcionando perfeitamente")
        print("🚀 Você pode usar main_m1_max.py para máxima performance")
        print("⚡ Speedup esperado: 5-10x vs CPU")
    elif tf_works:
        print("⚡ TensorFlow OK, mas GPU pode ter problemas")
        print("🔧 Verifique instalação do tensorflow-metal")
        print("💪 Ainda assim, terá boa performance com CPU M1 Max")
    else:
        print("❌ Problemas detectados")
        print("🔧 Reinstale TensorFlow:")
        print("   pip uninstall tensorflow tensorflow-metal")
        print("   pip install tensorflow-macos tensorflow-metal")
    
    # Recomendações
    recommend_optimizations()
    
    print(f"\n📊 === RESUMO ===")
    print(f"Sistema Apple Silicon: {'✅' if is_apple_silicon else '❌'}")
    print(f"TensorFlow: {'✅' if tf_works else '❌'}")
    print(f"Metal GPU: {'✅' if metal_works else '❌'}")
    print(f"GPUs detectadas: {len(gpus) if tf_works else 0}")
    print(f"Memória: {'✅' if memory_ok else '❌'}")

if __name__ == "__main__":
    run_comprehensive_diagnosis()

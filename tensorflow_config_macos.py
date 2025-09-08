#!/usr/bin/env python3
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

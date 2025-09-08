#!/usr/bin/env python3
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

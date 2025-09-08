#!/usr/bin/env python3
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

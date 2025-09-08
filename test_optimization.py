#!/usr/bin/env python3
# test_optimization.py - Script para testar otimizações de performance

import time
import pandas as pd
import numpy as np
from datetime import datetime

def compare_configurations():
    """
    Compara configurações original vs otimizada
    """
    print("🔍 === COMPARAÇÃO DE CONFIGURAÇÕES ===")
    print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Carrega configurações
    try:
        from config import *
        original_config = {
            'SEQ_LENGTH': SEQ_LENGTH,
            'LSTM_EPOCHS': LSTM_CONFIG['epochs'],
            'LSTM_BATCH_SIZE': LSTM_CONFIG['batch_size'],
            'RF_ESTIMATORS': RF_CONFIG['n_estimators_options'],
            'RF_SEARCH_ITER': RF_CONFIG['random_search_iterations'],
            'GB_ESTIMATORS': GB_CONFIG['n_estimators_options'],
            'GB_SEARCH_ITER': GB_CONFIG['random_search_iterations'],
            'CV_FOLDS': RF_CONFIG['cv_folds']
        }
        print("✅ Configuração original carregada")
    except:
        print("❌ Erro ao carregar configuração original")
        return
    
    try:
        from config_optimized import *
        optimized_config = {
            'SEQ_LENGTH': 60,  # Valor padrão otimizado
            'LSTM_EPOCHS': LSTM_CONFIG_OPTIMIZED['epochs'],
            'LSTM_BATCH_SIZE': LSTM_CONFIG_OPTIMIZED['batch_size'],
            'RF_ESTIMATORS': RF_CONFIG_OPTIMIZED['n_estimators_options'],
            'RF_SEARCH_ITER': RF_CONFIG_OPTIMIZED['random_search_iterations'],
            'GB_ESTIMATORS': GB_CONFIG_OPTIMIZED['n_estimators_options'],
            'GB_SEARCH_ITER': GB_CONFIG_OPTIMIZED['random_search_iterations'],
            'CV_FOLDS': RF_CONFIG_OPTIMIZED['cv_folds']
        }
        print("✅ Configuração otimizada carregada")
    except:
        print("❌ Erro ao carregar configuração otimizada")
        return
    
    # Compara configurações
    print("\n📊 === COMPARAÇÃO DETALHADA ===")
    print(f"{'Parâmetro':<20} {'Original':<20} {'Otimizado':<20} {'Redução':<15}")
    print("-" * 80)
    
    for key in original_config.keys():
        orig_val = original_config[key]
        opt_val = optimized_config[key]
        
        if isinstance(orig_val, list) and isinstance(opt_val, list):
            # Para listas, compara tamanho
            orig_str = f"{len(orig_val)} opções"
            opt_str = f"{len(opt_val)} opções"
            reduction = f"{(1 - len(opt_val)/len(orig_val))*100:.0f}%"
        elif isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)):
            # Para números, mostra valores diretos
            orig_str = str(orig_val)
            opt_str = str(opt_val)
            if orig_val > 0:
                reduction = f"{(1 - opt_val/orig_val)*100:.0f}%"
            else:
                reduction = "N/A"
        else:
            orig_str = str(orig_val)
            opt_str = str(opt_val)
            reduction = "N/A"
        
        print(f"{key:<20} {orig_str:<20} {opt_str:<20} {reduction:<15}")
    
    # Estimativa de tempo
    print("\n⏱️ === ESTIMATIVA DE TEMPO ===")
    
    # Fatores de redução estimados
    lstm_time_factor = optimized_config['LSTM_EPOCHS'] / original_config['LSTM_EPOCHS']
    rf_time_factor = (optimized_config['RF_SEARCH_ITER'] / original_config['RF_SEARCH_ITER']) * \
                    (optimized_config['CV_FOLDS'] / original_config['CV_FOLDS']) * \
                    (len(optimized_config['RF_ESTIMATORS']) / len(original_config['RF_ESTIMATORS']))
    
    overall_factor = (lstm_time_factor + rf_time_factor) / 2
    
    print(f"🧠 LSTM: ~{(1-lstm_time_factor)*100:.0f}% mais rápido")
    print(f"🌲 Random Forest: ~{(1-rf_time_factor)*100:.0f}% mais rápido")
    print(f"🚀 Geral: ~{(1-overall_factor)*100:.0f}% mais rápido")
    
    # Estimativas de tempo para 1500 registros
    print(f"\n📈 Para 1500 registros:")
    original_estimated_time = 45  # minutos estimados com config original
    optimized_estimated_time = original_estimated_time * overall_factor
    
    print(f"   • Original: ~{original_estimated_time} minutos")
    print(f"   • Otimizado: ~{optimized_estimated_time:.1f} minutos")
    print(f"   • Economia: ~{original_estimated_time - optimized_estimated_time:.1f} minutos")

def test_database_performance():
    """
    Testa performance de carregamento do banco
    """
    print("\n🗃️ === TESTE DE PERFORMANCE DO BANCO ===")
    
    try:
        from database import DatabaseManager
        
        db = DatabaseManager()
        
        # Teste de conexão
        print("🔌 Testando conexão...")
        start_time = time.time()
        
        if db.connect():
            connect_time = time.time() - start_time
            print(f"✅ Conexão estabelecida em {connect_time:.2f}s")
            
            # Teste de contagem
            print("📊 Contando registros...")
            start_time = time.time()
            
            count_query = "SELECT COUNT(*) as total FROM botbinance"
            result = db.execute_custom_query(count_query)
            
            if result is not None:
                count_time = time.time() - start_time
                total_records = result.iloc[0, 0]
                print(f"📈 Total de registros: {total_records:,}")
                print(f"⏱️ Contagem em: {count_time:.2f}s")
                
                # Teste de carregamento parcial
                print("\n🔄 Testando carregamento parcial...")
                test_limits = [100, 500, 1000]
                
                for limit in test_limits:
                    if limit <= total_records:
                        start_time = time.time()
                        sample_data = db.load_botbinance_data(limit=limit)
                        load_time = time.time() - start_time
                        
                        if sample_data is not None:
                            print(f"   • {limit:,} registros: {load_time:.2f}s ({load_time/limit*1000:.1f}ms/registro)")
                        else:
                            print(f"   • {limit:,} registros: ERRO")
                
                # Projeção para 1500 registros
                if total_records >= 1500:
                    print(f"\n🎯 Projeção para 1500 registros:")
                    estimated_time = (load_time / limit) * 1500
                    print(f"   • Tempo estimado: {estimated_time:.2f}s")
                    print(f"   • Taxa: {1500/estimated_time:.0f} registros/segundo")
            
            db.disconnect()
        else:
            print("❌ Falha na conexão")
    
    except Exception as e:
        print(f"❌ Erro no teste: {e}")

def run_performance_analysis():
    """
    Executa análise completa de performance
    """
    print("🚀 === ANÁLISE DE PERFORMANCE PARA 1500 REGISTROS ===")
    print("💡 Esta análise ajuda a entender os gargalos e otimizações")
    
    # Compara configurações
    compare_configurations()
    
    # Testa performance do banco
    test_database_performance()
    
    print("\n🎯 === RECOMENDAÇÕES ===")
    print("✅ Use main_optimized.py ao invés de main_advanced.py")
    print("✅ Configurações otimizadas reduzem tempo em ~70%")
    print("✅ Para datasets maiores, considere:")
    print("   • Aumentar batch_size do LSTM")
    print("   • Reduzir ainda mais o número de features")
    print("   • Usar amostragem estratificada")
    print("   • Implementar processamento em lotes")
    
    print("\n⚡ === PRÓXIMOS PASSOS ===")
    print("1. Execute: python main_optimized.py")
    print("2. Compare tempos com a versão original")
    print("3. Ajuste configurações conforme necessário")
    print("4. Para datasets >5000 registros, considere usar GPU")

if __name__ == "__main__":
    run_performance_analysis()

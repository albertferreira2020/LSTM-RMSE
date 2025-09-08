# use_saved_models.py - Exemplo de como usar os modelos salvos

import pandas as pd
import numpy as np
from main_advanced import load_saved_models, predict_with_saved_models

def exemplo_uso_modelos_salvos():
    """
    Exemplo de como usar os modelos salvos para fazer novas previsões
    """
    print("=== EXEMPLO DE USO DOS MODELOS SALVOS ===")
    
    # 1. Carrega os modelos salvos
    print("📁 Carregando modelos salvos...")
    modelos = load_saved_models('models')
    
    if modelos is None:
        print("❌ Não foi possível carregar os modelos. Execute primeiro o treinamento.")
        return
    
    print("✅ Modelos carregados com sucesso!")
    print(f"🤖 Modelos disponíveis: {modelos['metadata']['models_available']}")
    
    # 2. Exemplo com dados simulados
    print("\n📊 Criando dados de exemplo...")
    
    # Simula dados de entrada (substitua pelos seus dados reais)
    dados_exemplo = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [102.0, 103.0, 104.0, 105.0, 106.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    print(f"Dados de exemplo: {dados_exemplo.shape}")
    print(dados_exemplo.tail())
    
    # 3. Faz previsões
    print("\n🔮 Fazendo previsões...")
    previsoes = predict_with_saved_models(dados_exemplo, 'models')
    
    if previsoes:
        print("\n📈 Previsões:")
        for modelo, predicao in previsoes.items():
            print(f"   {modelo}: {predicao:.2f}")
    
    return previsoes

def carregar_dados_reais_e_prever(arquivo_csv=None):
    """
    Carrega dados reais de um CSV e faz previsões
    """
    print("\n=== PREVISÃO COM DADOS REAIS ===")
    
    if arquivo_csv is None:
        arquivo_csv = 'relatorio_mensal_geral_2025-03 (1).csv'
    
    try:
        # Carrega dados
        print(f"📁 Carregando dados de: {arquivo_csv}")
        df = pd.read_csv(arquivo_csv, delimiter=';')
        
        # Pega apenas os últimos dados para previsão
        dados_recentes = df.tail(100)  # Últimos 100 pontos
        
        print(f"📊 Dados carregados: {dados_recentes.shape}")
        print("Últimos 5 registros:")
        print(dados_recentes[['close', 'open', 'high', 'low']].tail())
        
        # Faz previsão
        previsoes = predict_with_saved_models(dados_recentes, 'models')
        
        if previsoes:
            ultimo_preco = dados_recentes['close'].iloc[-1]
            print(f"\n📈 RESULTADO DA PREVISÃO:")
            print(f"💰 Último preço real: {ultimo_preco:.2f}")
            print(f"🔮 Previsões do próximo valor:")
            
            for modelo, predicao in previsoes.items():
                variacao = ((predicao - ultimo_preco) / ultimo_preco) * 100
                emoji = "📈" if variacao > 0 else "📉" if variacao < 0 else "➡️"
                print(f"   {emoji} {modelo}: {predicao:.2f} ({variacao:+.2f}%)")
        
        return previsoes
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados reais: {e}")
        return None

def monitorar_modelo_continuo():
    """
    Exemplo de como monitorar continuamente com os modelos salvos
    """
    print("\n=== MONITORAMENTO CONTÍNUO ===")
    print("Este é um exemplo de como você pode usar os modelos para monitoramento contínuo:")
    print("""
    import time
    import schedule
    
    def fazer_previsao_diaria():
        # Carrega dados mais recentes (de API, banco, etc.)
        dados_novos = carregar_dados_mais_recentes()
        
        # Faz previsão
        previsoes = predict_with_saved_models(dados_novos, 'models')
        
        # Salva resultados ou envia alertas
        salvar_previsoes(previsoes)
        
    # Agenda para rodar todos os dias às 9h
    schedule.every().day.at("09:00").do(fazer_previsao_diaria)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
    """)

if __name__ == "__main__":
    print("🚀 Demonstração de uso dos modelos salvos")
    
    # Exemplo básico
    exemplo_uso_modelos_salvos()
    
    # Tenta com dados reais se disponível
    try:
        carregar_dados_reais_e_prever()
    except:
        print("⚠️ Arquivo CSV não encontrado para teste com dados reais")
    
    # Mostra exemplo de monitoramento
    monitorar_modelo_continuo()
    
    print("\n✅ Demonstração concluída!")
    print("🔧 Customize as funções acima para suas necessidades específicas")

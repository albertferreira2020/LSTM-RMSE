# test_database.py - Script para testar conexão PostgreSQL

from database import DatabaseManager, test_database_connection
import sys

def main():
    """
    Script principal para testar conexão
    """
    print("=== TESTE DE CONEXÃO POSTGRESQL ===")
    print("Este script verifica se a conexão com PostgreSQL está funcionando")
    print()
    
    # Verifica se arquivo .env existe
    import os
    if not os.path.exists('.env'):
        print("❌ Arquivo .env não encontrado!")
        print()
        print("Crie um arquivo .env com as seguintes configurações:")
        print("DB_HOST=localhost")
        print("DB_PORT=5432")
        print("DB_NAME=seu_banco_de_dados")
        print("DB_USER=seu_usuario")
        print("DB_PASSWORD=sua_senha")
        print()
        return False
    
    # Executa teste de conexão
    success = test_database_connection()
    
    if success:
        print("\n✅ CONEXÃO TESTE BEM-SUCEDIDA!")
        print("Você pode executar o main_advanced.py agora")
        
        # Testa carregar uma amostra maior
        print("\n=== TESTE DE CARGA MAIOR ===")
        try:
            db = DatabaseManager()
            db.connect()
            
            # Conta total de registros
            total_query = "SELECT COUNT(*) as total FROM botbinance"
            total_df = db.execute_custom_query(total_query)
            total_records = total_df.iloc[0, 0] if total_df is not None else 0
            
            print(f"Total de registros na tabela: {total_records:,}")
            
            # Carrega últimos 1000 registros para teste
            sample_size = min(1000, total_records)
            if sample_size > 0:
                df = db.load_botbinance_data(limit=sample_size)
                if df is not None:
                    print(f"✅ Teste com {sample_size} registros bem-sucedido")
                    print(f"Período dos dados: {df.iloc[0]['created_at']} até {df.iloc[-1]['created_at']}")
            
            db.disconnect()
            
        except Exception as e:
            print(f"⚠️ Erro no teste de carga maior: {e}")
        
        return True
    else:
        print("\n❌ FALHA NA CONEXÃO!")
        print("Verifique:")
        print("1. Se o PostgreSQL está rodando")
        print("2. Se as credenciais no .env estão corretas")
        print("3. Se a tabela 'botbinance' existe")
        print("4. Se o usuário tem permissão para acessar a tabela")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

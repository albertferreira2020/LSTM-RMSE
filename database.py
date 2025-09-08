# database.py - Gerenciador de conexão com PostgreSQL

import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

class DatabaseManager:
    """
    Classe para gerenciar conexões com PostgreSQL
    """
    
    def __init__(self, env_file='.env'):
        """
        Inicializa o gerenciador de banco
        """
        # Carrega variáveis de ambiente
        load_dotenv(env_file)
        
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'sslmode': os.getenv('DB_SSLMODE', 'prefer')
        }
        
        # Verifica se as credenciais estão configuradas
        if not all([self.db_config['database'], self.db_config['user'], self.db_config['password']]):
            raise ValueError(
                "Credenciais do banco não configuradas! "
                "Verifique o arquivo .env com DB_NAME, DB_USER e DB_PASSWORD"
            )
        
        self.engine = None
        self.connection = None
    
    def create_connection_string(self):
        """
        Cria string de conexão para SQLAlchemy
        """
        return (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
    
    def connect(self):
        """
        Estabelece conexão com o banco
        """
        try:
            # Tenta primeiro com SQLAlchemy
            connection_string = self.create_connection_string()
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            
            # Também cria conexão direta com psycopg2
            self.psycopg2_conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            
            print(f"✅ Conectado ao PostgreSQL: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
            return True
        except Exception as e:
            print(f"❌ Erro ao conectar ao banco: {e}")
            return False
    
    def disconnect(self):
        """
        Fecha conexão com o banco
        """
        if hasattr(self, 'psycopg2_conn') and self.psycopg2_conn:
            self.psycopg2_conn.close()
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        print("✅ Conexão com banco fechada")
    
    def test_connection(self):
        """
        Testa a conexão com o banco
        """
        try:
            if not self.connection:
                self.connect()
            
            # Testa com uma query simples
            test_query = "SELECT 1 as test"
            result = pd.read_sql(test_query, self.psycopg2_conn)
            print(f"✅ Teste de conexão bem-sucedido: {result.iloc[0, 0]}")
            return True
        except Exception as e:
            print(f"❌ Teste de conexão falhou: {e}")
            return False
    
    def get_table_info(self, table_name):
        """
        Obtém informações sobre a tabela
        """
        try:
            if not self.connection:
                self.connect()
            
            # Query para obter informações da tabela
            info_query = f"""
            SELECT 
                column_name, 
                data_type, 
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
            """
            
            columns_info = pd.read_sql(info_query, self.psycopg2_conn)
            
            # Query para contar registros
            count_query = f"SELECT COUNT(*) as total_rows FROM {table_name}"
            row_count = pd.read_sql(count_query, self.psycopg2_conn)
            
            print(f"\n=== Informações da Tabela: {table_name} ===")
            print(f"Total de registros: {row_count.iloc[0, 0]:,}")
            print("\nColunas:")
            print(columns_info.to_string(index=False))
            
            return columns_info, row_count.iloc[0, 0]
            
        except Exception as e:
            print(f"❌ Erro ao obter informações da tabela: {e}")
            return None, 0
    
    def load_botbinance_data(self, limit=None, order_by='id'):
        """
        Carrega dados da tabela botbinance
        """
        try:
            if not self.connection:
                self.connect()
            
            # Monta a query
            query = "SELECT * FROM botbinance"
            
            if order_by:
                query += f" ORDER BY {order_by}"
            
            if limit:
                query += f" LIMIT {limit}"
            
            print(f"Executando query: {query}")
            
            # Carrega os dados
            df = pd.read_sql(query, self.psycopg2_conn)
            
            print(f"✅ Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
            print(f"Colunas disponíveis: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados da tabela botbinance: {e}")
            return None
    
    def execute_custom_query(self, query):
        """
        Executa uma query customizada
        """
        try:
            if not self.connection:
                self.connect()
            
            df = pd.read_sql(query, self.psycopg2_conn)
            print(f"✅ Query executada com sucesso: {len(df)} registros retornados")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao executar query: {e}")
            return None

def test_database_connection():
    """
    Função de teste para verificar a conexão
    """
    print("=== TESTE DE CONEXÃO COM BANCO ===")
    
    try:
        db = DatabaseManager()
        
        # Testa conexão
        if db.connect():
            # Testa query simples
            if db.test_connection():
                # Obtém informações da tabela
                db.get_table_info('botbinance')
                
                # Carrega uma amostra pequena
                sample_data = db.load_botbinance_data(limit=5)
                if sample_data is not None:
                    print("\n=== AMOSTRA DOS DADOS ===")
                    print(sample_data.head())
                    print(f"\nTipos de dados:")
                    print(sample_data.dtypes)
                
            db.disconnect()
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()

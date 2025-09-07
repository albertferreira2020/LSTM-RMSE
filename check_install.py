#!/usr/bin/env python3
"""
Script simples para verificar se todas as bibliotecas estão instaladas
"""

def check_library(name, import_name=None):
    """Verifica se uma biblioteca pode ser importada"""
    if import_name is None:
        import_name = name
    
    try:
        __import__(import_name)
        print(f"✅ {name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {name} - ERRO: {e}")
        return False

def main():
    """Função principal"""
    print("=== Verificação das Bibliotecas ===\n")
    
    libraries = [
        ("TensorFlow", "tensorflow"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
        ("Scikit-learn", "sklearn"),
        ("Matplotlib", "matplotlib")
    ]
    
    all_ok = True
    for name, import_name in libraries:
        if not check_library(name, import_name):
            all_ok = False
    
    print("\n" + "="*40)
    if all_ok:
        print("🎉 Todas as bibliotecas foram instaladas com sucesso!")
        print("\nVocê pode executar o projeto com:")
        print("python3 main.py")
    else:
        print("❌ Algumas bibliotecas não foram encontradas.")
        print("\nTente reinstalar as dependências:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()

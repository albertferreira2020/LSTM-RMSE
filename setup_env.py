#!/usr/bin/env python3
"""
Script para verificar e instalar dependências do projeto
"""

import subprocess
import sys
import os

def run_command(command):
    """Executa um comando e retorna o resultado"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Verifica a versão do Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Requer Python 3.8 ou superior")
        return False
    else:
        print("✅ Versão do Python compatível")
        return True

def check_virtual_env():
    """Verifica se está em um ambiente virtual"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("✅ Ambiente virtual ativo")
        return True
    else:
        print("⚠️  Ambiente virtual não detectado")
        print("Recomenda-se criar um ambiente virtual:")
        print("python3 -m venv env")
        print("source env/bin/activate  # macOS/Linux")
        print("env\\Scripts\\activate    # Windows")
        return False

def install_packages():
    """Instala os pacotes necessários"""
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0"
    ]
    
    print("\nInstalando pacotes básicos...")
    for package in packages:
        print(f"Instalando {package}...")
        success, stdout, stderr = run_command(f"pip install {package}")
        if not success:
            print(f"❌ Erro ao instalar {package}: {stderr}")
            return False
        else:
            print(f"✅ {package} instalado com sucesso")
    
    # TensorFlow separadamente devido aos problemas comuns
    print("\nInstalando TensorFlow...")
    success, stdout, stderr = run_command("pip install tensorflow>=2.12.0")
    if not success:
        print("❌ Erro ao instalar TensorFlow via pip")
        print("Tentando versão específica...")
        success, stdout, stderr = run_command("pip install tensorflow==2.15.0")
        if not success:
            print("❌ Falha na instalação do TensorFlow")
            print("Tente instalar manualmente:")
            print("pip cache purge")
            print("pip install tensorflow")
            return False
    
    print("✅ TensorFlow instalado com sucesso")
    return True

def test_imports():
    """Testa se todos os imports funcionam"""
    imports_to_test = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("sklearn.ensemble", "RandomForestRegressor"),
        ("sklearn.preprocessing", "MinMaxScaler"),
        ("sklearn.metrics", "mean_squared_error"),
        ("tensorflow.keras.models", "Sequential"),
        ("tensorflow.keras.layers", "LSTM, Dense, Dropout")
    ]
    
    print("\nTestando imports...")
    all_success = True
    
    for module, alias in imports_to_test:
        try:
            if alias.startswith("LSTM"):
                exec(f"from {module} import {alias}")
            else:
                exec(f"import {module} as {alias}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_success = False
    
    return all_success

def main():
    """Função principal"""
    print("=== Verificação do Ambiente Python ===\n")
    
    # Verifica Python
    if not check_python_version():
        sys.exit(1)
    
    # Verifica ambiente virtual
    check_virtual_env()
    
    # Atualiza pip
    print("\nAtualizando pip...")
    run_command("pip install --upgrade pip")
    
    # Instala pacotes
    if not install_packages():
        print("\n❌ Falha na instalação dos pacotes")
        sys.exit(1)
    
    # Testa imports
    if test_imports():
        print("\n🎉 Todas as dependências foram instaladas e testadas com sucesso!")
        print("\nVocê pode agora executar:")
        print("python main.py")
    else:
        print("\n❌ Alguns imports falharam. Verifique as mensagens de erro acima.")
        sys.exit(1)

if __name__ == "__main__":
    main()

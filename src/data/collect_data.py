import os
from pathlib import Path
import sys

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

from data.crypto_data_collector import CryptoDataCollector
from data.forex_data_collector import ForexDataCollector
from data.data_preprocessor import DataPreprocessor

def main():
    """
    Script principal para recolectar y procesar datos históricos de trading.
    """
    print("Iniciando recolección de datos históricos de trading...")
    
    # Crear directorios de datos si no existen
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data'
    os.makedirs(data_dir / 'crypto', exist_ok=True)
    os.makedirs(data_dir / 'forex', exist_ok=True)
    os.makedirs(data_dir / 'processed', exist_ok=True)
    
    # 1. Recolectar datos de criptomonedas
    print("\n=== Recolectando datos de criptomonedas ===")
    crypto_collector = CryptoDataCollector()
    
    # Lista de criptomonedas populares
    crypto_symbols_yf = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD']
    
    # Obtener datos de yfinance
    print("\nObteniendo datos de criptomonedas desde yfinance...")
    crypto_collector.get_multiple_crypto_data(crypto_symbols_yf, source='yfinance', period='2y', interval='1d')
    
    # 2. Recolectar datos de forex
    print("\n=== Recolectando datos de forex ===")
    forex_collector = ForexDataCollector()
    
    # Lista de pares forex populares
    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
    
    # Obtener datos
    print("\nObteniendo datos de forex desde yfinance...")
    forex_collector.get_multiple_forex_data(forex_symbols, period='2y', interval='1d')
    
    # 3. Preprocesar datos
    print("\n=== Preprocesando datos ===")
    preprocessor = DataPreprocessor()
    
    # Procesar datos de criptomonedas
    print("\nProcesando datos de criptomonedas...")
    crypto_data = preprocessor.process_all_files(source_type='crypto')
    
    # Procesar datos de forex
    print("\nProcesando datos de forex...")
    forex_data = preprocessor.process_all_files(source_type='forex')
    
    print("\n¡Recolección y procesamiento de datos completados!")
    print(f"Los datos se han guardado en: {data_dir}")

if __name__ == "__main__":
    main()

import pandas as pd
import yfinance as yf
import os
import datetime
import time
from pathlib import Path

class ForexDataCollector:
    """
    Clase para recolectar datos históricos de forex utilizando
    diferentes fuentes como yfinance.
    """
    
    def __init__(self, data_dir=None):
        """
        Inicializa el recolector de datos de forex.
        
        Args:
            data_dir: Directorio donde se guardarán los datos. Si es None,
                     se usará el directorio 'data' dentro del directorio actual.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'forex'
        else:
            self.data_dir = Path(data_dir)
        
        # Crear el directorio si no existe
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_data_from_yfinance(self, symbol, period="2y", interval="1d"):
        """
        Obtiene datos históricos de forex usando yfinance.
        
        Args:
            symbol: Símbolo del par forex (ej. 'EURUSD=X')
            period: Período de tiempo para los datos (ej. '1d', '1mo', '1y', '2y', 'max')
            interval: Intervalo de tiempo entre datos (ej. '1m', '1h', '1d')
            
        Returns:
            DataFrame de pandas con los datos históricos
        """
        try:
            data = yf.download(symbol, period=period, interval=interval)
            
            # Guardar los datos en un archivo CSV
            filename = f"{symbol.replace('=', '_')}_{interval}_{period}.csv"
            filepath = self.data_dir / filename
            data.to_csv(filepath)
            
            print(f"Datos de {symbol} guardados en {filepath}")
            return data
        except Exception as e:
            print(f"Error al obtener datos de {symbol} desde yfinance: {e}")
            return None
    
    def get_multiple_forex_data(self, symbols, **kwargs):
        """
        Obtiene datos históricos para múltiples pares forex.
        
        Args:
            symbols: Lista de símbolos de pares forex
            **kwargs: Argumentos adicionales para las funciones de obtención de datos
            
        Returns:
            Diccionario con los DataFrames de datos para cada símbolo
        """
        results = {}
        
        for symbol in symbols:
            print(f"Obteniendo datos para {symbol}...")
            data = self.get_data_from_yfinance(symbol, **kwargs)
            
            if data is not None:
                results[symbol] = data
            
            # Esperar un poco para no sobrecargar las APIs
            time.sleep(1)
        
        return results

# Ejemplo de uso
if __name__ == "__main__":
    collector = ForexDataCollector()
    
    # Lista de pares forex populares
    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
    
    # Obtener datos
    collector.get_multiple_forex_data(forex_symbols, period='2y', interval='1d')

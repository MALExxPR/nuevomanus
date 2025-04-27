import pandas as pd
import yfinance as yf
import ccxt
import os
import datetime
import time
from pathlib import Path

class CryptoDataCollector:
    """
    Clase para recolectar datos históricos de criptomonedas utilizando
    diferentes fuentes como yfinance y ccxt.
    """
    
    def __init__(self, data_dir=None):
        """
        Inicializa el recolector de datos de criptomonedas.
        
        Args:
            data_dir: Directorio donde se guardarán los datos. Si es None,
                     se usará el directorio 'data' dentro del directorio actual.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'crypto'
        else:
            self.data_dir = Path(data_dir)
        
        # Crear el directorio si no existe
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Inicializar el exchange de ccxt (Binance por defecto)
        self.exchange = ccxt.binance()
    
    def get_data_from_yfinance(self, symbol, period="2y", interval="1d"):
        """
        Obtiene datos históricos de criptomonedas usando yfinance.
        
        Args:
            symbol: Símbolo de la criptomoneda (ej. 'BTC-USD')
            period: Período de tiempo para los datos (ej. '1d', '1mo', '1y', '2y', 'max')
            interval: Intervalo de tiempo entre datos (ej. '1m', '1h', '1d')
            
        Returns:
            DataFrame de pandas con los datos históricos
        """
        try:
            data = yf.download(symbol, period=period, interval=interval)
            
            # Guardar los datos en un archivo CSV
            filename = f"{symbol.replace('-', '_')}_{interval}_{period}.csv"
            filepath = self.data_dir / filename
            data.to_csv(filepath)
            
            print(f"Datos de {symbol} guardados en {filepath}")
            return data
        except Exception as e:
            print(f"Error al obtener datos de {symbol} desde yfinance: {e}")
            return None
    
    def get_data_from_ccxt(self, symbol, timeframe='1d', since=None, limit=500):
        """
        Obtiene datos históricos de criptomonedas usando ccxt.
        
        Args:
            symbol: Símbolo de la criptomoneda (ej. 'BTC/USDT')
            timeframe: Intervalo de tiempo entre datos (ej. '1m', '1h', '1d')
            since: Timestamp desde donde obtener los datos (en milisegundos)
            limit: Número máximo de registros a obtener
            
        Returns:
            DataFrame de pandas con los datos históricos
        """
        try:
            # Si no se especifica since, obtener datos de los últimos 2 años
            if since is None:
                since = int((datetime.datetime.now() - datetime.timedelta(days=730)).timestamp() * 1000)
            
            # Obtener datos históricos
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Guardar los datos en un archivo CSV
            filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
            filepath = self.data_dir / filename
            df.to_csv(filepath)
            
            print(f"Datos de {symbol} guardados en {filepath}")
            return df
        except Exception as e:
            print(f"Error al obtener datos de {symbol} desde ccxt: {e}")
            return None
    
    def get_multiple_crypto_data(self, symbols, source='yfinance', **kwargs):
        """
        Obtiene datos históricos para múltiples criptomonedas.
        
        Args:
            symbols: Lista de símbolos de criptomonedas
            source: Fuente de datos ('yfinance' o 'ccxt')
            **kwargs: Argumentos adicionales para las funciones de obtención de datos
            
        Returns:
            Diccionario con los DataFrames de datos para cada símbolo
        """
        results = {}
        
        for symbol in symbols:
            print(f"Obteniendo datos para {symbol}...")
            
            if source == 'yfinance':
                data = self.get_data_from_yfinance(symbol, **kwargs)
            elif source == 'ccxt':
                data = self.get_data_from_ccxt(symbol, **kwargs)
            else:
                raise ValueError(f"Fuente de datos '{source}' no soportada")
            
            if data is not None:
                results[symbol] = data
            
            # Esperar un poco para no sobrecargar las APIs
            time.sleep(1)
        
        return results

# Ejemplo de uso
if __name__ == "__main__":
    collector = CryptoDataCollector()
    
    # Lista de criptomonedas populares
    crypto_symbols_yf = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD']
    crypto_symbols_ccxt = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'ADA/USDT']
    
    # Obtener datos de yfinance
    collector.get_multiple_crypto_data(crypto_symbols_yf, source='yfinance', period='2y', interval='1d')
    
    # Obtener datos de ccxt
    collector.get_multiple_crypto_data(crypto_symbols_ccxt, source='ccxt', timeframe='1d')

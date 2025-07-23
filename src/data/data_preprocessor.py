import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    Clase para preprocesar y limpiar datos históricos de trading
    para su uso en modelos de machine learning.
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Inicializa el preprocesador de datos.
        
        Args:
            data_dir: Directorio donde se encuentran los datos crudos.
            output_dir: Directorio donde se guardarán los datos procesados.
        """
        self.base_dir = Path(__file__).parent.parent.parent
        
        if data_dir is None:
            self.data_dir = self.base_dir / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        if output_dir is None:
            self.output_dir = self.base_dir / 'data' / 'processed'
        else:
            self.output_dir = Path(output_dir)
        
        # Crear el directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, filepath):
        """
        Carga datos desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV.
            
        Returns:
            DataFrame de pandas con los datos cargados.
        """
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"Error al cargar datos desde {filepath}: {e}")
            return None
    
    def clean_data(self, df):
        """
        Limpia los datos eliminando valores nulos y duplicados.
        
        Args:
            df: DataFrame de pandas con los datos a limpiar.
            
        Returns:
            DataFrame de pandas con los datos limpios.
        """
        if df is None:
            return None
        
        # Eliminar filas con valores nulos
        df_clean = df.dropna()
        
        # Eliminar filas duplicadas
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def add_technical_indicators(self, df):
        """
        Añade indicadores técnicos al DataFrame.
        
        Args:
            df: DataFrame de pandas con los datos.
            
        Returns:
            DataFrame de pandas con los indicadores técnicos añadidos.
        """
        if df is None or df.empty:
            return df
        
        # Verificar la estructura del DataFrame (sin imprimir en producción)
        
        # Asegurarse de que tenemos las columnas necesarias y que son numéricas
        # Para datos de yfinance, las columnas típicas son: Open, High, Low, Close, Adj Close, Volume
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        lowercase_required = ['open', 'high', 'low', 'close', 'volume']
        
        # Verificar si las columnas existen (con mayúsculas o minúsculas)
        if all(col in df.columns for col in required_columns):
            # Convertir a valores numéricos si es necesario
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Usar nombres en mayúsculas
            column_map = {col: col.lower() for col in required_columns}
            df = df.rename(columns=column_map)
            
        elif all(col in df.columns for col in lowercase_required):
            # Convertir a valores numéricos si es necesario
            for col in lowercase_required:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Si no encontramos las columnas esperadas, intentamos inferir
            # Asumimos que la primera columna numérica es el precio de cierre
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) >= 5:
                # Asignar las primeras 5 columnas numéricas a OHLCV
                df['open'] = df[numeric_cols[0]]
                df['high'] = df[numeric_cols[1]]
                df['low'] = df[numeric_cols[2]]
                df['close'] = df[numeric_cols[3]]
                df['volume'] = df[numeric_cols[4]]
            else:
                return df
        
        # Asegurarse de que no hay valores nulos antes de calcular indicadores
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Calcular medias móviles
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Calcular bandas de Bollinger (20 períodos, 2 desviaciones estándar)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calcular RSI (Relative Strength Index) - 14 períodos
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calcular MACD (Moving Average Convergence Divergence)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calcular ATR (Average True Range) - 14 períodos
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Eliminar filas con valores NaN (generados por los cálculos de indicadores)
        df = df.dropna()
        
        return df
    
    def normalize_data(self, df, columns=None):
        """
        Normaliza los datos utilizando MinMaxScaler.
        
        Args:
            df: DataFrame de pandas con los datos a normalizar.
            columns: Lista de columnas a normalizar. Si es None, se normalizan todas.
            
        Returns:
            DataFrame de pandas con los datos normalizados y el scaler utilizado.
        """
        if df is None or df.empty:
            return None, None
        
        if columns is None:
            columns = df.columns
        
        # Crear un nuevo DataFrame para los datos normalizados
        df_normalized = df.copy()
        
        # Inicializar el scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Normalizar las columnas especificadas
        df_normalized[columns] = scaler.fit_transform(df[columns])
        
        return df_normalized, scaler
    
    def prepare_data_for_ml(self, df, target_column='close', sequence_length=60, train_split=0.8):
        """
        Prepara los datos para su uso en modelos de machine learning.
        
        Args:
            df: DataFrame de pandas con los datos.
            target_column: Columna objetivo para la predicción.
            sequence_length: Longitud de la secuencia para modelos secuenciales.
            train_split: Proporción de datos para entrenamiento.
            
        Returns:
            Datos de entrenamiento y prueba preparados para modelos de ML.
        """
        if df is None or df.empty:
            return None
        
        # Asegurarse de que el target_column existe
        if target_column not in df.columns:
            print(f"La columna objetivo '{target_column}' no existe en el DataFrame.")
            return None
        
        # Normalizar los datos
        df_normalized, scaler = self.normalize_data(df)
        
        # Crear secuencias para modelos secuenciales (como LSTM)
        X, y = [], []
        for i in range(len(df_normalized) - sequence_length):
            X.append(df_normalized.iloc[i:i+sequence_length].values)
            y.append(df_normalized.iloc[i+sequence_length][target_column])
        
        X, y = np.array(X), np.array(y)
        
        # Dividir en conjuntos de entrenamiento y prueba
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'df_normalized': df_normalized
        }
    
    def process_all_files(self, source_type='crypto', add_indicators=True):
        """
        Procesa todos los archivos CSV en el directorio de datos.
        
        Args:
            source_type: Tipo de datos ('crypto' o 'forex').
            add_indicators: Si es True, añade indicadores técnicos.
            
        Returns:
            Diccionario con los DataFrames procesados.
        """
        # Determinar el directorio de origen
        source_dir = self.data_dir / source_type
        
        # Verificar que el directorio existe
        if not source_dir.exists():
            print(f"El directorio {source_dir} no existe.")
            return {}
        
        # Crear directorio de salida específico
        output_dir = self.output_dir / source_type
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Procesar cada archivo CSV en el directorio
        for file_path in source_dir.glob('*.csv'):
            print(f"Procesando {file_path}...")
            
            # Cargar datos
            df = self.load_data(file_path)
            
            if df is not None:
                # Limpiar datos
                df_clean = self.clean_data(df)
                
                # Añadir indicadores técnicos si se solicita
                if add_indicators:
                    df_processed = self.add_technical_indicators(df_clean)
                else:
                    df_processed = df_clean
                
                if df_processed is not None and not df_processed.empty:
                    # Guardar datos procesados
                    output_path = output_dir / f"processed_{file_path.name}"
                    df_processed.to_csv(output_path)
                    print(f"Datos procesados guardados en {output_path}")
                    
                    # Almacenar en resultados
                    results[file_path.stem] = df_processed
        
        return results

# Ejemplo de uso
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Procesar datos de criptomonedas
    crypto_data = preprocessor.process_all_files(source_type='crypto')
    
    # Procesar datos de forex
    forex_data = preprocessor.process_all_files(source_type='forex')

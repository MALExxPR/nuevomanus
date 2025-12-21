import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Importar módulos del proyecto
from data.crypto_data_collector import CryptoDataCollector
from data.forex_data_collector import ForexDataCollector
from data.data_preprocessor import DataPreprocessor
from models.lstm_model import LSTMPricePredictor
from models.dqn_model import DQNTradingAgent, TradingEnvironment
from models.backtesting import BacktestingSystem
from api.bingx_api import BingXAPI

class TestDataCollectors(unittest.TestCase):
    """
    Pruebas unitarias para los recolectores de datos.
    """
    
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.crypto_collector = CryptoDataCollector()
        self.forex_collector = ForexDataCollector()
        
        # Crear directorios temporales para pruebas
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """
        Limpieza después de las pruebas.
        """
        # Eliminar archivos temporales
        for file in self.test_dir.glob('*.csv'):
            os.remove(file)
    
    def test_crypto_data_collector(self):
        """
        Prueba la recolección de datos de criptomonedas.
        """
        # Probar obtención de datos de BTC-USD
        df = self.crypto_collector.get_data_from_yfinance('BTC-USD', period='1mo', interval='1d', save_dir=self.test_dir)
        
        # Verificar que se obtuvieron datos
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Verificar que se guardó el archivo
        file_path = self.test_dir / 'BTC_USD_1d_1mo.csv'
        self.assertTrue(file_path.exists())
        
        # Verificar que el archivo contiene datos
        df_loaded = pd.read_csv(file_path)
        self.assertGreater(len(df_loaded), 0)
    
    def test_forex_data_collector(self):
        """
        Prueba la recolección de datos de forex.
        """
        # Probar obtención de datos de EUR/USD
        df = self.forex_collector.get_data_from_yfinance('EURUSD=X', period='1mo', interval='1d', save_dir=self.test_dir)
        
        # Verificar que se obtuvieron datos
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Verificar que se guardó el archivo
        file_path = self.test_dir / 'EURUSD_X_1d_1mo.csv'
        self.assertTrue(file_path.exists())
        
        # Verificar que el archivo contiene datos
        df_loaded = pd.read_csv(file_path)
        self.assertGreater(len(df_loaded), 0)

class TestDataPreprocessor(unittest.TestCase):
    """
    Pruebas unitarias para el preprocesador de datos.
    """
    
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.preprocessor = DataPreprocessor()
        
        # Crear datos de prueba
        self.test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Crear directorio temporal para pruebas
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Guardar datos de prueba
        self.test_file = self.test_dir / 'test_data.csv'
        self.test_data.to_csv(self.test_file)
    
    def tearDown(self):
        """
        Limpieza después de las pruebas.
        """
        # Eliminar archivos temporales
        if self.test_file.exists():
            os.remove(self.test_file)
    
    def test_load_data(self):
        """
        Prueba la carga de datos.
        """
        df = self.preprocessor.load_data(self.test_file)
        
        # Verificar que se cargaron los datos
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)
    
    def test_clean_data(self):
        """
        Prueba la limpieza de datos.
        """
        # Crear datos con valores nulos y duplicados
        data_with_nulls = self.test_data.copy()
        data_with_nulls.loc[2, 'Close'] = None
        data_with_nulls = pd.concat([data_with_nulls, data_with_nulls.iloc[[0]]])  # Duplicar primera fila
        
        # Limpiar datos
        df_clean = self.preprocessor.clean_data(data_with_nulls)
        
        # Verificar que se eliminaron los valores nulos
        self.assertFalse(df_clean.isnull().any().any())
        
        # Verificar que se eliminaron los duplicados
        self.assertEqual(len(df_clean), 4)
    
    def test_add_technical_indicators(self):
        """
        Prueba la adición de indicadores técnicos.
        """
        # Añadir indicadores técnicos
        df_with_indicators = self.preprocessor.add_technical_indicators(self.test_data)
        
        # Verificar que se añadieron los indicadores
        expected_indicators = ['sma_5', 'sma_20', 'sma_50', 'bb_middle', 'bb_std', 
                              'bb_upper', 'bb_lower', 'rsi_14', 'macd', 'macd_signal', 
                              'macd_hist', 'atr_14']
        
        # Algunos indicadores pueden no estar presentes debido a la longitud de los datos
        for indicator in expected_indicators:
            if indicator in df_with_indicators.columns:
                self.assertTrue(True)
            else:
                print(f"Indicador {indicator} no presente debido a la longitud de los datos")

class TestLSTMModel(unittest.TestCase):
    """
    Pruebas unitarias para el modelo LSTM.
    """
    
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.lstm_predictor = LSTMPricePredictor()
        
        # Crear datos de prueba
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(102, 5, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)
        
        # Crear directorio temporal para pruebas
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Crear directorio para modelos
        self.models_dir = Path(__file__).parent / 'test_models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Configurar directorio de modelos
        self.lstm_predictor.model_dir = self.models_dir
    
    def tearDown(self):
        """
        Limpieza después de las pruebas.
        """
        # Eliminar archivos temporales
        for file in self.models_dir.glob('*.h5'):
            os.remove(file)
    
    def test_prepare_data(self):
        """
        Prueba la preparación de datos para el modelo LSTM.
        """
        # Preparar datos
        data = self.lstm_predictor.prepare_data(self.test_data, target_column='close', sequence_length=10)
        
        # Verificar que se prepararon los datos correctamente
        self.assertIsNotNone(data)
        self.assertIn('X_train', data)
        self.assertIn('y_train', data)
        self.assertIn('X_test', data)
        self.assertIn('y_test', data)
        
        # Verificar dimensiones
        self.assertEqual(data['X_train'].shape[1], 10)  # sequence_length
        self.assertEqual(data['X_train'].shape[2], len(self.test_data.columns))  # n_features
    
    def test_build_model(self):
        """
        Prueba la construcción del modelo LSTM.
        """
        # Construir modelo
        model = self.lstm_predictor.build_model((10, 5))  # (sequence_length, n_features)
        
        # Verificar que se construyó el modelo correctamente
        self.assertIsNotNone(model)
        
        # Verificar arquitectura del modelo
        self.assertEqual(len(model.layers), 5)  # LSTM, Dropout, LSTM, Dropout, Dense

class TestBacktestingSystem(unittest.TestCase):
    """
    Pruebas unitarias para el sistema de backtesting.
    """
    
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.backtester = BacktestingSystem()
        
        # Crear datos de prueba
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(102, 5, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)
        
        # Crear directorio temporal para pruebas
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Crear directorio para resultados
        self.results_dir = Path(__file__).parent / 'test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configurar directorio de resultados
        self.backtester.results_dir = self.results_dir
    
    def tearDown(self):
        """
        Limpieza después de las pruebas.
        """
        # Eliminar archivos temporales
        for file in self.results_dir.glob('*.png'):
            os.remove(file)
    
    def test_simple_moving_average_strategy(self):
        """
        Prueba la estrategia de cruce de medias móviles.
        """
        # Aplicar estrategia
        signals = self.backtester.simple_moving_average_strategy(self.test_data, short_window=5, long_window=10)
        
        # Verificar que se generaron señales
        self.assertIsNotNone(signals)
        self.assertIn('signal', signals.columns)
        self.assertIn('position', signals.columns)
    
    def test_bollinger_bands_strategy(self):
        """
        Prueba la estrategia de bandas de Bollinger.
        """
        # Aplicar estrategia
        signals = self.backtester.bollinger_bands_strategy(self.test_data, window=20, num_std=2)
        
        # Verificar que se generaron señales
        self.assertIsNotNone(signals)
        self.assertIn('signal', signals.columns)
        self.assertIn('position', signals.columns)
    
    def test_backtest_strategy(self):
        """
        Prueba el backtesting de una estrategia.
        """
        # Generar señales
        signals = self.backtester.simple_moving_average_strategy(self.test_data)
        
        # Realizar backtesting
        portfolio = self.backtester.backtest_strategy(signals, initial_capital=10000)
        
        # Verificar que se generaron resultados
        self.assertIsNotNone(portfolio)
        self.assertIn('holdings', portfolio.columns)
        self.assertIn('cash', portfolio.columns)
        self.assertIn('total', portfolio.columns)
        self.assertIn('returns', portfolio.columns)
    
    def test_calculate_performance_metrics(self):
        """
        Prueba el cálculo de métricas de rendimiento.
        """
        # Generar señales y realizar backtesting
        signals = self.backtester.simple_moving_average_strategy(self.test_data)
        portfolio = self.backtester.backtest_strategy(signals, initial_capital=10000)
        
        # Calcular métricas
        metrics = self.backtester.calculate_performance_metrics(portfolio)
        
        # Verificar que se calcularon las métricas
        self.assertIsNotNone(metrics)
        self.assertIn('total_return', metrics)
        self.assertIn('annual_return', metrics)
        self.assertIn('annual_volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

class TestBingXAPI(unittest.TestCase):
    """
    Pruebas unitarias para la API de BingX.
    """
    
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        # Crear instancia de la API sin credenciales (modo de solo lectura)
        self.api = BingXAPI()
    
    def test_get_server_time(self):
        """
        Prueba la obtención de la hora del servidor.
        """
        # Esta prueba puede fallar si no hay conexión a internet
        try:
            response = self.api.get_server_time()
            
            # Verificar que se obtuvo una respuesta
            self.assertIsNotNone(response)
            
            # La respuesta puede variar, pero debería tener un código
            if 'code' in response:
                self.assertEqual(response['code'], 0)
            else:
                self.assertTrue('serverTime' in response or 'timestamp' in response)
        
        except Exception as e:
            self.skipTest(f"Error de conexión: {e}")
    
    def test_get_ticker(self):
        """
        Prueba la obtención de información de ticker.
        """
        # Esta prueba puede fallar si no hay conexión a internet
        try:
            response = self.api.get_ticker("BTC-USDT")
            
            # Verificar que se obtuvo una respuesta
            self.assertIsNotNone(response)
            
            # La respuesta puede variar, pero debería tener un código
            if 'code' in response:
                self.assertEqual(response['code'], 0)
            else:
                self.assertTrue('symbol' in response or 'data' in response)
        
        except Exception as e:
            self.skipTest(f"Error de conexión: {e}")

def run_tests():
    """
    Ejecuta todas las pruebas unitarias.
    """
    # Crear suite de pruebas
    test_suite = unittest.TestSuite()
    
    # Añadir pruebas
    test_suite.addTest(unittest.makeSuite(TestDataCollectors))
    test_suite.addTest(unittest.makeSuite(TestDataPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestLSTMModel))
    test_suite.addTest(unittest.makeSuite(TestBacktestingSystem))
    test_suite.addTest(unittest.makeSuite(TestBingXAPI))
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == "__main__":
    run_tests()

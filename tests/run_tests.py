import unittest
import sys
from pathlib import Path
import os

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

# Importar pruebas individuales
from test_system import TestDataCollectors, TestDataPreprocessor, TestLSTMModel, TestBacktestingSystem, TestBingXAPI

def run_integration_tests():
    """
    Ejecuta pruebas de integración para verificar que los componentes funcionan correctamente juntos.
    """
    # Crear directorios necesarios
    test_dir = Path(__file__).parent / 'test_data'
    os.makedirs(test_dir, exist_ok=True)
    
    models_dir = Path(__file__).parent / 'test_models'
    os.makedirs(models_dir, exist_ok=True)
    
    results_dir = Path(__file__).parent / 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Importar módulos necesarios
    from data.crypto_data_collector import CryptoDataCollector
    from data.data_preprocessor import DataPreprocessor
    from models.lstm_model import LSTMPricePredictor
    from models.backtesting import BacktestingSystem
    
    print("\n=== Ejecutando pruebas de integración ===\n")
    
    # Prueba 1: Recolectar datos, preprocesarlos y entrenar un modelo
    print("Prueba 1: Flujo completo de datos a modelo")
    
    try:
        # 1. Recolectar datos
        print("1.1 Recolectando datos...")
        collector = CryptoDataCollector()
        df = collector.get_data_from_yfinance('BTC-USD', period='1mo', interval='1d', save_dir=test_dir)
        
        if df is None or len(df) == 0:
            print("❌ Error: No se pudieron recolectar datos")
            return
        
        print(f"✅ Datos recolectados: {len(df)} registros")
        
        # 2. Preprocesar datos
        print("1.2 Preprocesando datos...")
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        df_processed = preprocessor.add_technical_indicators(df_clean)
        
        if df_processed is None or len(df_processed) == 0:
            print("❌ Error: No se pudieron preprocesar datos")
            return
        
        print(f"✅ Datos preprocesados: {len(df_processed)} registros con {len(df_processed.columns)} características")
        
        # 3. Preparar datos para modelo
        print("1.3 Preparando datos para modelo...")
        lstm_predictor = LSTMPricePredictor(model_dir=models_dir)
        data = lstm_predictor.prepare_data(df_processed, target_column='close', sequence_length=10)
        
        if data is None:
            print("❌ Error: No se pudieron preparar datos para el modelo")
            return
        
        print(f"✅ Datos preparados para modelo: {data['X_train'].shape[0]} muestras de entrenamiento, {data['X_test'].shape[0]} muestras de prueba")
        
        # 4. Construir y entrenar modelo (entrenamiento mínimo para prueba)
        print("1.4 Construyendo y entrenando modelo (entrenamiento mínimo)...")
        history = lstm_predictor.train_model(data, epochs=1, batch_size=32, model_name='test_model')
        
        if history is None:
            print("❌ Error: No se pudo entrenar el modelo")
            return
        
        print("✅ Modelo entrenado correctamente")
        
        # 5. Realizar predicciones
        print("1.5 Realizando predicciones...")
        results = lstm_predictor.predict(data)
        
        if results is None:
            print("❌ Error: No se pudieron realizar predicciones")
            return
        
        print(f"✅ Predicciones realizadas: {len(results['predictions'])} predicciones")
        
        # 6. Evaluar modelo
        print("1.6 Evaluando modelo...")
        metrics = lstm_predictor.evaluate(results['predictions'], results['real_values'])
        
        if metrics is None:
            print("❌ Error: No se pudo evaluar el modelo")
            return
        
        print(f"✅ Modelo evaluado: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        print("\nPrueba 1 completada con éxito ✅")
    
    except Exception as e:
        print(f"❌ Error en Prueba 1: {e}")
    
    # Prueba 2: Backtesting de estrategias
    print("\nPrueba 2: Backtesting de estrategias")
    
    try:
        # 1. Cargar datos preprocesados
        print("2.1 Cargando datos preprocesados...")
        backtester = BacktestingSystem(results_dir=results_dir)
        
        # Usar los datos procesados de la prueba anterior
        if 'df_processed' not in locals() or df_processed is None:
            print("❌ Error: No hay datos preprocesados disponibles")
            return
        
        print(f"✅ Datos cargados: {len(df_processed)} registros")
        
        # 2. Aplicar estrategia de medias móviles
        print("2.2 Aplicando estrategia de medias móviles...")
        signals_sma = backtester.simple_moving_average_strategy(df_processed)
        
        if signals_sma is None:
            print("❌ Error: No se pudieron generar señales para la estrategia SMA")
            return
        
        print(f"✅ Señales generadas para estrategia SMA")
        
        # 3. Aplicar estrategia de RSI
        print("2.3 Aplicando estrategia de RSI...")
        signals_rsi = backtester.rsi_strategy(df_processed)
        
        if signals_rsi is None:
            print("❌ Error: No se pudieron generar señales para la estrategia RSI")
            return
        
        print(f"✅ Señales generadas para estrategia RSI")
        
        # 4. Realizar backtesting de estrategia SMA
        print("2.4 Realizando backtesting de estrategia SMA...")
        portfolio_sma = backtester.backtest_strategy(signals_sma, initial_capital=10000)
        
        if portfolio_sma is None:
            print("❌ Error: No se pudo realizar backtesting para la estrategia SMA")
            return
        
        print(f"✅ Backtesting realizado para estrategia SMA")
        
        # 5. Calcular métricas de rendimiento
        print("2.5 Calculando métricas de rendimiento...")
        metrics_sma = backtester.calculate_performance_metrics(portfolio_sma)
        
        if metrics_sma is None:
            print("❌ Error: No se pudieron calcular métricas para la estrategia SMA")
            return
        
        print(f"✅ Métricas calculadas para estrategia SMA: Retorno total={metrics_sma['total_return']:.4f}, Ratio de Sharpe={metrics_sma['sharpe_ratio']:.4f}")
        
        print("\nPrueba 2 completada con éxito ✅")
    
    except Exception as e:
        print(f"❌ Error en Prueba 2: {e}")
    
    print("\n=== Pruebas de integración completadas ===\n")

if __name__ == "__main__":
    # Ejecutar pruebas unitarias
    unittest.main(module='test_system')
    
    # Ejecutar pruebas de integración
    run_integration_tests()

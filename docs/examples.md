# Ejemplos de Uso - Sistema de Trading con Machine Learning

Este documento proporciona ejemplos prácticos de cómo utilizar las diferentes funcionalidades del sistema de trading con machine learning.

## 1. Recolección de Datos

### Ejemplo 1: Obtener datos históricos de Bitcoin

```python
from src.data.crypto_data_collector import CryptoDataCollector

# Crear instancia del recolector
collector = CryptoDataCollector()

# Obtener datos de Bitcoin de los últimos 6 meses con intervalos diarios
btc_data = collector.get_data_from_yfinance(
    symbol='BTC-USD',
    period='6mo',
    interval='1d',
    save_dir='/home/usuario/trading_ml_project/data/crypto'
)

print(f"Datos obtenidos: {len(btc_data)} registros")
print(btc_data.head())
```

### Ejemplo 2: Obtener datos históricos de EUR/USD

```python
from src.data.forex_data_collector import ForexDataCollector

# Crear instancia del recolector
collector = ForexDataCollector()

# Obtener datos de EUR/USD de los últimos 3 meses con intervalos de 1 hora
eurusd_data = collector.get_data_from_yfinance(
    symbol='EURUSD=X',
    period='3mo',
    interval='1h',
    save_dir='/home/usuario/trading_ml_project/data/forex'
)

print(f"Datos obtenidos: {len(eurusd_data)} registros")
print(eurusd_data.head())
```

## 2. Procesamiento de Datos

### Ejemplo 1: Limpiar datos y añadir indicadores técnicos

```python
import pandas as pd
from src.data.data_preprocessor import DataPreprocessor

# Cargar datos
data_path = '/home/usuario/trading_ml_project/data/crypto/BTC_USD_1d_6mo.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Crear instancia del preprocesador
preprocessor = DataPreprocessor()

# Limpiar datos
df_clean = preprocessor.clean_data(df)
print(f"Datos después de limpieza: {len(df_clean)} registros")

# Añadir indicadores técnicos
df_processed = preprocessor.add_technical_indicators(df_clean)
print(f"Columnas después de añadir indicadores: {df_processed.columns.tolist()}")

# Guardar datos procesados
output_path = '/home/usuario/trading_ml_project/data/processed/BTC_USD_processed.csv'
df_processed.to_csv(output_path)
print(f"Datos procesados guardados en: {output_path}")
```

### Ejemplo 2: Normalizar datos para modelos de ML

```python
import pandas as pd
from src.data.data_preprocessor import DataPreprocessor

# Cargar datos procesados
data_path = '/home/usuario/trading_ml_project/data/processed/BTC_USD_processed.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Crear instancia del preprocesador
preprocessor = DataPreprocessor()

# Normalizar datos
df_normalized = preprocessor.normalize_data(df)
print("Primeras filas de datos normalizados:")
print(df_normalized.head())

# Guardar datos normalizados
output_path = '/home/usuario/trading_ml_project/data/processed/BTC_USD_normalized.csv'
df_normalized.to_csv(output_path)
print(f"Datos normalizados guardados en: {output_path}")
```

## 3. Entrenamiento de Modelos

### Ejemplo 1: Entrenar modelo LSTM para predicción de precios

```python
import pandas as pd
from src.models.lstm_model import LSTMPricePredictor

# Cargar datos procesados
data_path = '/home/usuario/trading_ml_project/data/processed/BTC_USD_processed.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Crear instancia del predictor LSTM
lstm_predictor = LSTMPricePredictor(
    model_dir='/home/usuario/trading_ml_project/models/lstm'
)

# Preparar datos para el modelo
data = lstm_predictor.prepare_data(
    df,
    target_column='close',
    sequence_length=60,
    train_split=0.8
)

# Entrenar modelo
history = lstm_predictor.train_model(
    data,
    epochs=50,
    batch_size=32,
    model_name='btc_lstm_model'
)

# Evaluar modelo
results = lstm_predictor.predict(data)
metrics = lstm_predictor.evaluate(results['predictions'], results['real_values'])

print(f"Métricas de evaluación: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
```

### Ejemplo 2: Entrenar agente DQN para trading

```python
import pandas as pd
from src.models.dqn_model import DQNTradingAgent, TradingEnvironment

# Cargar datos procesados
data_path = '/home/usuario/trading_ml_project/data/processed/BTC_USD_processed.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Crear entorno de trading
env = TradingEnvironment(df)

# Obtener tamaño del estado
state = env.reset()
state_size = state.shape[1]

# Crear agente DQN
dqn_agent = DQNTradingAgent(
    state_size=state_size,
    action_size=3,  # mantener, comprar, vender
    model_dir='/home/usuario/trading_ml_project/models/dqn'
)

# Entrenar agente
episodes = 100
batch_size = 32

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Seleccionar acción
        action = dqn_agent.act(state, training=True)
        
        # Ejecutar acción
        next_state, reward, done, _ = env.step(action)
        
        # Guardar experiencia
        dqn_agent.remember(state, action, reward, next_state, done)
        
        # Actualizar estado y recompensa
        state = next_state
        total_reward += reward
        
        # Entrenar agente con experiencias pasadas
        if len(dqn_agent.memory) > batch_size:
            dqn_agent.replay(batch_size)
    
    print(f"Episodio {episode+1}/{episodes}, Recompensa total: {total_reward:.2f}")

# Guardar modelo entrenado
dqn_agent.save('btc_dqn_model')
print("Modelo DQN guardado correctamente")
```

## 4. Backtesting de Estrategias

### Ejemplo 1: Backtesting de estrategia de cruce de medias móviles

```python
import pandas as pd
from src.models.backtesting import BacktestingSystem
import matplotlib.pyplot as plt

# Cargar datos procesados
data_path = '/home/usuario/trading_ml_project/data/processed/BTC_USD_processed.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Crear instancia del sistema de backtesting
backtester = BacktestingSystem(
    results_dir='/home/usuario/trading_ml_project/results/backtesting'
)

# Aplicar estrategia de cruce de medias móviles
signals = backtester.simple_moving_average_strategy(
    df,
    short_window=20,
    long_window=50
)

# Realizar backtesting
portfolio = backtester.backtest_strategy(
    signals,
    initial_capital=10000,
    position_size=1.0,
    commission=0.001
)

# Calcular métricas de rendimiento
metrics = backtester.calculate_performance_metrics(portfolio)

print("Métricas de rendimiento:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Visualizar resultados
plt.figure(figsize=(12, 6))
plt.plot(portfolio['total'])
plt.title('Evolución del capital - Estrategia SMA')
plt.xlabel('Fecha')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.savefig('/home/usuario/trading_ml_project/results/backtesting/sma_strategy_performance.png')
plt.close()
```

### Ejemplo 2: Comparación de múltiples estrategias

```python
import pandas as pd
import numpy as np
from src.models.backtesting import BacktestingSystem
import matplotlib.pyplot as plt

# Cargar datos procesados
data_path = '/home/usuario/trading_ml_project/data/processed/BTC_USD_processed.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Crear instancia del sistema de backtesting
backtester = BacktestingSystem(
    results_dir='/home/usuario/trading_ml_project/results/backtesting'
)

# Definir estrategias a comparar
strategies = {
    'SMA': backtester.simple_moving_average_strategy,
    'Bollinger': backtester.bollinger_bands_strategy,
    'RSI': backtester.rsi_strategy,
    'MACD': backtester.macd_strategy
}

# Realizar backtesting para cada estrategia
results = {}
metrics = {}

for name, strategy_func in strategies.items():
    # Aplicar estrategia
    signals = strategy_func(df)
    
    # Realizar backtesting
    portfolio = backtester.backtest_strategy(
        signals,
        initial_capital=10000,
        position_size=1.0,
        commission=0.001
    )
    
    # Guardar resultados
    results[name] = portfolio
    
    # Calcular métricas
    metrics[name] = backtester.calculate_performance_metrics(portfolio)

# Comparar rendimiento
plt.figure(figsize=(12, 6))

for name, portfolio in results.items():
    plt.plot(portfolio['total'], label=name)

plt.title('Comparación de Estrategias')
plt.xlabel('Fecha')
plt.ylabel('Capital ($)')
plt.legend()
plt.grid(True)
plt.savefig('/home/usuario/trading_ml_project/results/backtesting/strategy_comparison.png')
plt.close()

# Mostrar métricas
print("Métricas de rendimiento por estrategia:")
for name, metric in metrics.items():
    print(f"\n{name}:")
    for key, value in metric.items():
        print(f"  {key}: {value:.4f}")
```

## 5. Trading en Vivo

### Ejemplo 1: Configurar y ejecutar el bot de trading

```python
from src.api.trading_bot import TradingBot

# Crear instancia del bot de trading
bot = TradingBot()

# Configurar credenciales de API
bot.update_api_credentials(
    api_key='TU_API_KEY',
    api_secret='TU_API_SECRET'
)

# Configurar parámetros de trading
config = {
    'trading_pairs': ['BTC-USDT', 'ETH-USDT'],
    'strategy': 'lstm',  # 'lstm', 'dqn', 'sma', 'rsi', 'macd'
    'interval': '1h',
    'trade_amount': 100,  # USDT
    'max_trades_per_day': 5,
    'stop_loss_pct': 2.0,
    'take_profit_pct': 3.0,
    'risk_level': 'medium',  # 'low', 'medium', 'high'
    'auto_trading': True
}

bot.save_config(config)

# Cargar modelos
bot.load_lstm_model()
bot.load_dqn_agent()

# Ejecutar un ciclo de trading
results = bot.run_trading_cycle()
print("Resultados del ciclo de trading:")
for result in results:
    print(result)

# Obtener resumen de la cuenta
summary = bot.get_account_summary()
print("\nResumen de la cuenta:")
for key, value in summary.items():
    print(f"{key}: {value}")
```

### Ejemplo 2: Iniciar el bot de trading en modo continuo

```python
from src.api.trading_bot import TradingBot
import time

# Crear instancia del bot de trading
bot = TradingBot()

# Verificar si las credenciales están configuradas
if bot.config.get('api_key') is None or bot.config.get('api_secret') is None:
    print("Error: Credenciales de API no configuradas")
    exit(1)

# Iniciar el bot en modo continuo
try:
    print("Iniciando bot de trading en modo continuo...")
    print("Presiona Ctrl+C para detener")
    
    # Configurar intervalo (en minutos)
    interval_minutes = 60
    
    bot.start_trading_bot(interval_minutes=interval_minutes)
    
except KeyboardInterrupt:
    print("\nDeteniendo bot de trading...")
    bot.stop_trading_bot()
    
    # Guardar resultados
    bot._save_trading_results()
    
    print("Bot de trading detenido correctamente")
```

## 6. Uso de la Interfaz de Usuario

### Ejemplo: Ejecutar la aplicación Streamlit

```bash
# Navegar al directorio del proyecto
cd /home/usuario/trading_ml_project

# Activar entorno virtual
source venv/bin/activate

# Ejecutar la aplicación
streamlit run src/ui/app.py
```

Una vez que la aplicación esté en ejecución, se abrirá en tu navegador web y podrás interactuar con todas las funcionalidades del sistema a través de la interfaz gráfica.

## 7. Integración Completa

### Ejemplo: Flujo de trabajo completo

```python
import pandas as pd
import os
from src.data.crypto_data_collector import CryptoDataCollector
from src.data.data_preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPricePredictor
from src.models.backtesting import BacktestingSystem
from src.api.trading_bot import TradingBot

# Definir directorios
base_dir = '/home/usuario/trading_ml_project'
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')
results_dir = os.path.join(base_dir, 'results')

# 1. Recolectar datos
print("1. Recolectando datos...")
collector = CryptoDataCollector()
df = collector.get_data_from_yfinance(
    symbol='BTC-USDT',
    period='1y',
    interval='1d',
    save_dir=os.path.join(data_dir, 'crypto')
)

# 2. Procesar datos
print("2. Procesando datos...")
preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)
df_processed = preprocessor.add_technical_indicators(df_clean)
df_processed.to_csv(os.path.join(data_dir, 'processed', 'BTC_USDT_processed.csv'))

# 3. Entrenar modelo LSTM
print("3. Entrenando modelo LSTM...")
lstm_predictor = LSTMPricePredictor(model_dir=os.path.join(models_dir, 'lstm'))
data = lstm_predictor.prepare_data(df_processed, target_column='close')
history = lstm_predictor.train_model(data, epochs=50, model_name='btc_lstm_model')

# 4. Realizar backtesting
print("4. Realizando backtesting...")
backtester = BacktestingSystem(results_dir=os.path.join(results_dir, 'backtesting'))
signals = backtester.simple_moving_average_strategy(df_processed)
portfolio = backtester.backtest_strategy(signals, initial_capital=10000)
metrics = backtester.calculate_performance_metrics(portfolio)
print("Métricas de backtesting:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# 5. Configurar bot de trading
print("5. Configurando bot de trading...")
bot = TradingBot()
config = {
    'trading_pairs': ['BTC-USDT'],
    'strategy': 'lstm',
    'interval': '1d',
    'trade_amount': 100,
    'max_trades_per_day': 3,
    'stop_loss_pct': 2.0,
    'take_profit_pct': 3.0,
    'risk_level': 'medium',
    'auto_trading': False  # Modo simulación
}
bot.save_config(config)

# 6. Ejecutar ciclo de trading (simulación)
print("6. Ejecutando ciclo de trading (simulación)...")
bot.load_lstm_model()
results = bot.run_trading_cycle()
print("Resultados del ciclo de trading:")
for result in results:
    print(result)

print("\n¡Flujo de trabajo completo ejecutado con éxito!")
```

Estos ejemplos te ayudarán a entender cómo utilizar cada componente del sistema de trading con machine learning y cómo integrarlos para crear un flujo de trabajo completo.

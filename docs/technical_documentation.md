# Documentación Técnica - Sistema de Trading con Machine Learning

## Arquitectura del Sistema

El sistema de trading con machine learning está diseñado con una arquitectura modular que separa claramente las diferentes funcionalidades:

```
+-------------------+      +-------------------+      +-------------------+
|                   |      |                   |      |                   |
|  Recolección de   +----->+  Procesamiento   +----->+    Modelos de     |
|      Datos        |      |    de Datos      |      |  Machine Learning |
|                   |      |                   |      |                   |
+-------------------+      +-------------------+      +--------+----------+
                                                               |
                                                               v
+-------------------+      +-------------------+      +-------------------+
|                   |      |                   |      |                   |
|    Interfaz de    |<-----+    Trading Bot    |<-----+    Sistema de    |
|     Usuario       |      |                   |      |    Backtesting   |
|                   |      |                   |      |                   |
+-------------------+      +-------------------+      +-------------------+
                                    ^
                                    |
                           +--------+----------+
                           |                   |
                           |    API de BingX   |
                           |                   |
                           +-------------------+
```

## Componentes Principales

### 1. Módulo de Recolección de Datos

**Ubicación**: `src/data/crypto_data_collector.py` y `src/data/forex_data_collector.py`

**Funcionalidad**:
- Recolecta datos históricos de criptomonedas y forex desde yfinance
- Guarda los datos en formato CSV para su posterior procesamiento
- Soporta diferentes intervalos de tiempo y períodos

**Clases principales**:
- `CryptoDataCollector`: Recolecta datos de criptomonedas
- `ForexDataCollector`: Recolecta datos de forex

**Ejemplo de uso**:
```python
from data.crypto_data_collector import CryptoDataCollector

collector = CryptoDataCollector()
df = collector.get_data_from_yfinance('BTC-USD', period='1y', interval='1d')
```

### 2. Módulo de Procesamiento de Datos

**Ubicación**: `src/data/data_preprocessor.py`

**Funcionalidad**:
- Limpia los datos eliminando valores nulos y duplicados
- Añade indicadores técnicos (SMA, Bollinger Bands, RSI, MACD, ATR)
- Normaliza los datos para su uso en modelos de machine learning
- Prepara secuencias para modelos secuenciales (LSTM)

**Clase principal**:
- `DataPreprocessor`: Procesa y prepara los datos para modelos de ML

**Ejemplo de uso**:
```python
from data.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)
df_processed = preprocessor.add_technical_indicators(df_clean)
```

### 3. Módulo de Modelos de Machine Learning

**Ubicación**: `src/models/lstm_model.py` y `src/models/dqn_model.py`

**Funcionalidad**:
- Implementa modelos LSTM para predicción de precios
- Implementa modelos DQN para aprendizaje por refuerzo
- Entrena, evalúa y guarda modelos
- Realiza predicciones con modelos entrenados

**Clases principales**:
- `LSTMPricePredictor`: Predice precios futuros con LSTM
- `DQNTradingAgent`: Aprende estrategias de trading con DQN
- `TradingEnvironment`: Entorno de trading para el agente DQN

**Ejemplo de uso**:
```python
from models.lstm_model import LSTMPricePredictor

predictor = LSTMPricePredictor()
data = predictor.prepare_data(df_processed, target_column='close')
history = predictor.train_model(data, epochs=50)
results = predictor.predict(data)
```

### 4. Módulo de Backtesting

**Ubicación**: `src/models/backtesting.py`

**Funcionalidad**:
- Implementa diferentes estrategias de trading (SMA, Bollinger, RSI, MACD)
- Realiza backtesting de estrategias con datos históricos
- Calcula métricas de rendimiento (retorno total, ratio de Sharpe, drawdown)
- Visualiza resultados de backtesting

**Clase principal**:
- `BacktestingSystem`: Sistema completo de backtesting

**Ejemplo de uso**:
```python
from models.backtesting import BacktestingSystem

backtester = BacktestingSystem()
signals = backtester.simple_moving_average_strategy(df_processed)
portfolio = backtester.backtest_strategy(signals, initial_capital=10000)
metrics = backtester.calculate_performance_metrics(portfolio)
```

### 5. Módulo de API de BingX

**Ubicación**: `src/api/bingx_api.py`

**Funcionalidad**:
- Proporciona una interfaz para la API de BingX
- Implementa autenticación y firma de solicitudes
- Soporta operaciones de trading (obtener datos, colocar órdenes, etc.)

**Clase principal**:
- `BingXAPI`: Cliente para la API de BingX

**Ejemplo de uso**:
```python
from api.bingx_api import BingXAPI

api = BingXAPI(api_key='YOUR_API_KEY', api_secret='YOUR_API_SECRET')
ticker = api.get_ticker('BTC-USDT')
```

### 6. Módulo de Trading Bot

**Ubicación**: `src/api/trading_bot.py`

**Funcionalidad**:
- Integra todos los componentes del sistema
- Implementa lógica de trading automático
- Gestiona posiciones y operaciones
- Registra historial de trading

**Clase principal**:
- `TradingBot`: Bot de trading automático

**Ejemplo de uso**:
```python
from api.trading_bot import TradingBot

bot = TradingBot()
bot.update_api_credentials('YOUR_API_KEY', 'YOUR_API_SECRET')
bot.run_trading_cycle()
```

### 7. Módulo de Interfaz de Usuario

**Ubicación**: `src/ui/app.py`

**Funcionalidad**:
- Proporciona una interfaz web con Streamlit
- Permite interactuar con todos los componentes del sistema
- Visualiza datos, modelos y resultados

**Clase principal**:
- `TradingMLApp`: Aplicación web para el sistema de trading

**Ejemplo de uso**:
```python
from ui.app import TradingMLApp

app = TradingMLApp()
app.run()
```

## Flujo de Datos

1. Los datos históricos se recolectan desde yfinance
2. Los datos se procesan y se añaden indicadores técnicos
3. Los datos procesados se utilizan para entrenar modelos LSTM y DQN
4. Los modelos entrenados se utilizan para generar predicciones y señales de trading
5. Las señales se evalúan mediante backtesting
6. Las estrategias seleccionadas se implementan en el trading bot
7. El trading bot ejecuta operaciones a través de la API de BingX
8. Los resultados se visualizan en la interfaz de usuario

## Detalles de Implementación

### Modelo LSTM

El modelo LSTM utiliza la siguiente arquitectura:
- Capa LSTM con 50 unidades y retorno de secuencias
- Capa Dropout (20%)
- Capa LSTM con 50 unidades sin retorno de secuencias
- Capa Dropout (20%)
- Capa Dense con 1 unidad (salida)

La función de pérdida es el error cuadrático medio (MSE) y el optimizador es Adam.

### Modelo DQN

El modelo DQN utiliza la siguiente arquitectura:
- Capa Dense con 64 unidades y activación ReLU
- Capa BatchNormalization
- Capa Dense con 64 unidades y activación ReLU
- Capa BatchNormalization
- Capa Dense con 32 unidades y activación ReLU
- Capa Dense con 3 unidades (acciones: mantener, comprar, vender)

El agente DQN utiliza un buffer de experiencia para almacenar transiciones (estado, acción, recompensa, siguiente estado, terminado) y aprende mediante Q-learning con redes neuronales.

### Estrategias de Trading

Las estrategias implementadas incluyen:

1. **Cruce de Medias Móviles (SMA)**:
   - Señal de compra: SMA corta cruza por encima de SMA larga
   - Señal de venta: SMA corta cruza por debajo de SMA larga

2. **Bandas de Bollinger**:
   - Señal de compra: Precio cae por debajo de la banda inferior
   - Señal de venta: Precio sube por encima de la banda superior

3. **RSI (Relative Strength Index)**:
   - Señal de compra: RSI cae por debajo del nivel de sobreventa (30)
   - Señal de venta: RSI sube por encima del nivel de sobrecompra (70)

4. **MACD (Moving Average Convergence Divergence)**:
   - Señal de compra: MACD cruza por encima de la línea de señal
   - Señal de venta: MACD cruza por debajo de la línea de señal

### API de BingX

La implementación de la API de BingX sigue la documentación oficial y utiliza el protocolo HMAC-SHA256 para la autenticación. Las solicitudes se realizan mediante la biblioteca `requests` y se manejan las respuestas en formato JSON.

## Consideraciones de Rendimiento

- Los modelos LSTM pueden ser computacionalmente intensivos durante el entrenamiento
- El procesamiento de grandes volúmenes de datos históricos puede requerir optimización
- Las solicitudes a la API de BingX están sujetas a límites de tasa

## Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

- Nuevos modelos de ML pueden ser añadidos en el directorio `src/models/`
- Nuevas estrategias de trading pueden ser implementadas en `src/models/backtesting.py`
- Soporte para otros exchanges puede ser añadido creando nuevas clases de API

## Pruebas

El sistema incluye pruebas unitarias y de integración en el directorio `tests/`:

- `test_system.py`: Pruebas unitarias para todos los componentes
- `run_tests.py`: Script para ejecutar todas las pruebas

## Limitaciones Conocidas

- El sistema no implementa análisis fundamental
- Las predicciones de precios tienen limitaciones inherentes debido a la naturaleza estocástica de los mercados
- El rendimiento en backtesting no garantiza resultados similares en trading en vivo

## Referencias

- [Documentación de la API de BingX](https://bingx-api.github.io/docs/)
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Deep Q-Network Paper](https://www.nature.com/articles/nature14236)
- [Streamlit Documentation](https://docs.streamlit.io/)

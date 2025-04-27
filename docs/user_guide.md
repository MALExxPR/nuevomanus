# Guía de Usuario - Sistema de Trading con Machine Learning

## Introducción

Bienvenido al Sistema de Trading con Machine Learning. Esta guía te ayudará a entender cómo utilizar todas las funcionalidades del sistema para realizar trading algorítmico con criptomonedas y forex utilizando modelos de machine learning.

## Instalación y Configuración

### Requisitos previos
- Python 3.10 o superior
- Conexión a internet
- Cuenta en BingX (para trading en vivo)

### Instalación

1. Descomprime el archivo del proyecto en tu directorio preferido
2. Abre una terminal y navega hasta el directorio del proyecto
3. Ejecuta el script de instalación:

```bash
bash run.sh
```

Este script creará un entorno virtual, instalará todas las dependencias necesarias y lanzará la aplicación.

## Interfaz de Usuario

La interfaz de usuario está dividida en cinco secciones principales:

### 1. Inicio

La página de inicio proporciona una visión general del sistema y estadísticas básicas sobre los datos y modelos disponibles.

### 2. Datos

En esta sección puedes:
- **Recolectar Datos**: Obtener datos históricos de criptomonedas y forex.
- **Ver Datos**: Visualizar y explorar los datos recolectados.
- **Procesar Datos**: Limpiar los datos y añadir indicadores técnicos.

#### Recolección de datos

1. Selecciona el tipo de datos (Criptomonedas o Forex)
2. Elige los símbolos que deseas recolectar
3. Selecciona el período e intervalo
4. Haz clic en "Recolectar Datos"

![Recolección de datos](https://ejemplo.com/recoleccion_datos.png)

#### Visualización de datos

1. Selecciona el tipo de datos
2. Elige un archivo de datos
3. Explora la vista previa, visualización y estadísticas

#### Procesamiento de datos

1. Selecciona el tipo de datos
2. Elige los archivos a procesar
3. Configura las opciones de procesamiento
4. Haz clic en "Procesar Datos"

### 3. Modelos

En esta sección puedes:
- **Modelo LSTM**: Entrenar y evaluar modelos LSTM para predicción de precios.
- **Modelo DQN**: Entrenar y evaluar agentes DQN para aprendizaje por refuerzo.

#### Entrenamiento de modelo LSTM

1. Selecciona un archivo de datos procesados
2. Configura los parámetros del modelo (longitud de secuencia, épocas, etc.)
3. Haz clic en "Entrenar Modelo LSTM"
4. Visualiza los resultados y métricas de evaluación

![Entrenamiento de modelo](https://ejemplo.com/entrenamiento_modelo.png)

### 4. Backtesting

En esta sección puedes:
- Seleccionar datos para backtesting
- Elegir estrategias a evaluar
- Configurar parámetros de backtesting
- Comparar rendimiento de diferentes estrategias

#### Ejecución de backtesting

1. Selecciona un archivo de datos procesados
2. Marca las estrategias que deseas evaluar
3. Configura el capital inicial y comisión
4. Haz clic en "Ejecutar Backtesting"
5. Analiza los resultados y métricas de rendimiento

![Backtesting](https://ejemplo.com/backtesting.png)

### 5. Trading en Vivo

En esta sección puedes:
- Configurar la conexión con la API de BingX
- Establecer parámetros de trading
- Iniciar el trading automático

#### Configuración de API

1. Ingresa tu API Key y API Secret de BingX
2. Haz clic en "Guardar Configuración"

#### Configuración de trading

1. Selecciona el par de trading
2. Elige la estrategia a utilizar
3. Configura la cantidad por operación y máximo de operaciones por día
4. Haz clic en "Iniciar Trading Automático"

## Flujo de Trabajo Recomendado

Para obtener los mejores resultados, te recomendamos seguir este flujo de trabajo:

1. **Recolectar datos históricos** de los activos que te interesan
2. **Procesar los datos** añadiendo indicadores técnicos
3. **Entrenar modelos LSTM** para predicción de precios
4. **Realizar backtesting** de diferentes estrategias
5. **Seleccionar la mejor estrategia** basada en métricas de rendimiento
6. **Configurar el trading en vivo** con la estrategia seleccionada

## Estrategias Disponibles

### Cruce de Medias Móviles (SMA)
Esta estrategia genera señales de compra cuando la media móvil corta cruza por encima de la media móvil larga, y señales de venta cuando cruza por debajo.

### Bandas de Bollinger
Esta estrategia genera señales de compra cuando el precio cae por debajo de la banda inferior, y señales de venta cuando sube por encima de la banda superior.

### RSI (Relative Strength Index)
Esta estrategia genera señales de compra cuando el RSI cae por debajo del nivel de sobreventa (30), y señales de venta cuando sube por encima del nivel de sobrecompra (70).

### MACD (Moving Average Convergence Divergence)
Esta estrategia genera señales de compra cuando la línea MACD cruza por encima de la línea de señal, y señales de venta cuando cruza por debajo.

### Predicción LSTM
Esta estrategia utiliza un modelo LSTM entrenado para predecir precios futuros y generar señales de compra/venta basadas en la dirección predicha.

### Agente DQN
Esta estrategia utiliza un agente de aprendizaje por refuerzo entrenado para tomar decisiones de trading óptimas basadas en el estado actual del mercado.

## Gestión de Riesgos

El sistema incluye funcionalidades básicas de gestión de riesgos:

- **Stop Loss**: Puedes configurar un porcentaje de stop loss para limitar pérdidas.
- **Take Profit**: Puedes configurar un porcentaje de take profit para asegurar ganancias.
- **Límite de operaciones**: Puedes establecer un máximo de operaciones por día.

## Solución de Problemas

### Problemas comunes

1. **Error al conectar con la API de BingX**
   - Verifica que tus credenciales sean correctas
   - Asegúrate de tener conexión a internet
   - Comprueba que la API de BingX esté operativa

2. **Error al entrenar modelos**
   - Verifica que los datos estén correctamente procesados
   - Intenta con un conjunto de datos más grande
   - Ajusta los hiperparámetros del modelo

3. **Rendimiento insatisfactorio en backtesting**
   - Prueba diferentes estrategias
   - Ajusta los parámetros de las estrategias
   - Utiliza un período de datos más representativo

## Contacto y Soporte

Si tienes alguna pregunta o problema, no dudes en contactarnos:

- Email: soporte@tradingml.com
- Sitio web: www.tradingml.com/soporte

## Actualizaciones Futuras

Estamos trabajando constantemente para mejorar el sistema. Próximas actualizaciones incluirán:

- Más modelos de machine learning
- Estrategias avanzadas de trading
- Optimización automática de parámetros
- Análisis de sentimiento de mercado
- Integración con más exchanges

¡Gracias por utilizar nuestro Sistema de Trading con Machine Learning!

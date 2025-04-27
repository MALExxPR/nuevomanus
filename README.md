# Sistema de Trading con Machine Learning

Este proyecto implementa un sistema completo de trading con machine learning, capaz de analizar datos históricos de criptomonedas y forex, generar predicciones de precios, evaluar estrategias mediante backtesting y ejecutar operaciones automáticas a través de la API de BingX.

## Características principales

- **Recolección de datos históricos**: Obtiene datos de criptomonedas y forex desde diversas fuentes.
- **Preprocesamiento de datos**: Limpia los datos y añade indicadores técnicos relevantes para el trading.
- **Modelos de machine learning**:
  - Modelo LSTM para predicción de precios
  - Modelo DQN (Deep Q-Network) para aprendizaje por refuerzo
  - Sistema de backtesting para evaluar diferentes estrategias
- **Interfaz de usuario**: Interfaz web intuitiva desarrollada con Streamlit.
- **Integración con BingX**: Conexión directa con la API de BingX para trading automático.

## Estructura del proyecto

```
trading_ml_project/
├── data/                  # Datos históricos y procesados
│   ├── crypto/            # Datos de criptomonedas
│   ├── forex/             # Datos de forex
│   └── processed/         # Datos procesados con indicadores técnicos
├── models/                # Modelos entrenados
│   ├── lstm/              # Modelos LSTM para predicción de precios
│   └── dqn/               # Modelos DQN para aprendizaje por refuerzo
├── results/               # Resultados de backtesting y trading
│   └── backtesting/       # Resultados de backtesting de estrategias
├── logs/                  # Logs del sistema
├── src/                   # Código fuente
│   ├── api/               # Integración con API de BingX
│   ├── data/              # Módulos para recolección y procesamiento de datos
│   ├── models/            # Implementación de modelos de ML
│   └── ui/                # Interfaz de usuario con Streamlit
├── tests/                 # Pruebas unitarias y de integración
├── requirements.txt       # Dependencias del proyecto
└── run.sh                 # Script para ejecutar el sistema
```

## Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/trading_ml_project.git
cd trading_ml_project
```

2. Crear un entorno virtual y activarlo:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar la aplicación

```bash
bash run.sh
```

O manualmente:

```bash
streamlit run src/ui/app.py
```

### Ejecutar pruebas

```bash
cd tests
python run_tests.py
```

## Configuración de la API de BingX

Para utilizar la funcionalidad de trading automático, es necesario configurar las credenciales de la API de BingX:

1. Obtener API Key y API Secret desde la plataforma BingX
2. Configurar las credenciales en la interfaz de usuario (página "Trading en Vivo")

## Modelos de Machine Learning

### Modelo LSTM

El modelo LSTM (Long Short-Term Memory) se utiliza para predecir los precios futuros de los activos basándose en datos históricos. Este modelo es especialmente efectivo para capturar patrones temporales en series de tiempo.

### Modelo DQN

El modelo DQN (Deep Q-Network) implementa aprendizaje por refuerzo para aprender estrategias de trading óptimas. El agente aprende a tomar decisiones (comprar, vender, mantener) para maximizar las ganancias a largo plazo.

### Sistema de Backtesting

El sistema de backtesting permite evaluar diferentes estrategias de trading utilizando datos históricos. Implementa varias estrategias comunes:
- Cruce de medias móviles
- Bandas de Bollinger
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

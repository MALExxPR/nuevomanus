import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Trading con Machine Learning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
    }
    .success-card {
        background-color: #e8f5e9;
        border-left: 5px solid #43a047;
    }
    .warning-card {
        background-color: #fff8e1;
        border-left: 5px solid #ffb300;
    }
    .info-text {
        font-size: 1rem;
        color: #546e7a;
    }
    .highlight {
        background-color: #f1f8e9;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar datos - CORREGIDA
@st.cache_data
def load_data(symbol, period='1y', interval='1d'):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty or len(data) == 0:
            st.error(f"No se pudieron obtener datos para {symbol}. Por favor, intenta con otro símbolo o período.")
            return None
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Función para añadir indicadores técnicos - CORREGIDA
def add_indicators(df):
    if df is None or df.empty:
        return None
    
    # Crear copia para evitar advertencias
    df_with_indicators = df.copy()
    
    try:
        # Medias móviles
        df_with_indicators['SMA20'] = ta.trend.sma_indicator(df_with_indicators['Close'], window=20)
        df_with_indicators['SMA50'] = ta.trend.sma_indicator(df_with_indicators['Close'], window=50)
        df_with_indicators['SMA200'] = ta.trend.sma_indicator(df_with_indicators['Close'], window=200)
        
        # Bandas de Bollinger
        bollinger = ta.volatility.BollingerBands(df_with_indicators['Close'], window=20, window_dev=2)
        df_with_indicators['BB_upper'] = bollinger.bollinger_hband()
        df_with_indicators['BB_middle'] = bollinger.bollinger_mavg()
        df_with_indicators['BB_lower'] = bollinger.bollinger_lband()
        
        # RSI
        df_with_indicators['RSI'] = ta.momentum.rsi(df_with_indicators['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df_with_indicators['Close'])
        df_with_indicators['MACD'] = macd.macd()
        df_with_indicators['MACD_signal'] = macd.macd_signal()
        df_with_indicators['MACD_hist'] = macd.macd_diff()
        
        # ATR
        df_with_indicators['ATR'] = ta.volatility.average_true_range(
            df_with_indicators['High'], 
            df_with_indicators['Low'], 
            df_with_indicators['Close']
        )
        
        return df_with_indicators
    except Exception as e:
        st.error(f"Error al añadir indicadores: {e}")
        return df

# Función para generar señales con estrategia SMA - CORREGIDA
def sma_strategy(df, short_window=20, long_window=50):
    if df is None or df.empty:
        return None
    
    signals = df.copy()
    signals['signal'] = 0.0
    
    try:
        # Generar señales
        signals['signal'] = np.where(
            signals['SMA20'] > signals['SMA50'], 1.0, 0.0
        )
        
        # Generar posiciones
        signals['position'] = signals['signal'].diff()
        
        return signals
    except Exception as e:
        st.error(f"Error en estrategia SMA: {e}")
        return df

# Función para generar señales con estrategia Bollinger Bands - CORREGIDA
def bollinger_strategy(df):
    if df is None or df.empty:
        return None
    
    signals = df.copy()
    signals['signal'] = 0.0
    
    try:
        # Generar señales
        signals['signal'] = np.where(signals['Close'] < signals['BB_lower'], 1.0, 
                                np.where(signals['Close'] > signals['BB_upper'], -1.0, 0.0))
        
        # Generar posiciones
        signals['position'] = signals['signal'].diff()
        
        return signals
    except Exception as e:
        st.error(f"Error en estrategia Bollinger: {e}")
        return df

# Función para generar señales con estrategia RSI - CORREGIDA
def rsi_strategy(df, overbought=70, oversold=30):
    if df is None or df.empty:
        return None
    
    signals = df.copy()
    signals['signal'] = 0.0
    
    try:
        # Generar señales
        signals['signal'] = np.where(signals['RSI'] < oversold, 1.0, 
                                np.where(signals['RSI'] > overbought, -1.0, 0.0))
        
        # Generar posiciones
        signals['position'] = signals['signal'].diff()
        
        return signals
    except Exception as e:
        st.error(f"Error en estrategia RSI: {e}")
        return df

# Función para generar señales con estrategia MACD - CORREGIDA
def macd_strategy(df):
    if df is None or df.empty:
        return None
    
    signals = df.copy()
    signals['signal'] = 0.0
    
    try:
        # Generar señales
        signals['signal'] = np.where(signals['MACD'] > signals['MACD_signal'], 1.0, 
                                np.where(signals['MACD'] < signals['MACD_signal'], -1.0, 0.0))
        
        # Generar posiciones
        signals['position'] = signals['signal'].diff()
        
        return signals
    except Exception as e:
        st.error(f"Error en estrategia MACD: {e}")
        return df

# Función para realizar backtesting - CORREGIDA
def backtest_strategy(signals, initial_capital=10000, position_size=1.0, commission=0.001):
    if signals is None or signals.empty:
        return None
    
    try:
        # Crear DataFrame para el portafolio
        portfolio = signals[['Close']].copy()
        
        # Calcular posiciones
        portfolio['position'] = position_size * signals['signal']
        
        # Calcular retornos diarios
        portfolio['returns'] = portfolio['Close'].pct_change()
        
        # Calcular retornos de la estrategia
        portfolio['strategy_returns'] = portfolio['position'].shift(1) * portfolio['returns']
        
        # Calcular retornos acumulados
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod()
        portfolio['cumulative_strategy_returns'] = (1 + portfolio['strategy_returns']).cumprod()
        
        # Calcular valor del portafolio
        portfolio['holdings'] = initial_capital * portfolio['cumulative_strategy_returns']
        
        # Calcular efectivo (considerando comisiones)
        portfolio['cash'] = initial_capital
        
        # Calcular valor total
        portfolio['total'] = portfolio['holdings']
        
        # Calcular drawdown
        portfolio['peak'] = portfolio['total'].cummax()
        portfolio['drawdown'] = (portfolio['total'] - portfolio['peak']) / portfolio['peak']
        
        return portfolio
    except Exception as e:
        st.error(f"Error en backtesting: {e}")
        return None

# Función para calcular métricas de rendimiento - CORREGIDA
def calculate_performance_metrics(portfolio):
    if portfolio is None or portfolio.empty:
        return {
            'total_return': 0,
            'annual_return': 0,
            'annual_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'num_trades': 0,
            'win_rate': 0
        }
    
    try:
        # Calcular retorno total
        total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
        
        # Calcular retorno anualizado
        days = max(1, (portfolio.index[-1] - portfolio.index[0]).days)
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Calcular volatilidad anualizada
        annual_volatility = portfolio['strategy_returns'].std() * np.sqrt(252)
        
        # Calcular ratio de Sharpe
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Calcular máximo drawdown
        max_drawdown = portfolio['drawdown'].min()
        
        # Calcular número de operaciones
        trades = portfolio['position'].dropna()
        trades = trades[trades != 0]
        num_trades = len(trades)
        
        # Calcular porcentaje de operaciones ganadoras
        winning_trades = trades[trades > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate
        }
    except Exception as e:
        st.error(f"Error al calcular métricas: {e}")
        return {
            'total_return': 0,
            'annual_return': 0,
            'annual_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'num_trades': 0,
            'win_rate': 0
        }

# Función para visualizar resultados de backtesting - CORREGIDA
def plot_backtest_results(portfolio, signals):
    if portfolio is None or signals is None or portfolio.empty or signals.empty:
        st.error("No hay datos suficientes para visualizar resultados de backtesting.")
        return None
    
    try:
        # Crear figura con subplots
        fig = make_subplots(rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=('Precio y Señales', 'Valor del Portafolio', 'Drawdown'))
        
        # Añadir gráfico de precios
        fig.add_trace(
            go.Scatter(x=portfolio.index, y=signals['Close'], name='Precio', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Añadir señales de compra
        buy_signals = signals[signals['position'] > 0]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=buy_signals['Close'], name='Compra',
                        mode='markers', marker=dict(color='green', size=10, symbol='triangle-up')),
                row=1, col=1
            )
        
        # Añadir señales de venta
        sell_signals = signals[signals['position'] < 0]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=sell_signals['Close'], name='Venta',
                        mode='markers', marker=dict(color='red', size=10, symbol='triangle-down')),
                row=1, col=1
            )
        
        # Añadir gráfico de valor del portafolio
        fig.add_trace(
            go.Scatter(x=portfolio.index, y=portfolio['total'], name='Portafolio', line=dict(color='green')),
            row=2, col=1
        )
        
        # Añadir gráfico de drawdown
        fig.add_trace(
            go.Scatter(x=portfolio.index, y=portfolio['drawdown'], name='Drawdown', 
                    line=dict(color='red'), fill='tozeroy'),
            row=3, col=1
        )
        
        # Actualizar diseño
        fig.update_layout(
            height=800,
            title_text='Resultados de Backtesting',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error al visualizar resultados: {e}")
        return None

# Función para visualizar indicadores técnicos - CORREGIDA
def plot_technical_indicators(df):
    if df is None or df.empty:
        st.error("No hay datos suficientes para visualizar indicadores técnicos.")
        return None
    
    try:
        # Crear figura con subplots
        fig = make_subplots(rows=4, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.4, 0.2, 0.2, 0.2],
                            subplot_titles=('Precio y Medias Móviles', 'RSI', 'MACD', 'Volumen'))
        
        # Añadir gráfico de precios y medias móviles
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Precio', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA200'], name='SMA 200', line=dict(color='red')),
            row=1, col=1
        )
        
        # Añadir bandas de Bollinger
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_upper'], name='BB Superior', 
                    line=dict(color='rgba(173, 216, 230, 0.7)', dash='dash')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_lower'], name='BB Inferior', 
                    line=dict(color='rgba(173, 216, 230, 0.7)', dash='dash'),
                    fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)'),
            row=1, col=1
        )
        
        # Añadir RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Añadir líneas de referencia para RSI
        fig.add_shape(
            type="line", line_color="red", line_dash="dash",
            x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
            row=2, col=1
        )
        
        fig.add_shape(
            type="line", line_color="green", line_dash="dash",
            x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
            row=2, col=1
        )
        
        # Añadir MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_signal'], name='Señal MACD', line=dict(color='red')),
            row=3, col=1
        )
        
        # Añadir histograma MACD
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_hist'], name='Histograma MACD', marker_color=colors),
            row=3, col=1
        )
        
        # Añadir volumen
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volumen', marker_color='lightblue'),
            row=4, col=1
        )
        
        # Actualizar diseño
        fig.update_layout(
            height=1000,
            title_text='Análisis Técnico',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Actualizar rangos de y para RSI
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        
        return fig
    except Exception as e:
        st.error(f"Error al visualizar indicadores técnicos: {e}")
        return None

# Función para simular predicciones LSTM - CORREGIDA
def simulate_lstm_predictions(df, days=30):
    if df is None or df.empty:
        st.error("No hay datos suficientes para generar predicciones.")
        return None
    
    try:
        # Esta es una simulación simple para demostración
        # En un sistema real, usaríamos un modelo LSTM entrenado
        
        last_price = df['Close'].iloc[-1]
        dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
        
        # Simular predicciones con algo de ruido y tendencia
        trend = np.random.choice([-1, 1]) * 0.001  # Tendencia aleatoria
        predictions = [last_price]
        
        for i in range(days-1):
            next_price = predictions[-1] * (1 + trend + np.random.normal(0, 0.01))
            predictions.append(next_price)
        
        predictions_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Close': predictions
        })
        
        predictions_df.set_index('Date', inplace=True)
        
        return predictions_df
    except Exception as e:
        st.error(f"Error al generar predicciones: {e}")
        return None

# Función para visualizar predicciones - CORREGIDA
def plot_predictions(df, predictions_df):
    if df is None or predictions_df is None or df.empty or predictions_df.empty:
        st.error("No hay datos suficientes para visualizar predicciones.")
        return None
    
    try:
        # Crear figura
        fig = go.Figure()
        
        # Añadir datos históricos
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Histórico', line=dict(color='blue'))
        )
        
        # Añadir predicciones
        fig.add_trace(
            go.Scatter(x=predictions_df.index, y=predictions_df['Predicted_Close'], 
                    name='Predicción', line=dict(color='red', dash='dash'))
        )
        
        # Actualizar diseño
        fig.update_layout(
            title='Predicción de Precios',
            xaxis_title='Fecha',
            yaxis_title='Precio',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error al visualizar predicciones: {e}")
        return None

# Función para simular operaciones de trading - CORREGIDA
def simulate_trading(df, strategy='sma', initial_capital=10000):
    if df is None or df.empty:
        st.error("No hay datos suficientes para simular trading.")
        return None, None, None
    
    try:
        # Aplicar estrategia seleccionada
        if strategy == 'sma':
            signals = sma_strategy(df)
        elif strategy == 'bollinger':
            signals = bollinger_strategy(df)
        elif strategy == 'rsi':
            signals = rsi_strategy(df)
        elif strategy == 'macd':
            signals = macd_strategy(df)
        else:
            signals = sma_strategy(df)  # Por defecto
        
        if signals is None:
            return None, None, None
        
        # Realizar backtesting
        portfolio = backtest_strategy(signals, initial_capital=initial_capital)
        
        if portfolio is None:
            return signals, None, None
        
        # Calcular métricas
        metrics = calculate_performance_metrics(portfolio)
        
        return signals, portfolio, metrics
    except Exception as e:
        st.error(f"Error al simular trading: {e}")
        return None, None, None

# Función para mostrar métricas en tarjetas - CORREGIDA
def display_metrics_cards(metrics):
    if metrics is None:
        st.error("No hay métricas disponibles para mostrar.")
        return
    
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="card metric-card">
                <h3>Retorno Total</h3>
                <h2>{metrics['total_return']*100:.2f}%</h2>
                <p class="info-text">Retorno anualizado: {metrics['annual_return']*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card metric-card">
                <h3>Ratio de Sharpe</h3>
                <h2>{metrics['sharpe_ratio']:.2f}</h2>
                <p class="info-text">Volatilidad anual: {metrics['annual_volatility']*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card {'success-card' if metrics['max_drawdown'] > -0.2 else 'warning-card'}">
                <h3>Máximo Drawdown</h3>
                <h2>{metrics['max_drawdown']*100:.2f}%</h2>
                <p class="info-text">Menor es mejor</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card metric-card">
                <h3>Número de Operaciones</h3>
                <h2>{metrics['num_trades']}</h2>
                <p class="info-text">Frecuencia de trading</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card {'success-card' if metrics['win_rate'] > 0.5 else 'warning-card'}">
                <h3>Tasa de Éxito</h3>
                <h2>{metrics['win_rate']*100:.2f}%</h2>
                <p class="info-text">Porcentaje de operaciones ganadoras</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al mostrar métricas: {e}")

# Sidebar para navegación
st.sidebar.title("Sistema de Trading ML")
page = st.sidebar.selectbox(
    "Navegación",
    ["Inicio", "Análisis de Mercado", "Backtesting", "Predicciones", "Trading en Vivo"]
)

# Página de inicio
if page == "Inicio":
    st.markdown('<h1 class="main-header">Sistema de Trading con Machine Learning</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Bienvenido al Sistema de Trading con Machine Learning. Esta plataforma te permite analizar mercados financieros, 
        realizar backtesting de estrategias, generar predicciones con modelos de machine learning y ejecutar operaciones 
        de trading automáticas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Información general
    st.markdown('<h2 class="sub-header">Características Principales</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Análisis de Mercado</h3>
            <ul>
                <li>Visualización de datos históricos</li>
                <li>Indicadores técnicos avanzados</li>
                <li>Análisis de patrones de precio</li>
                <li>Detección de tendencias</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Backtesting</h3>
            <ul>
                <li>Evaluación de estrategias</li>
                <li>Métricas de rendimiento</li>
                <li>Optimización de parámetros</li>
                <li>Comparación de estrategias</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Predicciones con ML</h3>
            <ul>
                <li>Modelos LSTM para predicción de precios</li>
                <li>Aprendizaje por refuerzo para decisiones</li>
                <li>Evaluación de precisión</li>
                <li>Visualización de predicciones</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Trading Automático</h3>
            <ul>
                <li>Integración con BingX</li>
                <li>Ejecución automática de órdenes</li>
                <li>Gestión de riesgos</li>
                <li>Seguimiento de rendimiento</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Mercados disponibles
    st.markdown('<h2 class="sub-header">Mercados Disponibles</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Criptomonedas</h3>
            <p>BTC, ETH, BNB, XRP, ADA, SOL, DOT, DOGE, AVAX, MATIC</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Forex</h3>
            <p>EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>Índices</h3>
            <p>S&P 500, NASDAQ, Dow Jones, FTSE 100, DAX, Nikkei 225</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cómo empezar
    st.markdown('<h2 class="sub-header">Cómo Empezar</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <ol>
            <li>Explora el <span class="highlight">Análisis de Mercado</span> para visualizar datos históricos e indicadores técnicos</li>
            <li>Utiliza el <span class="highlight">Backtesting</span> para evaluar diferentes estrategias de trading</li>
            <li>Genera <span class="highlight">Predicciones</span> utilizando modelos de machine learning</li>
            <li>Configura el <span class="highlight">Trading en Vivo</span> para operar automáticamente</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Página de análisis de mercado
elif page == "Análisis de Mercado":
    st.markdown('<h1 class="main-header">Análisis de Mercado</h1>', unsafe_allow_html=True)
    
    # Selección de mercado y símbolo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market = st.selectbox("Mercado", ["Criptomonedas", "Forex", "Índices"])
    
    with col2:
        if market == "Criptomonedas":
            symbol = st.selectbox("Símbolo", ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"])
        elif market == "Forex":
            symbol = st.selectbox("Símbolo", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"])
        else:
            symbol = st.selectbox("Símbolo", ["^GSPC", "^IXIC", "^DJI", "^FTSE", "^GDAXI", "^N225"])
    
    with col3:
        timeframe = st.selectbox("Período", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
    
    # Cargar datos
    data = load_data(symbol, period=timeframe)
    
    if data is not None and not data.empty:
        # Mostrar información básica
        st.markdown('<h2 class="sub-header">Información General</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Último Precio", f"{data['Close'].iloc[-1]:.2f}", 
                     f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100:.2f}%")
        
        with col2:
            st.metric("Volumen (24h)", f"{data['Volume'].iloc[-1]:,.0f}")
        
        with col3:
            st.metric("Máximo (Período)", f"{data['High'].max():.2f}")
        
        with col4:
            st.metric("Mínimo (Período)", f"{data['Low'].min():.2f}")
        
        # Añadir indicadores técnicos
        data_with_indicators = add_indicators(data)
        
        # Visualizar datos
        st.markdown('<h2 class="sub-header">Gráfico de Precios</h2>', unsafe_allow_html=True)
        
        # Opciones de visualización
        chart_type = st.radio("Tipo de Gráfico", ["Velas", "Línea"], horizontal=True)
        
        if chart_type == "Velas":
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Precio'
            )])
            
            fig.update_layout(
                title=f'{symbol} - Gráfico de Velas',
                xaxis_title='Fecha',
                yaxis_title='Precio',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Precio de Cierre'
            ))
            
            fig.update_layout(
                title=f'{symbol} - Gráfico de Línea',
                xaxis_title='Fecha',
                yaxis_title='Precio',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar indicadores técnicos
        if data_with_indicators is not None:
            show_indicators = st.checkbox("Mostrar Indicadores Técnicos", value=True)
            
            if show_indicators:
                fig = plot_technical_indicators(data_with_indicators)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar datos en tabla
            show_data = st.checkbox("Mostrar Datos en Tabla")
            
            if show_data:
                st.dataframe(data_with_indicators)
    else:
        st.warning("Por favor, selecciona otro símbolo o período para visualizar datos.")

# Página de backtesting
elif page == "Backtesting":
    st.markdown('<h1 class="main-header">Backtesting de Estrategias</h1>', unsafe_allow_html=True)
    
    # Selección de mercado y símbolo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market = st.selectbox("Mercado", ["Criptomonedas", "Forex", "Índices"], key="bt_market")
    
    with col2:
        if market == "Criptomonedas":
            symbol = st.selectbox("Símbolo", ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"], key="bt_symbol")
        elif market == "Forex":
            symbol = st.selectbox("Símbolo", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"], key="bt_symbol")
        else:
            symbol = st.selectbox("Símbolo", ["^GSPC", "^IXIC", "^DJI", "^FTSE", "^GDAXI", "^N225"], key="bt_symbol")
    
    with col3:
        timeframe = st.selectbox("Período", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], key="bt_timeframe")
    
    # Cargar datos
    data = load_data(symbol, period=timeframe)
    
    if data is not None and not data.empty:
        # Añadir indicadores técnicos
        data_with_indicators = add_indicators(data)
        
        if data_with_indicators is not None:
            # Selección de estrategia
            st.markdown('<h2 class="sub-header">Selección de Estrategia</h2>', unsafe_allow_html=True)
            
            strategy = st.selectbox(
                "Estrategia",
                ["Cruce de Medias Móviles (SMA)", "Bandas de Bollinger", "RSI", "MACD"]
            )
            
            # Mapear selección a código de estrategia
            strategy_map = {
                "Cruce de Medias Móviles (SMA)": "sma",
                "Bandas de Bollinger": "bollinger",
                "RSI": "rsi",
                "MACD": "macd"
            }
            
            # Parámetros de backtesting
            st.markdown('<h2 class="sub-header">Parámetros de Backtesting</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                initial_capital = st.number_input("Capital Inicial ($)", min_value=100, value=10000, step=1000)
            
            with col2:
                position_size = st.slider("Tamaño de Posición (%)", min_value=10, max_value=100, value=100, step=10) / 100
            
            with col3:
                commission = st.slider("Comisión (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.1) / 100
            
            # Ejecutar backtesting
            if st.button("Ejecutar Backtesting"):
                with st.spinner("Ejecutando backtesting..."):
                    # Simular trading con la estrategia seleccionada
                    signals, portfolio, metrics = simulate_trading(
                        data_with_indicators, 
                        strategy=strategy_map[strategy], 
                        initial_capital=initial_capital
                    )
                    
                    if signals is not None and portfolio is not None and metrics is not None:
                        # Mostrar métricas
                        st.markdown('<h2 class="sub-header">Resultados</h2>', unsafe_allow_html=True)
                        
                        display_metrics_cards(metrics)
                        
                        # Visualizar resultados
                        fig = plot_backtest_results(portfolio, signals)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar tabla de operaciones
                        st.markdown('<h2 class="sub-header">Operaciones</h2>', unsafe_allow_html=True)
                        
                        # Filtrar solo las filas con operaciones
                        trades = signals[signals['position'] != 0].copy()
                        if not trades.empty:
                            trades['type'] = trades['position'].apply(lambda x: 'Compra' if x > 0 else 'Venta')
                            trades['price'] = trades['Close']
                            
                            # Mostrar tabla de operaciones
                            st.dataframe(trades[['type', 'price']])
                        else:
                            st.info("No se generaron operaciones con esta estrategia.")
                    else:
                        st.error("No se pudieron generar resultados de backtesting. Intenta con otro símbolo o período.")
    else:
        st.warning("Por favor, selecciona otro símbolo o período para realizar backtesting.")

# Página de predicciones
elif page == "Predicciones":
    st.markdown('<h1 class="main-header">Predicciones con Machine Learning</h1>', unsafe_allow_html=True)
    
    # Selección de mercado y símbolo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market = st.selectbox("Mercado", ["Criptomonedas", "Forex", "Índices"], key="pred_market")
    
    with col2:
        if market == "Criptomonedas":
            symbol = st.selectbox("Símbolo", ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"], key="pred_symbol")
        elif market == "Forex":
            symbol = st.selectbox("Símbolo", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"], key="pred_symbol")
        else:
            symbol = st.selectbox("Símbolo", ["^GSPC", "^IXIC", "^DJI", "^FTSE", "^GDAXI", "^N225"], key="pred_symbol")
    
    with col3:
        timeframe = st.selectbox("Período Histórico", ["1mo", "3mo", "6mo", "1y"], key="pred_timeframe")
    
    # Cargar datos
    data = load_data(symbol, period=timeframe)
    
    if data is not None and not data.empty:
        # Añadir indicadores técnicos
        data_with_indicators = add_indicators(data)
        
        if data_with_indicators is not None:
            # Selección de modelo
            st.markdown('<h2 class="sub-header">Modelo de Predicción</h2>', unsafe_allow_html=True)
            
            model_type = st.selectbox("Tipo de Modelo", ["LSTM", "DQN (Aprendizaje por Refuerzo)"])
            
            # Parámetros de predicción
            st.markdown('<h2 class="sub-header">Parámetros de Predicción</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                prediction_days = st.slider("Días a Predecir", min_value=1, max_value=30, value=7)
            
            with col2:
                confidence_interval = st.slider("Intervalo de Confianza (%)", min_value=50, max_value=99, value=95, step=5)
            
            # Ejecutar predicción
            if st.button("Generar Predicción"):
                with st.spinner("Generando predicción..."):
                    # Simular predicciones LSTM
                    predictions_df = simulate_lstm_predictions(data, days=prediction_days)
                    
                    if predictions_df is not None:
                        # Mostrar predicciones
                        st.markdown('<h2 class="sub-header">Resultados de Predicción</h2>', unsafe_allow_html=True)
                        
                        # Mostrar último precio y predicción
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Último Precio", f"{data['Close'].iloc[-1]:.2f}")
                        
                        with col2:
                            predicted_price = predictions_df['Predicted_Close'].iloc[-1]
                            change = (predicted_price - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100
                            st.metric(f"Precio Predicho ({prediction_days} días)", f"{predicted_price:.2f}", f"{change:.2f}%")
                        
                        with col3:
                            direction = "Alcista" if change > 0 else "Bajista"
                            st.metric("Tendencia Predicha", direction)
                        
                        # Visualizar predicciones
                        fig = plot_predictions(data, predictions_df)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recomendación de trading
                        st.markdown('<h2 class="sub-header">Recomendación de Trading</h2>', unsafe_allow_html=True)
                        
                        if change > 3:
                            recommendation = "Comprar"
                            color = "green"
                        elif change < -3:
                            recommendation = "Vender"
                            color = "red"
                        else:
                            recommendation = "Mantener"
                            color = "orange"
                        
                        st.markdown(f"""
                        <div class="card" style="border-left: 5px solid {color};">
                            <h3>Recomendación: {recommendation}</h3>
                            <p>Basado en la predicción de precio para los próximos {prediction_days} días, 
                            la recomendación es <strong>{recommendation}</strong>.</p>
                            <p>Cambio esperado: {change:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mostrar tabla de predicciones
                        st.markdown('<h2 class="sub-header">Tabla de Predicciones</h2>', unsafe_allow_html=True)
                        st.dataframe(predictions_df)
                    else:
                        st.error("No se pudieron generar predicciones. Intenta con otro símbolo o período.")
    else:
        st.warning("Por favor, selecciona otro símbolo o período para generar predicciones.")

# Página de trading en vivo
elif page == "Trading en Vivo":
    st.markdown('<h1 class="main-header">Trading en Vivo</h1>', unsafe_allow_html=True)
    
    # Configuración de API
    st.markdown('<h2 class="sub-header">Configuración de API</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input("API Key de BingX", type="password")
    
    with col2:
        api_secret = st.text_input("API Secret de BingX", type="password")
    
    # Guardar configuración
    if st.button("Guardar Configuración de API"):
        if api_key and api_secret:
            st.success("Configuración de API guardada correctamente")
        else:
            st.error("Por favor, introduce API Key y API Secret")
    
    # Configuración de trading
    st.markdown('<h2 class="sub-header">Configuración de Trading</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        trading_pairs = st.multiselect(
            "Pares de Trading",
            ["BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "ADA-USDT", "SOL-USDT"],
            ["BTC-USDT"]
        )
        
        strategy = st.selectbox(
            "Estrategia",
            ["LSTM", "DQN", "SMA", "Bollinger", "RSI", "MACD"]
        )
        
        interval = st.selectbox(
            "Intervalo",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        )
    
    with col2:
        trade_amount = st.number_input("Cantidad por Operación (USDT)", min_value=10, value=100)
        
        max_trades = st.number_input("Máximo de Operaciones por Día", min_value=1, value=5)
        
        risk_level = st.select_slider(
            "Nivel de Riesgo",
            options=["Bajo", "Medio", "Alto"]
        )
    
    # Configuración de gestión de riesgos
    st.markdown('<h2 class="sub-header">Gestión de Riesgos</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        stop_loss = st.slider("Stop Loss (%)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    
    with col2:
        take_profit = st.slider("Take Profit (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    
    # Iniciar/Detener trading
    st.markdown('<h2 class="sub-header">Control de Trading</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Iniciar Trading Automático"):
            if not api_key or not api_secret:
                st.error("Por favor, configura las credenciales de API primero")
            elif not trading_pairs:
                st.error("Por favor, selecciona al menos un par de trading")
            else:
                st.success("Trading automático iniciado correctamente")
    
    with col2:
        if st.button("Detener Trading"):
            st.info("Trading automático detenido")
    
    # Estado del trading
    st.markdown('<h2 class="sub-header">Estado del Trading</h2>', unsafe_allow_html=True)
    
    # Simular algunas posiciones abiertas
    positions = [
        {"symbol": "BTC-USDT", "side": "LONG", "amount": 0.01, "entry_price": 50000, "current_price": 51000, "pnl": 10},
        {"symbol": "ETH-USDT", "side": "SHORT", "amount": 0.1, "entry_price": 3000, "current_price": 2950, "pnl": 5}
    ]
    
    # Mostrar posiciones abiertas
    if positions:
        st.markdown('<h3>Posiciones Abiertas</h3>', unsafe_allow_html=True)
        
        for pos in positions:
            pnl_pct = pos["pnl"] / (pos["amount"] * pos["entry_price"]) * 100
            color = "green" if pos["pnl"] >= 0 else "red"
            
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid {color};">
                <h3>{pos["symbol"]} - {pos["side"]}</h3>
                <p>Cantidad: {pos["amount"]}</p>
                <p>Precio de entrada: {pos["entry_price"]}</p>
                <p>Precio actual: {pos["current_price"]}</p>
                <p>PnL: <span style="color: {color};">{pos["pnl"]} USDT ({pnl_pct:.2f}%)</span></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No hay posiciones abiertas actualmente")
    
    # Historial de operaciones
    st.markdown('<h3>Historial de Operaciones</h3>', unsafe_allow_html=True)
    
    # Simular historial de operaciones
    trade_history = [
        {"symbol": "BTC-USDT", "side": "BUY", "amount": 0.01, "price": 49000, "time": "2023-04-25 10:30:00", "pnl": 20},
        {"symbol": "BTC-USDT", "side": "SELL", "amount": 0.01, "price": 51000, "time": "2023-04-25 14:45:00", "pnl": 20},
        {"symbol": "ETH-USDT", "side": "BUY", "amount": 0.15, "price": 2800, "time": "2023-04-24 09:15:00", "pnl": -15},
        {"symbol": "ETH-USDT", "side": "SELL", "amount": 0.15, "price": 2700, "time": "2023-04-24 16:20:00", "pnl": -15}
    ]
    
    if trade_history:
        # Crear DataFrame
        history_df = pd.DataFrame(trade_history)
        
        # Mostrar tabla
        st.dataframe(history_df)
    else:
        st.info("No hay historial de operaciones disponible")
    
    # Resumen de rendimiento
    st.markdown('<h3>Resumen de Rendimiento</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Balance Total", "10,120 USDT", "+120 USDT")
    
    with col2:
        st.metric("Operaciones Totales", "4")
    
    with col3:
        st.metric("Tasa de Éxito", "50%")

# Ejecutar la aplicación
if __name__ == "__main__":
    pass

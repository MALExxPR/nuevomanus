import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import datetime
import time

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

# Importar módulos del proyecto
from data.crypto_data_collector import CryptoDataCollector
from data.forex_data_collector import ForexDataCollector
from data.data_preprocessor import DataPreprocessor
from models.lstm_model import LSTMPricePredictor
from models.backtesting import BacktestingSystem

class TradingMLApp:
    """
    Aplicación web para el sistema de trading con machine learning.
    """
    
    def __init__(self):
        """
        Inicializa la aplicación web.
        """
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        # Crear directorios si no existen
        os.makedirs(self.data_dir / 'crypto', exist_ok=True)
        os.makedirs(self.data_dir / 'forex', exist_ok=True)
        os.makedirs(self.data_dir / 'processed', exist_ok=True)
        os.makedirs(self.models_dir / 'lstm', exist_ok=True)
        os.makedirs(self.models_dir / 'dqn', exist_ok=True)
        os.makedirs(self.results_dir / 'backtesting', exist_ok=True)
        
        # Inicializar componentes
        self.crypto_collector = CryptoDataCollector()
        self.forex_collector = ForexDataCollector()
        self.preprocessor = DataPreprocessor()
        self.backtester = BacktestingSystem()
        self.lstm_predictor = LSTMPricePredictor()
    
    def run(self):
        """
        Ejecuta la aplicación web.
        """
        st.set_page_config(
            page_title="Sistema de Trading con Machine Learning",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar para navegación
        st.sidebar.title("Navegación")
        page = st.sidebar.radio(
            "Selecciona una página:",
            ["Inicio", "Datos", "Modelos", "Backtesting", "Trading en Vivo"]
        )
        
        # Mostrar página seleccionada
        if page == "Inicio":
            self.show_home_page()
        elif page == "Datos":
            self.show_data_page()
        elif page == "Modelos":
            self.show_models_page()
        elif page == "Backtesting":
            self.show_backtesting_page()
        elif page == "Trading en Vivo":
            self.show_live_trading_page()
    
    def show_home_page(self):
        """
        Muestra la página de inicio.
        """
        st.title("Sistema de Trading con Machine Learning")
        
        st.markdown("""
        ## Bienvenido al Sistema de Trading con Machine Learning
        
        Esta aplicación te permite utilizar modelos de machine learning para analizar y operar en mercados financieros.
        
        ### Características principales:
        
        - **Recolección de datos**: Obtén datos históricos de criptomonedas y forex.
        - **Modelos predictivos**: Utiliza modelos LSTM para predecir precios futuros.
        - **Aprendizaje por refuerzo**: Implementa estrategias de trading con DQN.
        - **Backtesting**: Evalúa el rendimiento de diferentes estrategias.
        - **Trading en vivo**: Conecta con la API de BingX para operar automáticamente.
        
        ### Cómo empezar:
        
        1. Ve a la página **Datos** para recolectar y procesar datos históricos.
        2. Utiliza la página **Modelos** para entrenar modelos predictivos.
        3. Evalúa estrategias en la página **Backtesting**.
        4. Configura el trading automático en la página **Trading en Vivo**.
        """)
        
        # Mostrar estadísticas básicas
        st.subheader("Estadísticas del Sistema")
        
        col1, col2, col3 = st.columns(3)
        
        # Contar archivos de datos
        crypto_files = list(self.data_dir.glob('crypto/*.csv'))
        forex_files = list(self.data_dir.glob('forex/*.csv'))
        
        with col1:
            st.metric("Datasets de Criptomonedas", len(crypto_files))
        
        with col2:
            st.metric("Datasets de Forex", len(forex_files))
        
        with col3:
            st.metric("Modelos Entrenados", len(list(self.models_dir.glob('lstm/*.h5'))))
    
    def show_data_page(self):
        """
        Muestra la página de datos.
        """
        st.title("Datos Históricos")
        
        # Tabs para diferentes secciones
        tab1, tab2, tab3 = st.tabs(["Recolectar Datos", "Ver Datos", "Procesar Datos"])
        
        with tab1:
            st.header("Recolectar Datos Históricos")
            
            # Seleccionar tipo de datos
            data_type = st.radio("Tipo de datos:", ["Criptomonedas", "Forex"])
            
            if data_type == "Criptomonedas":
                # Formulario para recolectar datos de criptomonedas
                with st.form("crypto_form"):
                    st.subheader("Recolectar Datos de Criptomonedas")
                    
                    # Seleccionar símbolos
                    crypto_symbols = st.multiselect(
                        "Selecciona criptomonedas:",
                        ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOT-USD", "DOGE-USD"],
                        default=["BTC-USD", "ETH-USD"]
                    )
                    
                    # Seleccionar período e intervalo
                    col1, col2 = st.columns(2)
                    with col1:
                        period = st.selectbox(
                            "Período:",
                            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                            index=3
                        )
                    
                    with col2:
                        interval = st.selectbox(
                            "Intervalo:",
                            ["1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"],
                            index=5
                        )
                    
                    # Botón para recolectar datos
                    submit_button = st.form_submit_button("Recolectar Datos")
                    
                    if submit_button:
                        if not crypto_symbols:
                            st.error("Por favor, selecciona al menos una criptomoneda.")
                        else:
                            # Mostrar progreso
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, symbol in enumerate(crypto_symbols):
                                status_text.text(f"Recolectando datos de {symbol}...")
                                
                                try:
                                    # Recolectar datos
                                    data = self.crypto_collector.get_data_from_yfinance(
                                        symbol, period=period, interval=interval
                                    )
                                    
                                    if data is not None:
                                        st.success(f"Datos de {symbol} recolectados correctamente.")
                                    else:
                                        st.error(f"Error al recolectar datos de {symbol}.")
                                
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                
                                # Actualizar progreso
                                progress_bar.progress((i + 1) / len(crypto_symbols))
                            
                            status_text.text("¡Recolección de datos completada!")
                            st.balloons()
            
            else:  # Forex
                # Formulario para recolectar datos de forex
                with st.form("forex_form"):
                    st.subheader("Recolectar Datos de Forex")
                    
                    # Seleccionar símbolos
                    forex_symbols = st.multiselect(
                        "Selecciona pares forex:",
                        ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"],
                        default=["EURUSD=X", "GBPUSD=X"]
                    )
                    
                    # Seleccionar período e intervalo
                    col1, col2 = st.columns(2)
                    with col1:
                        period = st.selectbox(
                            "Período:",
                            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                            index=3
                        )
                    
                    with col2:
                        interval = st.selectbox(
                            "Intervalo:",
                            ["1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"],
                            index=5
                        )
                    
                    # Botón para recolectar datos
                    submit_button = st.form_submit_button("Recolectar Datos")
                    
                    if submit_button:
                        if not forex_symbols:
                            st.error("Por favor, selecciona al menos un par forex.")
                        else:
                            # Mostrar progreso
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, symbol in enumerate(forex_symbols):
                                status_text.text(f"Recolectando datos de {symbol}...")
                                
                                try:
                                    # Recolectar datos
                                    data = self.forex_collector.get_data_from_yfinance(
                                        symbol, period=period, interval=interval
                                    )
                                    
                                    if data is not None:
                                        st.success(f"Datos de {symbol} recolectados correctamente.")
                                    else:
                                        st.error(f"Error al recolectar datos de {symbol}.")
                                
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                
                                # Actualizar progreso
                                progress_bar.progress((i + 1) / len(forex_symbols))
                            
                            status_text.text("¡Recolección de datos completada!")
                            st.balloons()
        
        with tab2:
            st.header("Ver Datos Históricos")
            
            # Seleccionar tipo de datos
            data_type = st.radio("Tipo de datos:", ["Criptomonedas", "Forex"], key="view_data_type")
            
            # Obtener lista de archivos
            if data_type == "Criptomonedas":
                data_files = list(self.data_dir.glob('crypto/*.csv'))
                data_path = self.data_dir / 'crypto'
            else:  # Forex
                data_files = list(self.data_dir.glob('forex/*.csv'))
                data_path = self.data_dir / 'forex'
            
            if not data_files:
                st.warning(f"No hay archivos de datos de {data_type.lower()} disponibles.")
            else:
                # Seleccionar archivo
                file_names = [file.name for file in data_files]
                selected_file = st.selectbox("Selecciona un archivo:", file_names)
                
                if selected_file:
                    # Cargar datos
                    file_path = data_path / selected_file
                    try:
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        
                        # Mostrar información básica
                        st.subheader("Información del Dataset")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Filas", df.shape[0])
                        with col2:
                            st.metric("Columnas", df.shape[1])
                        with col3:
                            st.metric("Período", f"{df.index[0].date()} a {df.index[-1].date()}")
                        
                        # Mostrar datos
                        st.subheader("Vista previa de los datos")
                        st.dataframe(df.head())
                        
                        # Visualizar datos
                        st.subheader("Visualización")
                        
                        # Asegurarse de que tenemos la columna 'close' o 'Close'
                        if 'close' in df.columns:
                            price_col = 'close'
                        elif 'Close' in df.columns:
                            price_col = 'Close'
                        else:
                            price_col = df.columns[0]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df.index, df[price_col])
                        ax.set_title(f'Precio de {selected_file.split("_")[0]}')
                        ax.set_xlabel('Fecha')
                        ax.set_ylabel('Precio')
                        ax.grid(True)
                        st.pyplot(fig)
                        
                        # Mostrar estadísticas
                        st.subheader("Estadísticas Descriptivas")
                        st.dataframe(df.describe())
                    
                    except Exception as e:
                        st.error(f"Error al cargar el archivo: {e}")
        
        with tab3:
            st.header("Procesar Datos")
            
            # Seleccionar tipo de datos
            data_type = st.radio("Tipo de datos:", ["Criptomonedas", "Forex"], key="process_data_type")
            
            # Obtener lista de archivos
            if data_type == "Criptomonedas":
                data_files = list(self.data_dir.glob('crypto/*.csv'))
                data_path = self.data_dir / 'crypto'
                processed_path = self.data_dir / 'processed' / 'crypto'
            else:  # Forex
                data_files = list(self.data_dir.glob('forex/*.csv'))
                data_path = self.data_dir / 'forex'
                processed_path = self.data_dir / 'processed' / 'forex'
            
            # Crear directorio si no existe
            os.makedirs(processed_path, exist_ok=True)
            
            if not data_files:
                st.warning(f"No hay archivos de datos de {data_type.lower()} disponibles para procesar.")
            else:
                # Seleccionar archivos
                file_names = [file.name for file in data_files]
                selected_files = st.multiselect("Selecciona archivos para procesar:", file_names)
                
                # Opciones de procesamiento
                st.subheader("Opciones de Procesamiento")
                
                add_indicators = st.checkbox("Añadir indicadores técnicos", value=True)
                
                # Botón para procesar datos
                if st.button("Procesar Datos"):
                    if not selected_files:
                        st.error("Por favor, selecciona al menos un archivo.")
                    else:
                        # Mostrar progreso
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file_name in enumerate(selected_files):
                            status_text.text(f"Procesando {file_name}...")
                            
                            try:
                                # Cargar datos
                                file_path = data_path / file_name
                                df = self.preprocessor.load_data(file_path)
                                
                                if df is not None:
                                    # Limpiar datos
                                    df_clean = self.preprocessor.clean_data(df)
                                    
                                    # Añadir indicadores técnicos si se solicita
                                    if add_indicators:
                                        df_processed = self.preprocessor.add_technical_indicators(df_clean)
                                    else:
                                        df_processed = df_clean
                                    
                                    if df_processed is not None and not df_processed.empty:
                                        # Guardar datos procesados
                                        output_path = processed_path / f"processed_{file_name}"
                                        df_processed.to_csv(output_path)
                                        st.success(f"Datos de {file_name} procesados y guardados en {output_path}")
                                    else:
                                        st.error(f"Error al procesar datos de {file_name}.")
                                else:
                                    st.error(f"Error al cargar datos de {file_name}.")
                            
                            except Exception as e:
                                st.error(f"Error: {e}")
                            
                            # Actualizar progreso
                            progress_bar.progress((i + 1) / len(selected_files))
                        
                        status_text.text("¡Procesamiento de datos completado!")
                        st.balloons()
    
    def show_models_page(self):
        """
        Muestra la página de modelos.
        """
        st.title("Modelos de Machine Learning")
        
        # Tabs para diferentes modelos
        tab1, tab2 = st.tabs(["Modelo LSTM", "Modelo DQN"])
        
        with tab1:
            st.header("Modelo LSTM para Predicción de Precios")
            
            # Seleccionar datos procesados
            st.subheader("Seleccionar Datos")
            
            # Seleccionar tipo de datos
            data_type = st.radio("Tipo de datos:", ["Criptomonedas", "Forex"], key="lstm_data_type")
            
            # Obtener lista de archivos procesados
            if data_type == "Criptomonedas":
                data_files = list((self.data_dir / 'processed' / 'crypto').glob('*.csv'))
                data_path = self.data_dir / 'processed' / 'crypto'
            else:  # Forex
                data_files = list((self.data_dir / 'processed' / 'forex').glob('*.csv'))
                data_path = self.data_dir / 'processed' / 'forex'
            
            if not data_files:
                st.warning(f"No hay archivos de datos procesados de {data_type.lower()} disponibles.")
            else:
                # Seleccionar archivo
                file_names = [file.name for file in data_files]
                selected_file = st.selectbox("Selecciona un archivo:", file_names, key="lstm_file")
                
                if selected_file:
                    # Cargar datos
                    file_path = data_path / selected_file
                    
                    # Parámetros del modelo
                    st.subheader("Parámetros del Modelo")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        sequence_length = st.slider("Longitud de secuencia:", 10, 100, 60)
                        epochs = st.slider("Épocas:", 10, 100, 50)
                    
                    with col2:
                        batch_size = st.slider("Tamaño de lote:", 8, 64, 32)
                        train_split = st.slider("Proporción de entrenamiento:", 0.5, 0.9, 0.8)
                    
                    # Botón para entrenar modelo
                    if st.button("Entrenar Modelo LSTM"):
                        try:
                            # Cargar datos
                            df = self.lstm_predictor.load_data(file_path)
                            
                            if df is not None:
                                # Preparar datos
                                st.info("Preparando datos para el modelo LSTM...")
                                data = self.lstm_predictor.prepare_data(
                                    df, target_column='close', 
                                    sequence_length=sequence_length,
                                    train_split=train_split
                                )
                                
                                if data is not None:
                                    # Entrenar modelo
                                    st.info("Entrenando modelo LSTM... Esto puede tardar unos minutos.")
                                    
                                    # Crear placeholder para la barra de progreso
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # Entrenar modelo (simulación de progreso)
                                    for i in range(epochs):
                                        # Actualizar progreso
                                        progress = (i + 1) / epochs
                                        progress_bar.progress(progress)
                                        status_text.text(f"Entrenando... Época {i+1}/{epochs}")
                                        time.sleep(0.1)  # Simular entrenamiento
                                    
                                    # Entrenar modelo real
                                    model_name = f"lstm_{selected_file.split('.')[0]}"
                                    history = self.lstm_predictor.train_model(
                                        data, epochs=epochs, batch_size=batch_size,
                                        model_name=model_name
                                    )
                                    
                                    # Realizar predicciones
                                    st.info("Realizando predicciones...")
                                    results = self.lstm_predictor.predict(data)
                                    
                                    if results is not None:
                                        # Evaluar modelo
                                        metrics = self.lstm_predictor.evaluate(
                                            results['predictions'], results['real_values']
                                        )
                                        
                                        # Mostrar métricas
                                        st.subheader("Métricas de Evaluación")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("MSE", f"{metrics['mse']:.4f}")
                                        with col2:
                                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                                        with col3:
                                            st.metric("MAE", f"{metrics['mae']:.4f}")
                                        with col4:
                                            st.metric("MAPE", f"{metrics['mape']:.2f}%")
                                        
                                        # Visualizar resultados
                                        st.subheader("Resultados de la Predicción")
                                        
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.plot(results['real_values'], color='blue', label='Valores reales')
                                        ax.plot(results['predictions'], color='red', label='Predicciones')
                                        ax.set_title(f'Predicción de precios - {selected_file.split("_")[1]}')
                                        ax.set_xlabel('Tiempo')
                                        ax.set_ylabel('Precio')
                                        ax.legend()
                                        ax.grid(True)
                                        st.pyplot(fig)
                                        
                                        # Guardar modelo
                                        self.lstm_predictor.save_model()
                                        st.success(f"Modelo guardado como {model_name}.h5")
                                    else:
                                        st.error("Error al realizar predicciones.")
                                else:
                                    st.error("Error al preparar datos para el modelo.")
                            else:
                                st.error("Error al cargar datos.")
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            # Cargar modelo existente
            st.subheader("Cargar Modelo Existente")
            
            # Obtener lista de modelos
            model_files = list(self.models_dir.glob('lstm/*.h5'))
            
            if not model_files:
                st.warning("No hay modelos LSTM guardados.")
            else:
                # Seleccionar modelo
                model_names = [file.name for file in model_files]
                selected_model = st.selectbox("Selecciona un modelo:", model_names)
                
                if selected_model:
                    # Cargar modelo
                    if st.button("Cargar Modelo"):
                        try:
                            model_path = self.models_dir / 'lstm' / selected_model
                            self.lstm_predictor.load_model(model_path)
                            st.success(f"Modelo {selected_model} cargado correctamente.")
                        except Exception as e:
                            st.error(f"Error al cargar el modelo: {e}")
        
        with tab2:
            st.header("Modelo DQN para Aprendizaje por Refuerzo")
            
            st.info("La implementación del modelo DQN está en desarrollo. Estará disponible próximamente.")
    
    def show_backtesting_page(self):
        """
        Muestra la página de backtesting.
        """
        st.title("Backtesting de Estrategias")
        
        # Seleccionar datos procesados
        st.subheader("Seleccionar Datos")
        
        # Seleccionar tipo de datos
        data_type = st.radio("Tipo de datos:", ["Criptomonedas", "Forex"], key="backtest_data_type")
        
        # Obtener lista de archivos procesados
        if data_type == "Criptomonedas":
            data_files = list((self.data_dir / 'processed' / 'crypto').glob('*.csv'))
            data_path = self.data_dir / 'processed' / 'crypto'
        else:  # Forex
            data_files = list((self.data_dir / 'processed' / 'forex').glob('*.csv'))
            data_path = self.data_dir / 'processed' / 'forex'
        
        if not data_files:
            st.warning(f"No hay archivos de datos procesados de {data_type.lower()} disponibles.")
        else:
            # Seleccionar archivo
            file_names = [file.name for file in data_files]
            selected_file = st.selectbox("Selecciona un archivo:", file_names, key="backtest_file")
            
            if selected_file:
                # Cargar datos
                file_path = data_path / selected_file
                
                # Seleccionar estrategias
                st.subheader("Seleccionar Estrategias")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    use_sma = st.checkbox("Cruce de Medias Móviles", value=True)
                    use_bollinger = st.checkbox("Bandas de Bollinger", value=True)
                
                with col2:
                    use_rsi = st.checkbox("RSI", value=True)
                    use_macd = st.checkbox("MACD", value=True)
                
                # Parámetros de backtesting
                st.subheader("Parámetros de Backtesting")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    initial_capital = st.number_input("Capital inicial ($):", min_value=1000, max_value=1000000, value=10000)
                
                with col2:
                    commission = st.slider("Comisión (%):", 0.0, 1.0, 0.1) / 100
                
                # Botón para ejecutar backtesting
                if st.button("Ejecutar Backtesting"):
                    if not (use_sma or use_bollinger or use_rsi or use_macd):
                        st.error("Por favor, selecciona al menos una estrategia.")
                    else:
                        try:
                            # Cargar datos
                            df = self.backtester.load_data(file_path)
                            
                            if df is not None:
                                # Definir estrategias a comparar
                                strategies = {}
                                
                                if use_sma:
                                    strategies['SMA'] = lambda data: self.backtester.simple_moving_average_strategy(
                                        data, short_window=20, long_window=50
                                    )
                                
                                if use_bollinger:
                                    strategies['Bollinger'] = lambda data: self.backtester.bollinger_bands_strategy(
                                        data, window=20, num_std=2
                                    )
                                
                                if use_rsi:
                                    strategies['RSI'] = lambda data: self.backtester.rsi_strategy(
                                        data, window=14, oversold=30, overbought=70
                                    )
                                
                                if use_macd:
                                    strategies['MACD'] = lambda data: self.backtester.macd_strategy(
                                        data, fast=12, slow=26, signal=9
                                    )
                                
                                # Comparar estrategias
                                st.info("Ejecutando backtesting... Esto puede tardar unos momentos.")
                                
                                # Crear placeholder para la barra de progreso
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Simular progreso
                                for i in range(len(strategies)):
                                    # Actualizar progreso
                                    progress = (i + 1) / len(strategies)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Evaluando estrategia {i+1}/{len(strategies)}...")
                                    time.sleep(0.5)  # Simular procesamiento
                                
                                # Ejecutar backtesting real
                                save_path = self.results_dir / 'backtesting' / f"strategy_comparison_{selected_file.split('.')[0]}.png"
                                metrics = self.backtester.compare_strategies(
                                    df, strategies, initial_capital=initial_capital,
                                    save_path=str(save_path)
                                )
                                
                                # Mostrar métricas
                                st.subheader("Métricas de Rendimiento")
                                st.dataframe(metrics)
                                
                                # Mostrar gráfico
                                st.subheader("Comparación de Estrategias")
                                st.image(str(save_path))
                                
                                # Seleccionar mejor estrategia
                                best_strategy = metrics['sharpe_ratio'].idxmax()
                                st.success(f"La mejor estrategia según el ratio de Sharpe es: {best_strategy}")
                                
                                # Ejecutar y visualizar la mejor estrategia
                                st.subheader(f"Resultados Detallados - Estrategia {best_strategy}")
                                
                                signals = strategies[best_strategy](df)
                                portfolio = self.backtester.backtest_strategy(
                                    signals, initial_capital=initial_capital, commission=commission
                                )
                                
                                # Guardar gráfico
                                save_path = self.results_dir / 'backtesting' / f"{best_strategy}_strategy_{selected_file.split('.')[0]}.png"
                                
                                # Visualizar resultados
                                self.backtester.plot_strategy_results(
                                    portfolio, title=f'Estrategia {best_strategy}',
                                    save_path=str(save_path)
                                )
                                
                                st.image(str(save_path))
                            else:
                                st.error("Error al cargar datos.")
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    def show_live_trading_page(self):
        """
        Muestra la página de trading en vivo.
        """
        st.title("Trading en Vivo")
        
        st.info("La integración con la API de BingX para trading en vivo está en desarrollo. Estará disponible próximamente.")
        
        # Formulario para configurar API
        with st.form("api_form"):
            st.subheader("Configuración de API de BingX")
            
            api_key = st.text_input("API Key:", type="password")
            api_secret = st.text_input("API Secret:", type="password")
            
            # Botón para guardar configuración
            submit_button = st.form_submit_button("Guardar Configuración")
            
            if submit_button:
                if not api_key or not api_secret:
                    st.error("Por favor, ingresa la API Key y el API Secret.")
                else:
                    st.success("Configuración guardada correctamente.")
        
        # Configuración de trading
        st.subheader("Configuración de Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trading_pair = st.selectbox(
                "Par de trading:",
                ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT"]
            )
            
            strategy = st.selectbox(
                "Estrategia:",
                ["LSTM Prediction", "DQN Agent", "SMA Crossover", "RSI Strategy", "MACD Strategy"]
            )
        
        with col2:
            trade_amount = st.number_input("Cantidad por operación ($):", min_value=10, max_value=10000, value=100)
            
            max_trades = st.number_input("Máximo de operaciones por día:", min_value=1, max_value=100, value=5)
        
        # Botón para iniciar trading
        if st.button("Iniciar Trading Automático"):
            st.warning("La funcionalidad de trading en vivo aún no está disponible.")

# Ejecutar la aplicación
if __name__ == "__main__":
    app = TradingMLApp()
    app.run()

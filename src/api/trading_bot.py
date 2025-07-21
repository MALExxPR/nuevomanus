import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
import logging
import json
from datetime import datetime, timedelta

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

# Importar módulos del proyecto
from api.bingx_api import BingXAPI
from models.lstm_model import LSTMPricePredictor
from models.dqn_model import DQNTradingAgent, TradingEnvironment
from models.backtesting import BacktestingSystem
from data.data_preprocessor import DataPreprocessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent.parent / 'logs' / 'trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('trading_bot')

class TradingBot:
    """
    Bot de trading que utiliza modelos de machine learning para operar
    automáticamente a través de la API de BingX.
    """
    
    def __init__(self, config_file=None):
        """
        Inicializa el bot de trading.
        
        Args:
            config_file: Ruta al archivo de configuración.
        """
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        self.logs_dir = self.base_dir / 'logs'
        
        # Crear directorios si no existen
        os.makedirs(self.data_dir / 'crypto', exist_ok=True)
        os.makedirs(self.data_dir / 'forex', exist_ok=True)
        os.makedirs(self.data_dir / 'processed', exist_ok=True)
        os.makedirs(self.models_dir / 'lstm', exist_ok=True)
        os.makedirs(self.models_dir / 'dqn', exist_ok=True)
        os.makedirs(self.models_dir / 'ql', exist_ok=True)
        os.makedirs(self.results_dir / 'backtesting', exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Cargar configuración
        self.config = self._load_config(config_file)
        
        # Inicializar API de BingX
        self.api = BingXAPI(
            api_key=self.config.get('api_key'),
            api_secret=self.config.get('api_secret')
        )
        
        # Inicializar componentes
        self.preprocessor = DataPreprocessor()
        self.backtester = BacktestingSystem()
        
        # Inicializar modelos
        self.lstm_model = None
        self.dqn_agent = None
        self.ql_agent = None
        
        # Estado del bot
        self.is_running = False
        self.current_positions = {}
        self.trade_history = []
        
        logger.info("Bot de trading inicializado")
    
    def _load_config(self, config_file=None):
        """
        Carga la configuración del bot.
        
        Args:
            config_file: Ruta al archivo de configuración.
            
        Returns:
            Configuración cargada.
        """
        default_config = {
            'api_key': None,
            'api_secret': None,
            'trading_pairs': ['BTC-USDT', 'ETH-USDT'],
            'strategy': 'lstm',  # 'lstm', 'dqn', 'ql', 'sma', 'rsi', 'macd'
            'interval': '1h',
            'trade_amount': 100,  # USDT
            'max_trades_per_day': 5,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 3.0,
            'risk_level': 'medium',  # 'low', 'medium', 'high'
            'auto_trading': False
        }
        
        if config_file is None:
            config_file = self.base_dir / 'config' / 'trading_bot_config.json'
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Actualizar configuración por defecto con valores cargados
                    default_config.update(loaded_config)
                    logger.info(f"Configuración cargada desde {config_file}")
            except Exception as e:
                logger.error(f"Error al cargar el archivo de configuración: {e}")
        else:
            logger.warning(f"Archivo de configuración no encontrado: {config_file}")
            logger.info("Usando configuración por defecto")
            
            # Guardar configuración por defecto
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Configuración por defecto guardada en {config_file}")
        
        return default_config
    
    def save_config(self, config, config_file=None):
        """
        Guarda la configuración del bot.
        
        Args:
            config: Configuración a guardar.
            config_file: Ruta al archivo de configuración.
            
        Returns:
            True si se guardó correctamente, False en caso contrario.
        """
        if config_file is None:
            config_file = self.base_dir / 'config' / 'trading_bot_config.json'
        
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Actualizar configuración actual
            self.config.update(config)
            
            logger.info(f"Configuración guardada en {config_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error al guardar el archivo de configuración: {e}")
            return False
    
    def update_api_credentials(self, api_key, api_secret):
        """
        Actualiza las credenciales de la API de BingX.
        
        Args:
            api_key: API Key de BingX.
            api_secret: API Secret de BingX.
            
        Returns:
            True si se actualizaron correctamente, False en caso contrario.
        """
        try:
            # Actualizar configuración
            self.config['api_key'] = api_key
            self.config['api_secret'] = api_secret
            
            # Guardar configuración
            self.save_config(self.config)
            
            # Actualizar API
            self.api = BingXAPI(api_key=api_key, api_secret=api_secret)
            
            logger.info("Credenciales de API actualizadas")
            return True
        
        except Exception as e:
            logger.error(f"Error al actualizar credenciales de API: {e}")
            return False
    
    def load_lstm_model(self, model_path=None, symbol=None):
        """
        Carga un modelo LSTM para predicción de precios.
        
        Args:
            model_path: Ruta al modelo guardado.
            symbol: Símbolo para el que se cargará un modelo específico.
            
        Returns:
            True si se cargó correctamente, False en caso contrario.
        """
        try:
            self.lstm_model = LSTMPricePredictor()
            
            if model_path:
                model_file = model_path
            elif symbol:
                # Buscar modelo específico para el símbolo
                symbol_name = symbol.replace('-', '_').lower()
                model_file = self.models_dir / 'lstm' / f"lstm_{symbol_name}.h5"
                
                if not os.path.exists(model_file):
                    # Buscar cualquier modelo disponible
                    model_files = list(self.models_dir.glob('lstm/*.h5'))
                    if model_files:
                        model_file = model_files[0]
                    else:
                        logger.error("No se encontraron modelos LSTM disponibles")
                        return False
            else:
                # Buscar cualquier modelo disponible
                model_files = list(self.models_dir.glob('lstm/*.h5'))
                if model_files:
                    model_file = model_files[0]
                else:
                    logger.error("No se encontraron modelos LSTM disponibles")
                    return False
            
            self.lstm_model.load_model(model_file)
            logger.info(f"Modelo LSTM cargado desde {model_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error al cargar el modelo LSTM: {e}")
            return False
    
    def load_dqn_agent(self, model_path=None, symbol=None):
        """
        Carga un agente DQN para trading.
        
        Args:
            model_path: Ruta al modelo guardado.
            symbol: Símbolo para el que se cargará un modelo específico.
            
        Returns:
            True si se cargó correctamente, False en caso contrario.
        """
        try:
            # Crear un entorno temporal para obtener el tamaño del estado
            # Esto es necesario para inicializar el agente DQN
            temp_df = self._get_historical_data(symbol or self.config['trading_pairs'][0], '1d', limit=100)
            if temp_df is None:
                logger.error("No se pudieron obtener datos históricos para inicializar el agente DQN")
                return False
            
            temp_env = TradingEnvironment(temp_df)
            state = temp_env.reset()
            state_size = state.shape[1]
            
            # Inicializar agente DQN
            self.dqn_agent = DQNTradingAgent(state_size=state_size)
            
            if model_path:
                model_file = model_path
            elif symbol:
                # Buscar modelo específico para el símbolo
                symbol_name = symbol.replace('-', '_').lower()
                model_file = self.models_dir / 'dqn' / f"dqn_{symbol_name}.h5"
                
                if not os.path.exists(model_file):
                    # Buscar cualquier modelo disponible
                    model_files = list(self.models_dir.glob('dqn/*.h5'))
                    if model_files:
                        model_file = model_files[0]
                    else:
                        logger.warning("No se encontraron modelos DQN disponibles, se usará un agente sin entrenar")
                        return True
            else:
                # Buscar cualquier modelo disponible
                model_files = list(self.models_dir.glob('dqn/*.h5'))
                if model_files:
                    model_file = model_files[0]
                else:
                    logger.warning("No se encontraron modelos DQN disponibles, se usará un agente sin entrenar")
                    return True
            
            self.dqn_agent.load(model_file)
            logger.info(f"Agente DQN cargado desde {model_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error al cargar el agente DQN: {e}")
            return False

    def load_ql_agent(self, model_path=None):
        """Carga un agente de Q-learning."""
        try:
            from models.q_learning_agent import QLearningTradingAgent
            if model_path is None:
                model_path = self.models_dir / 'ql' / 'q_table.json'
            self.ql_agent = QLearningTradingAgent(model_path=model_path)
            self.ql_agent.load()
            logger.info(f"Agente Q-learning cargado desde {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar el agente Q-learning: {e}")
            return False
    
    def _get_historical_data(self, symbol, interval, limit=500):
        """
        Obtiene datos históricos para un símbolo.
        
        Args:
            symbol: Símbolo del par de trading.
            interval: Intervalo de tiempo.
            limit: Número máximo de velas a obtener.
            
        Returns:
            DataFrame con datos históricos.
        """
        try:
            # Obtener datos de la API
            klines = self.api.get_klines(symbol, interval, limit=limit)
            
            if isinstance(klines, pd.DataFrame):
                # Renombrar columnas para consistencia
                klines = klines.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Establecer índice
                klines.set_index('openTime', inplace=True)
                
                return klines
            else:
                logger.error(f"Error al obtener datos históricos: {klines}")
                return None
        
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocesa los datos para su uso en modelos de ML.
        
        Args:
            df: DataFrame con datos históricos.
            
        Returns:
            DataFrame con datos preprocesados.
        """
        try:
            # Limpiar datos
            df_clean = self.preprocessor.clean_data(df)
            
            # Añadir indicadores técnicos
            df_processed = self.preprocessor.add_technical_indicators(df_clean)
            
            return df_processed
        
        except Exception as e:
            logger.error(f"Error al preprocesar datos: {e}")
            return None
    
    def predict_with_lstm(self, df, target_column='close', sequence_length=60):
        """
        Realiza predicciones con el modelo LSTM.
        
        Args:
            df: DataFrame con datos históricos.
            target_column: Columna objetivo para la predicción.
            sequence_length: Longitud de la secuencia para el modelo LSTM.
            
        Returns:
            Predicción del precio.
        """
        if self.lstm_model is None:
            logger.error("No hay un modelo LSTM cargado")
            return None
        
        try:
            # Preparar datos
            data = self.lstm_model.prepare_data(df, target_column=target_column, sequence_length=sequence_length)
            
            if data is None:
                logger.error("Error al preparar datos para el modelo LSTM")
                return None
            
            # Realizar predicción
            results = self.lstm_model.predict(data)
            
            if results is None:
                logger.error("Error al realizar predicción con el modelo LSTM")
                return None
            
            # Obtener última predicción
            last_prediction = results['predictions'][-1]
            
            return last_prediction
        
        except Exception as e:
            logger.error(f"Error al predecir con LSTM: {e}")
            return None
    
    def get_action_with_dqn(self, df):
        """
        Obtiene la acción recomendada por el agente DQN.
        
        Args:
            df: DataFrame con datos históricos.
            
        Returns:
            Acción recomendada (0: mantener, 1: comprar, 2: vender).
        """
        if self.dqn_agent is None:
            logger.error("No hay un agente DQN cargado")
            return None

        try:
            env = TradingEnvironment(df)
            state = env.reset()
            action = self.dqn_agent.act(state, training=False)
            return action
        except Exception as e:
            logger.error(f"Error al obtener acción con DQN: {e}")
            return None

    def train_ql_agent(self, df):
        """Entrena el agente Q-learning con datos históricos."""
        if self.ql_agent is None:
            return
        try:
            for i in range(len(df) - 1):
                state = self.ql_agent.state_from_row(df.iloc[i])
                next_state = self.ql_agent.state_from_row(df.iloc[i + 1])
                price_diff = df['close'].iloc[i + 1] - df['close'].iloc[i]
                reward = price_diff
                action = self.ql_agent.choose_action(state)
                self.ql_agent.update(state, action, reward, next_state, i == len(df) - 2)
        except Exception as e:
            logger.error(f"Error al entrenar agente Q-learning: {e}")

    def get_action_with_ql(self, df, training=False):
        """Obtiene la acción recomendada por el agente Q-learning."""
        if self.ql_agent is None:
            logger.error("No hay un agente Q-learning cargado")
            return None
        try:
            last_row = df.iloc[-1]
            state = self.ql_agent.state_from_row(last_row)
            return self.ql_agent.choose_action(state, training=training)
        except Exception as e:
            logger.error(f"Error al obtener acción con Q-learning: {e}")
            return None
    
    def get_action_with_strategy(self, df, strategy='sma'):
        """
        Obtiene la acción recomendada por una estrategia técnica.
        
        Args:
            df: DataFrame con datos históricos.
            strategy: Estrategia a utilizar ('sma', 'bollinger', 'rsi', 'macd').
            
        Returns:
            Acción recomendada (0: mantener, 1: comprar, 2: vender).
        """
        try:
            if strategy == 'sma':
                signals = self.backtester.simple_moving_average_strategy(df)
            elif strategy == 'bollinger':
                signals = self.backtester.bollinger_bands_strategy(df)
            elif strategy == 'rsi':
                signals = self.backtester.rsi_strategy(df)
            elif strategy == 'macd':
                signals = self.backtester.macd_strategy(df)
            else:
                logger.error(f"Estrategia no reconocida: {strategy}")
                return None
            
            if signals is None:
                logger.error(f"Error al generar señales con la estrategia {strategy}")
                return None
            
            # Obtener última señal
            last_position = signals['position'].iloc[-1]
            
            # Convertir señal a acción
            if last_position > 0:  # Comprar
                return 1
            elif last_position < 0:  # Vender
                return 2
            else:  # Mantener
                return 0
        
        except Exception as e:
            logger.error(f"Error al obtener acción con estrategia {strategy}: {e}")
            return None
    
    def execute_trade(self, symbol, action, amount):
        """
        Ejecuta una operación de trading.
        
        Args:
            symbol: Símbolo del par de trading.
            action: Acción a ejecutar (0: mantener, 1: comprar, 2: vender).
            amount: Cantidad a operar en USDT.
            
        Returns:
            Resultado de la operación.
        """
        if not self.config.get('auto_trading', False):
            logger.info(f"Trading automático desactivado. Acción recomendada para {symbol}: {action}")
            return {
                'symbol': symbol,
                'action': action,
                'amount': amount,
                'status': 'simulated',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Obtener precio actual
            ticker = self.api.get_ticker(symbol)
            
            if 'code' in ticker and ticker['code'] != 0:
                logger.error(f"Error al obtener ticker para {symbol}: {ticker}")
                return None
            
            current_price = float(ticker['data'][0]['lastPrice'])
            
            # Calcular cantidad en unidades del activo
            quantity = amount / current_price
            
            # Ejecutar operación
            if action == 1:  # Comprar
                result = self.api.place_order(
                    symbol=symbol,
                    side='BUY',
                    order_type='MARKET',
                    quantity=quantity
                )
                
                if result.get('code') == 0:
                    logger.info(f"Orden de compra ejecutada para {symbol}: {quantity} unidades a {current_price}")
                    
                    # Registrar operación
                    trade = {
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': quantity,
                        'price': current_price,
                        'amount': amount,
                        'timestamp': datetime.now().isoformat(),
                        'order_id': result.get('data', {}).get('orderId')
                    }
                    
                    self.trade_history.append(trade)
                    
                    # Actualizar posiciones actuales
                    if symbol in self.current_positions:
                        self.current_positions[symbol]['quantity'] += quantity
                        self.current_positions[symbol]['avg_price'] = (
                            (self.current_positions[symbol]['avg_price'] * self.current_positions[symbol]['quantity'] - quantity) +
                            (current_price * quantity)
                        ) / self.current_positions[symbol]['quantity']
                    else:
                        self.current_positions[symbol] = {
                            'quantity': quantity,
                            'avg_price': current_price,
                            'entry_time': datetime.now().isoformat()
                        }
                    
                    return trade
                else:
                    logger.error(f"Error al ejecutar orden de compra para {symbol}: {result}")
                    return None
            
            elif action == 2:  # Vender
                # Verificar si tenemos posición abierta
                if symbol not in self.current_positions or self.current_positions[symbol]['quantity'] <= 0:
                    logger.warning(f"No hay posición abierta para {symbol}")
                    return None
                
                # Obtener cantidad a vender
                quantity = self.current_positions[symbol]['quantity']
                
                result = self.api.place_order(
                    symbol=symbol,
                    side='SELL',
                    order_type='MARKET',
                    quantity=quantity
                )
                
                if result.get('code') == 0:
                    logger.info(f"Orden de venta ejecutada para {symbol}: {quantity} unidades a {current_price}")
                    
                    # Registrar operación
                    trade = {
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': quantity,
                        'price': current_price,
                        'amount': quantity * current_price,
                        'timestamp': datetime.now().isoformat(),
                        'order_id': result.get('data', {}).get('orderId'),
                        'profit': (current_price - self.current_positions[symbol]['avg_price']) * quantity
                    }
                    
                    self.trade_history.append(trade)
                    
                    # Actualizar posiciones actuales
                    del self.current_positions[symbol]
                    
                    return trade
                else:
                    logger.error(f"Error al ejecutar orden de venta para {symbol}: {result}")
                    return None
            
            else:  # Mantener
                logger.info(f"Manteniendo posición para {symbol}")
                return {
                    'symbol': symbol,
                    'action': 'hold',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error al ejecutar operación para {symbol}: {e}")
            return None
    
    def run_trading_cycle(self):
        """
        Ejecuta un ciclo completo de trading para todos los pares configurados.
        
        Returns:
            Resultados del ciclo de trading.
        """
        results = []
        
        for symbol in self.config['trading_pairs']:
            try:
                logger.info(f"Iniciando ciclo de trading para {symbol}")
                
                # Obtener datos históricos
                df = self._get_historical_data(symbol, self.config['interval'], limit=100)
                
                if df is None:
                    logger.error(f"No se pudieron obtener datos históricos para {symbol}")
                    continue
                
                # Preprocesar datos
                df_processed = self.preprocess_data(df)
                
                if df_processed is None:
                    logger.error(f"Error al preprocesar datos para {symbol}")
                    continue
                
                # Obtener acción recomendada según la estrategia configurada
                action = None
                
                if self.config['strategy'] == 'lstm':
                    # Cargar modelo LSTM si no está cargado
                    if self.lstm_model is None:
                        self.load_lstm_model(symbol=symbol)
                    
                    # Predecir precio
                    prediction = self.predict_with_lstm(df_processed)
                    
                    if prediction is not None:
                        current_price = df_processed['close'].iloc[-1]
                        
                        # Determinar acción basada en la predicción
                        if prediction > current_price * 1.01:  # 1% de aumento
                            action = 1  # Comprar
                        elif prediction < current_price * 0.99:  # 1% de disminución
                            action = 2  # Vender
                        else:
                            action = 0  # Mantener
                        
                        logger.info(f"Predicción LSTM para {symbol}: {prediction} (actual: {current_price})")
                
                elif self.config['strategy'] == 'dqn':
                    # Cargar agente DQN si no está cargado
                    if self.dqn_agent is None:
                        self.load_dqn_agent(symbol=symbol)

                    # Obtener acción recomendada
                    action = self.get_action_with_dqn(df_processed)

                elif self.config['strategy'] == 'ql':
                    if self.ql_agent is None:
                        self.load_ql_agent()
                    self.train_ql_agent(df_processed)
                    action = self.get_action_with_ql(df_processed, training=False)

                else:
                    # Usar estrategia técnica
                    action = self.get_action_with_strategy(df_processed, strategy=self.config['strategy'])
                
                if action is None:
                    logger.error(f"No se pudo determinar una acción para {symbol}")
                    continue
                
                # Ejecutar operación
                trade_result = self.execute_trade(symbol, action, self.config['trade_amount'])
                
                if trade_result:
                    results.append(trade_result)
                    logger.info(f"Operación completada para {symbol}: {trade_result}")
                
            except Exception as e:
                logger.error(f"Error en ciclo de trading para {symbol}: {e}")
        
        return results
    
    def start_trading_bot(self, interval_minutes=60):
        """
        Inicia el bot de trading en modo continuo.
        
        Args:
            interval_minutes: Intervalo en minutos entre ciclos de trading.
        """
        if self.is_running:
            logger.warning("El bot de trading ya está en ejecución")
            return
        
        self.is_running = True
        logger.info(f"Bot de trading iniciado con intervalo de {interval_minutes} minutos")
        
        try:
            while self.is_running:
                # Ejecutar ciclo de trading
                results = self.run_trading_cycle()
                
                # Guardar resultados
                self._save_trading_results()
                
                # Esperar hasta el próximo ciclo
                logger.info(f"Esperando {interval_minutes} minutos hasta el próximo ciclo")
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("Bot de trading detenido por el usuario")
            self.is_running = False
        
        except Exception as e:
            logger.error(f"Error en el bot de trading: {e}")
            self.is_running = False
    
    def stop_trading_bot(self):
        """
        Detiene el bot de trading.
        """
        self.is_running = False
        logger.info("Bot de trading detenido")
    
    def _save_trading_results(self):
        """
        Guarda los resultados de trading en un archivo.
        """
        try:
            # Crear directorio si no existe
            results_dir = self.results_dir / 'trading'
            os.makedirs(results_dir, exist_ok=True)
            
            # Guardar historial de operaciones
            history_file = results_dir / 'trade_history.json'
            with open(history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=4)
            
            # Guardar posiciones actuales
            positions_file = results_dir / 'current_positions.json'
            with open(positions_file, 'w') as f:
                json.dump(self.current_positions, f, indent=4)
            
            logger.info(f"Resultados de trading guardados en {results_dir}")
        
        except Exception as e:
            logger.error(f"Error al guardar resultados de trading: {e}")
    
    def get_account_summary(self):
        """
        Obtiene un resumen de la cuenta.
        
        Returns:
            Resumen de la cuenta.
        """
        try:
            # Obtener información de la cuenta
            account_info = self.api.get_account_info()
            
            if 'code' in account_info and account_info['code'] != 0:
                logger.error(f"Error al obtener información de la cuenta: {account_info}")
                return None
            
            # Obtener posiciones abiertas
            positions = self.api.get_positions()
            
            if 'code' in positions and positions['code'] != 0:
                logger.error(f"Error al obtener posiciones: {positions}")
                positions = {'data': []}
            
            # Calcular estadísticas de trading
            total_trades = len(self.trade_history)
            profitable_trades = sum(1 for trade in self.trade_history if trade.get('action') == 'sell' and trade.get('profit', 0) > 0)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            total_profit = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('action') == 'sell')
            
            # Crear resumen
            summary = {
                'balance': account_info.get('data', {}).get('balance'),
                'positions': positions.get('data', []),
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'current_positions': self.current_positions,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error al obtener resumen de la cuenta: {e}")
            return None

# Ejemplo de uso
if __name__ == "__main__":
    # Crear bot de trading
    bot = TradingBot()
    
    # Configurar bot
    bot.update_api_credentials(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET"
    )
    
    # Ejecutar un ciclo de trading
    results = bot.run_trading_cycle()
    print(f"Resultados del ciclo de trading: {results}")
    
    # Obtener resumen de la cuenta
    summary = bot.get_account_summary()
    print(f"Resumen de la cuenta: {summary}")

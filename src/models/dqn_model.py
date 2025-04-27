import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from collections import deque

class DQNTradingAgent:
    """
    Agente de trading basado en Deep Q-Learning (DQN) para aprender estrategias
    de trading a partir de datos históricos.
    """
    
    def __init__(self, state_size, action_size=3, model_dir=None):
        """
        Inicializa el agente de trading con DQN.
        
        Args:
            state_size: Tamaño del estado (número de características).
            action_size: Tamaño del espacio de acciones (comprar, vender, mantener).
            model_dir: Directorio donde se guardarán los modelos entrenados.
        """
        self.state_size = state_size
        self.action_size = action_size  # 0: mantener, 1: comprar, 2: vender
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent / 'models' / 'dqn'
        else:
            self.model_dir = Path(model_dir)
        
        # Crear el directorio si no existe
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Hiperparámetros
        self.gamma = 0.95  # Factor de descuento
        self.epsilon = 1.0  # Tasa de exploración inicial
        self.epsilon_min = 0.01  # Tasa de exploración mínima
        self.epsilon_decay = 0.995  # Tasa de decaimiento de epsilon
        self.learning_rate = 0.001  # Tasa de aprendizaje
        self.batch_size = 32  # Tamaño del lote para entrenamiento
        
        # Memoria de experiencia
        self.memory = deque(maxlen=2000)
        
        # Construir modelos
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Construye la red neuronal para el agente DQN.
        
        Returns:
            Modelo de red neuronal compilado.
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """
        Actualiza el modelo objetivo con los pesos del modelo principal.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Almacena una experiencia en la memoria.
        
        Args:
            state: Estado actual.
            action: Acción tomada.
            reward: Recompensa recibida.
            next_state: Estado siguiente.
            done: Indicador de fin de episodio.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Selecciona una acción según la política epsilon-greedy.
        
        Args:
            state: Estado actual.
            training: Si es True, se aplica exploración epsilon-greedy.
            
        Returns:
            Acción seleccionada.
        """
        if training and np.random.rand() <= self.epsilon:
            # Exploración: acción aleatoria
            return random.randrange(self.action_size)
        
        # Explotación: mejor acción según el modelo
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """
        Entrena el modelo con experiencias aleatorias de la memoria.
        
        Args:
            batch_size: Tamaño del lote para entrenamiento.
            
        Returns:
            Pérdida del entrenamiento.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0
        
        # Obtener un lote aleatorio de experiencias
        minibatch = random.sample(self.memory, batch_size)
        
        # Preparar datos para entrenamiento
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # Calcular el valor Q objetivo usando el modelo objetivo
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            
            # Obtener los valores Q actuales
            target_f = self.model.predict(state, verbose=0)
            
            # Actualizar el valor Q para la acción tomada
            target_f[0][action] = target
            
            states[i] = state
            targets[i] = target_f
        
        # Entrenar el modelo
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decaer epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def load(self, filepath):
        """
        Carga los pesos del modelo desde un archivo.
        
        Args:
            filepath: Ruta al archivo de pesos.
        """
        try:
            self.model.load_weights(filepath)
            self.update_target_model()
            print(f"Modelo cargado desde {filepath}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
    
    def save(self, filepath=None):
        """
        Guarda los pesos del modelo en un archivo.
        
        Args:
            filepath: Ruta donde guardar los pesos.
        """
        if filepath is None:
            filepath = self.model_dir / 'dqn_weights.h5'
        
        self.model.save_weights(filepath)
        print(f"Modelo guardado en {filepath}")

class TradingEnvironment:
    """
    Entorno de trading para el agente DQN.
    """
    
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001):
        """
        Inicializa el entorno de trading.
        
        Args:
            df: DataFrame con datos históricos.
            initial_balance: Saldo inicial.
            transaction_fee: Comisión por transacción.
        """
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Características para el estado
        self.features = self._prepare_features()
        
        # Normalizar características
        self.normalized_features = self._normalize_features()
        
        # Estado actual
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.asset_value = 0
        self.total_value = self.balance
        self.previous_total_value = self.total_value
        
        # Historial de trading
        self.trades = []
        self.rewards = []
        self.portfolio_values = [initial_balance]
    
    def _prepare_features(self):
        """
        Prepara las características para el estado.
        
        Returns:
            DataFrame con características seleccionadas.
        """
        # Seleccionar solo columnas numéricas
        features = self.df.select_dtypes(include=['number'])
        
        # Asegurarse de que tenemos las columnas necesarias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in features.columns:
                raise ValueError(f"La columna '{col}' no está presente en los datos.")
        
        return features
    
    def _normalize_features(self):
        """
        Normaliza las características.
        
        Returns:
            DataFrame con características normalizadas.
        """
        normalized = self.features.copy()
        
        for column in normalized.columns:
            normalized[column] = (normalized[column] - normalized[column].min()) / \
                               (normalized[column].max() - normalized[column].min())
        
        return normalized
    
    def reset(self):
        """
        Reinicia el entorno al estado inicial.
        
        Returns:
            Estado inicial.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.asset_value = 0
        self.total_value = self.balance
        self.previous_total_value = self.total_value
        
        self.trades = []
        self.rewards = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_state()
    
    def _get_state(self):
        """
        Obtiene el estado actual.
        
        Returns:
            Estado actual como un array numpy.
        """
        # Obtener características normalizadas para el paso actual
        features = self.normalized_features.iloc[self.current_step].values
        
        # Añadir información sobre la posición actual
        position = np.array([
            self.balance / self.initial_balance,  # Balance normalizado
            self.shares_held > 0,  # Indicador de posición larga
            self.shares_held < 0,  # Indicador de posición corta
        ])
        
        # Combinar características y posición
        state = np.concatenate([features, position])
        
        return np.reshape(state, [1, len(state)])
    
    def step(self, action):
        """
        Ejecuta un paso en el entorno.
        
        Args:
            action: Acción a tomar (0: mantener, 1: comprar, 2: vender).
            
        Returns:
            Tupla (estado siguiente, recompensa, terminado, info).
        """
        # Obtener precio actual
        current_price = self.features.iloc[self.current_step]['close']
        
        # Ejecutar acción
        if action == 1:  # Comprar
            # Calcular número de acciones a comprar (usar 90% del balance)
            max_shares = int(self.balance * 0.9 / current_price)
            
            if max_shares > 0:
                # Calcular costo con comisión
                cost = max_shares * current_price * (1 + self.transaction_fee)
                
                # Actualizar balance y acciones
                self.balance -= cost
                self.shares_held += max_shares
                
                # Registrar operación
                self.trades.append({
                    'step': self.current_step,
                    'price': current_price,
                    'type': 'buy',
                    'shares': max_shares,
                    'cost': cost
                })
        
        elif action == 2:  # Vender
            if self.shares_held > 0:
                # Calcular valor con comisión
                value = self.shares_held * current_price * (1 - self.transaction_fee)
                
                # Actualizar balance y acciones
                self.balance += value
                
                # Registrar operación
                self.trades.append({
                    'step': self.current_step,
                    'price': current_price,
                    'type': 'sell',
                    'shares': self.shares_held,
                    'value': value
                })
                
                self.shares_held = 0
        
        # Actualizar valor de los activos y valor total
        self.asset_value = self.shares_held * current_price
        self.total_value = self.balance + self.asset_value
        
        # Calcular recompensa (cambio porcentual en el valor total)
        reward = (self.total_value - self.previous_total_value) / self.previous_total_value
        self.rewards.append(reward)
        
        # Actualizar valor total anterior
        self.previous_total_value = self.total_value
        
        # Registrar valor del portafolio
        self.portfolio_values.append(self.total_value)
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Verificar si el episodio ha terminado
        done = self.current_step >= len(self.features) - 1
        
        # Información adicional
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'asset_value': self.asset_value,
            'total_value': self.total_value
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, mode='human'):
        """
        Visualiza el estado actual del entorno.
        
        Args:
            mode: Modo de visualización.
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Asset value: ${self.asset_value:.2f}")
            print(f"Total value: ${self.total_value:.2f}")
            print(f"Reward: {self.rewards[-1] if self.rewards else 0:.4f}")
            print("-" * 50)
    
    def plot_results(self, save_path=None):
        """
        Visualiza los resultados del trading.
        
        Args:
            save_path: Ruta para guardar el gráfico.
        """
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Graficar precios y operaciones en el primer subplot
        ax1.plot(self.features['close'], label='Precio de cierre', color='blue')
        
        # Marcar operaciones de compra y venta
        for trade in self.trades:
            if trade['type'] == 'buy':
                ax1.scatter(trade['step'], self.features['close'].iloc[trade['step']], 
                           color='green', marker='^', s=100)
            else:  # sell
                ax1.scatter(trade['step'], self.features['close'].iloc[trade['step']], 
                           color='red', marker='v', s=100)
        
        ax1.set_title('Precios y Operaciones')
        ax1.set_xlabel('Paso')
        ax1.set_ylabel('Precio')
        ax1.legend()
        ax1.grid(True)
        
        # Graficar valor del portafolio en el segundo subplot
        ax2.plot(self.portfolio_values, label='Valor del portafolio', color='purple')
        ax2.set_title('Valor del Portafolio')
        ax2.set_xlabel('Paso')
        ax2.set_ylabel('Valor ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

def train_dqn_agent(env, agent, episodes=100, batch_size=32, target_update_freq=10):
    """
    Entrena un agente DQN en un entorno de trading.
    
    Args:
        env: Entorno de trading.
        agent: Agente DQN.
        episodes: Número de episodios de entrenamiento.
        batch_size: Tamaño del lote para entrenamiento.
        target_update_freq: Frecuencia de actualización del modelo objetivo.
        
    Returns:
        Historial de entrenamiento.
    """
    # Historial de entrenamiento
    history = {
        'episode_rewards': [],
        'portfolio_values': [],
        'losses': []
    }
    
    for episode in range(episodes):
        # Reiniciar el entorno
        state = env.reset()
        total_reward = 0
        losses = []
        
        # Ejecutar un episodio
        done = False
        while not done:
            # Seleccionar acción
            action = agent.act(state)
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            
            # Almacenar experiencia
            agent.remember(state, action, reward, next_state, done)
            
            # Actualizar estado
            state = next_state
            
            # Entrenar agente
            loss = agent.replay(batch_size)
            losses.append(loss)
            
            # Acumular recompensa
            total_reward += reward
        
        # Actualizar modelo objetivo periódicamente
        if episode % target_update_freq == 0:
            agent.update_target_model()
        
        # Registrar resultados del episodio
        history['episode_rewards'].append(total_reward)
        history['portfolio_values'].append(env.total_value)
        history['losses'].append(np.mean(losses) if losses else 0)
        
        # Mostrar progreso
        print(f"Episodio: {episode+1}/{episodes}, "
              f"Recompensa: {total_reward:.4f}, "
              f"Valor final: ${env.total_value:.2f}, "
              f"Epsilon: {agent.epsilon:.4f}")
    
    return history

def evaluate_agent(env, agent, render=False):
    """
    Evalúa un agente entrenado en un entorno de trading.
    
    Args:
        env: Entorno de trading.
        agent: Agente DQN entrenado.
        render: Si es True, visualiza el estado del entorno en cada paso.
        
    Returns:
        Resultados de la evaluación.
    """
    # Reiniciar el entorno
    state = env.reset()
    total_reward = 0
    
    # Ejecutar un episodio completo
    done = False
    while not done:
        # Seleccionar acción (sin exploración)
        action = agent.act(state, training=False)
        
        # Ejecutar acción
        next_state, reward, done, info = env.step(action)
        
        # Actualizar estado
        state = next_state
        
        # Acumular recompensa
        total_reward += reward
        
        # Visualizar estado si se solicita
        if render:
            env.render()
    
    # Calcular rendimiento
    initial_value = env.initial_balance
    final_value = env.total_value
    roi = (final_value - initial_value) / initial_value * 100
    
    # Mostrar resultados
    print(f"Evaluación completada:")
    print(f"Valor inicial: ${initial_value:.2f}")
    print(f"Valor final: ${final_value:.2f}")
    print(f"ROI: {roi:.2f}%")
    print(f"Número de operaciones: {len(env.trades)}")
    
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'roi': roi,
        'total_reward': total_reward,
        'trades': env.trades,
        'portfolio_values': env.portfolio_values
    }

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos procesados
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'crypto' / 'processed_BTC_USD_1d_2y.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    if df is not None:
        # Crear entorno de trading
        env = TradingEnvironment(df)
        
        # Obtener tamaño del estado
        state = env.reset()
        state_size = state.shape[1]
        
        # Crear agente DQN
        agent = DQNTradingAgent(state_size=state_size)
        
        # Entrenar agente
        history = train_dqn_agent(env, agent, episodes=50)
        
        # Guardar modelo entrenado
        agent.save()
        
        # Evaluar agente
        results = evaluate_agent(env, agent)
        
        # Visualizar resultados
        env.plot_results(save_path=str(Path(__file__).parent.parent.parent / 'models' / 'dqn' / 'trading_results.png'))

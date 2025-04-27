import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class BacktestingSystem:
    """
    Sistema de backtesting para evaluar estrategias de trading.
    """
    
    def __init__(self, data_dir=None, results_dir=None):
        """
        Inicializa el sistema de backtesting.
        
        Args:
            data_dir: Directorio donde se encuentran los datos históricos.
            results_dir: Directorio donde se guardarán los resultados.
        """
        self.base_dir = Path(__file__).parent.parent.parent
        
        if data_dir is None:
            self.data_dir = self.base_dir / 'data' / 'processed'
        else:
            self.data_dir = Path(data_dir)
        
        if results_dir is None:
            self.results_dir = self.base_dir / 'results' / 'backtesting'
        else:
            self.results_dir = Path(results_dir)
        
        # Crear el directorio de resultados si no existe
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self, filepath):
        """
        Carga datos desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV con datos procesados.
            
        Returns:
            DataFrame de pandas con los datos cargados.
        """
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"Error al cargar datos desde {filepath}: {e}")
            return None
    
    def simple_moving_average_strategy(self, df, short_window=20, long_window=50):
        """
        Implementa una estrategia de cruce de medias móviles.
        
        Args:
            df: DataFrame con datos históricos.
            short_window: Ventana para la media móvil corta.
            long_window: Ventana para la media móvil larga.
            
        Returns:
            DataFrame con señales de trading.
        """
        # Crear una copia del DataFrame
        signals = df.copy()
        
        # Asegurarse de que tenemos la columna 'close'
        if 'close' not in signals.columns:
            if 'Close' in signals.columns:
                signals['close'] = signals['Close']
            else:
                print("No se encontró la columna 'close' o 'Close' en los datos.")
                return None
        
        # Calcular medias móviles
        signals['sma_short'] = signals['close'].rolling(window=short_window, min_periods=1).mean()
        signals['sma_long'] = signals['close'].rolling(window=long_window, min_periods=1).mean()
        
        # Generar señales
        signals['signal'] = 0.0
        signals['signal'][short_window:] = np.where(
            signals['sma_short'][short_window:] > signals['sma_long'][short_window:], 1.0, 0.0
        )
        
        # Generar posiciones (1: long, -1: short, 0: neutral)
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def bollinger_bands_strategy(self, df, window=20, num_std=2):
        """
        Implementa una estrategia basada en bandas de Bollinger.
        
        Args:
            df: DataFrame con datos históricos.
            window: Ventana para el cálculo de las bandas.
            num_std: Número de desviaciones estándar para las bandas.
            
        Returns:
            DataFrame con señales de trading.
        """
        # Crear una copia del DataFrame
        signals = df.copy()
        
        # Asegurarse de que tenemos la columna 'close'
        if 'close' not in signals.columns:
            if 'Close' in signals.columns:
                signals['close'] = signals['Close']
            else:
                print("No se encontró la columna 'close' o 'Close' en los datos.")
                return None
        
        # Calcular bandas de Bollinger
        signals['sma'] = signals['close'].rolling(window=window, min_periods=1).mean()
        signals['std'] = signals['close'].rolling(window=window, min_periods=1).std()
        signals['upper_band'] = signals['sma'] + (signals['std'] * num_std)
        signals['lower_band'] = signals['sma'] - (signals['std'] * num_std)
        
        # Generar señales
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['close'] < signals['lower_band'], 1.0, signals['signal'])  # Compra
        signals['signal'] = np.where(signals['close'] > signals['upper_band'], -1.0, signals['signal'])  # Venta
        
        # Generar posiciones
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def rsi_strategy(self, df, window=14, oversold=30, overbought=70):
        """
        Implementa una estrategia basada en el RSI (Relative Strength Index).
        
        Args:
            df: DataFrame con datos históricos.
            window: Ventana para el cálculo del RSI.
            oversold: Nivel de sobreventa.
            overbought: Nivel de sobrecompra.
            
        Returns:
            DataFrame con señales de trading.
        """
        # Crear una copia del DataFrame
        signals = df.copy()
        
        # Asegurarse de que tenemos la columna 'close'
        if 'close' not in signals.columns:
            if 'Close' in signals.columns:
                signals['close'] = signals['Close']
            else:
                print("No se encontró la columna 'close' o 'Close' en los datos.")
                return None
        
        # Calcular RSI si no está presente
        if 'rsi_14' not in signals.columns:
            # Calcular cambios diarios
            delta = signals['close'].diff()
            
            # Separar ganancias y pérdidas
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calcular promedio de ganancias y pérdidas
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            
            # Calcular RS y RSI
            rs = avg_gain / avg_loss
            signals['rsi'] = 100 - (100 / (1 + rs))
        else:
            signals['rsi'] = signals['rsi_14']
        
        # Generar señales
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, signals['signal'])  # Compra en sobreventa
        signals['signal'] = np.where(signals['rsi'] > overbought, -1.0, signals['signal'])  # Venta en sobrecompra
        
        # Generar posiciones
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def macd_strategy(self, df, fast=12, slow=26, signal=9):
        """
        Implementa una estrategia basada en el MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame con datos históricos.
            fast: Período para la EMA rápida.
            slow: Período para la EMA lenta.
            signal: Período para la línea de señal.
            
        Returns:
            DataFrame con señales de trading.
        """
        # Crear una copia del DataFrame
        signals = df.copy()
        
        # Asegurarse de que tenemos la columna 'close'
        if 'close' not in signals.columns:
            if 'Close' in signals.columns:
                signals['close'] = signals['Close']
            else:
                print("No se encontró la columna 'close' o 'Close' en los datos.")
                return None
        
        # Calcular MACD si no está presente
        if 'macd' not in signals.columns or 'macd_signal' not in signals.columns:
            # Calcular EMAs
            signals['ema_fast'] = signals['close'].ewm(span=fast, adjust=False).mean()
            signals['ema_slow'] = signals['close'].ewm(span=slow, adjust=False).mean()
            
            # Calcular MACD y línea de señal
            signals['macd'] = signals['ema_fast'] - signals['ema_slow']
            signals['macd_signal'] = signals['macd'].ewm(span=signal, adjust=False).mean()
            signals['macd_hist'] = signals['macd'] - signals['macd_signal']
        
        # Generar señales
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['macd'] > signals['macd_signal'], 1.0, signals['signal'])  # Compra
        signals['signal'] = np.where(signals['macd'] < signals['macd_signal'], -1.0, signals['signal'])  # Venta
        
        # Generar posiciones
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def backtest_strategy(self, signals, initial_capital=10000, position_size=1.0, commission=0.001):
        """
        Realiza un backtesting de una estrategia de trading.
        
        Args:
            signals: DataFrame con señales de trading.
            initial_capital: Capital inicial.
            position_size: Tamaño de la posición (proporción del capital).
            commission: Comisión por operación.
            
        Returns:
            DataFrame con resultados del backtesting.
        """
        # Crear una copia del DataFrame
        portfolio = signals.copy()
        
        # Asegurarse de que tenemos las columnas necesarias
        if 'close' not in portfolio.columns:
            if 'Close' in portfolio.columns:
                portfolio['close'] = portfolio['Close']
            else:
                print("No se encontró la columna 'close' o 'Close' en los datos.")
                return None
        
        if 'position' not in portfolio.columns:
            print("No se encontró la columna 'position' en los datos.")
            return None
        
        # Inicializar columnas de resultados
        portfolio['holdings'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['total'] = initial_capital
        portfolio['returns'] = 0.0
        
        # Calcular resultados
        for i in range(len(portfolio)):
            if i > 0:
                # Actualizar posiciones
                if portfolio['position'].iloc[i] == 1:  # Compra
                    # Calcular número de acciones a comprar
                    shares = (portfolio['cash'].iloc[i-1] * position_size) / portfolio['close'].iloc[i]
                    # Restar comisión
                    cost = shares * portfolio['close'].iloc[i] * (1 + commission)
                    
                    portfolio['holdings'].iloc[i] = portfolio['holdings'].iloc[i-1] + shares
                    portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1] - cost
                
                elif portfolio['position'].iloc[i] == -1:  # Venta
                    # Calcular valor de las acciones vendidas
                    value = portfolio['holdings'].iloc[i-1] * portfolio['close'].iloc[i] * (1 - commission)
                    
                    portfolio['holdings'].iloc[i] = 0
                    portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1] + value
                
                else:  # Mantener
                    portfolio['holdings'].iloc[i] = portfolio['holdings'].iloc[i-1]
                    portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1]
                
                # Calcular valor total y retornos
                portfolio['total'].iloc[i] = portfolio['cash'].iloc[i] + (portfolio['holdings'].iloc[i] * portfolio['close'].iloc[i])
                portfolio['returns'].iloc[i] = portfolio['total'].iloc[i] / portfolio['total'].iloc[i-1] - 1
        
        return portfolio
    
    def calculate_performance_metrics(self, portfolio):
        """
        Calcula métricas de rendimiento para una estrategia.
        
        Args:
            portfolio: DataFrame con resultados del backtesting.
            
        Returns:
            Diccionario con métricas de rendimiento.
        """
        # Calcular retorno total
        total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
        
        # Calcular retorno anualizado (asumiendo datos diarios)
        n_days = len(portfolio)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Calcular volatilidad anualizada
        daily_returns = portfolio['returns'].dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Calcular ratio de Sharpe (asumiendo tasa libre de riesgo de 0%)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Calcular drawdown máximo
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calcular número de operaciones
        n_trades = portfolio['position'].abs().sum()
        
        # Calcular porcentaje de operaciones ganadoras
        winning_trades = portfolio[portfolio['returns'] > 0]['position'].abs().sum()
        win_rate = winning_trades / n_trades if n_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate
        }
    
    def plot_strategy_results(self, portfolio, title='Resultados de la Estrategia', save_path=None):
        """
        Visualiza los resultados de una estrategia de trading.
        
        Args:
            portfolio: DataFrame con resultados del backtesting.
            title: Título del gráfico.
            save_path: Ruta para guardar el gráfico.
        """
        # Crear figura con tres subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Graficar precio y señales en el primer subplot
        ax1.plot(portfolio['close'], label='Precio de cierre', color='blue', alpha=0.5)
        
        # Marcar señales de compra y venta
        buy_signals = portfolio[portfolio['position'] == 1].index
        sell_signals = portfolio[portfolio['position'] == -1].index
        
        ax1.scatter(buy_signals, portfolio.loc[buy_signals, 'close'], 
                   color='green', marker='^', s=100, label='Compra')
        ax1.scatter(sell_signals, portfolio.loc[sell_signals, 'close'], 
                   color='red', marker='v', s=100, label='Venta')
        
        ax1.set_title(f'{title} - Precio y Señales')
        ax1.set_ylabel('Precio')
        ax1.legend()
        ax1.grid(True)
        
        # Graficar valor del portafolio en el segundo subplot
        ax2.plot(portfolio['total'], label='Valor del portafolio', color='purple')
        ax2.set_title('Valor del Portafolio')
        ax2.set_ylabel('Valor ($)')
        ax2.legend()
        ax2.grid(True)
        
        # Graficar retornos acumulados en el tercer subplot
        cumulative_returns = (1 + portfolio['returns'].fillna(0)).cumprod()
        ax3.plot(cumulative_returns, label='Retornos acumulados', color='green')
        ax3.set_title('Retornos Acumulados')
        ax3.set_xlabel('Fecha')
        ax3.set_ylabel('Retorno')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def compare_strategies(self, data, strategies, initial_capital=10000, save_path=None):
        """
        Compara múltiples estrategias de trading.
        
        Args:
            data: DataFrame con datos históricos.
            strategies: Diccionario con estrategias a comparar.
            initial_capital: Capital inicial.
            save_path: Ruta para guardar el gráfico.
            
        Returns:
            DataFrame con métricas de rendimiento para cada estrategia.
        """
        results = {}
        metrics = {}
        
        # Ejecutar backtesting para cada estrategia
        for name, strategy_func in strategies.items():
            signals = strategy_func(data)
            portfolio = self.backtest_strategy(signals, initial_capital=initial_capital)
            results[name] = portfolio
            metrics[name] = self.calculate_performance_metrics(portfolio)
        
        # Crear DataFrame con métricas
        metrics_df = pd.DataFrame(metrics).T
        
        # Graficar comparación de estrategias
        plt.figure(figsize=(16, 8))
        
        for name, portfolio in results.items():
            cumulative_returns = (1 + portfolio['returns'].fillna(0)).cumprod()
            plt.plot(cumulative_returns, label=name)
        
        plt.title('Comparación de Estrategias - Retornos Acumulados')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
        
        return metrics_df

# Ejemplo de uso
if __name__ == "__main__":
    backtester = BacktestingSystem()
    
    # Cargar datos procesados
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'crypto' / 'processed_BTC_USD_1d_2y.csv'
    df = backtester.load_data(data_path)
    
    if df is not None:
        # Definir estrategias a comparar
        strategies = {
            'SMA': lambda data: backtester.simple_moving_average_strategy(data, short_window=20, long_window=50),
            'Bollinger': lambda data: backtester.bollinger_bands_strategy(data, window=20, num_std=2),
            'RSI': lambda data: backtester.rsi_strategy(data, window=14, oversold=30, overbought=70),
            'MACD': lambda data: backtester.macd_strategy(data, fast=12, slow=26, signal=9)
        }
        
        # Comparar estrategias
        metrics = backtester.compare_strategies(
            df, 
            strategies, 
            initial_capital=10000,
            save_path=str(backtester.results_dir / 'strategy_comparison.png')
        )
        
        print("Métricas de rendimiento:")
        print(metrics)
        
        # Ejecutar y visualizar una estrategia específica
        signals = backtester.macd_strategy(df)
        portfolio = backtester.backtest_strategy(signals, initial_capital=10000)
        backtester.plot_strategy_results(
            portfolio, 
            title='Estrategia MACD',
            save_path=str(backtester.results_dir / 'macd_strategy_results.png')
        )

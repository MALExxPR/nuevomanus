import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from pathlib import Path

class LSTMPricePredictor:
    """
    Clase para predecir precios de activos financieros utilizando redes LSTM.
    """
    
    def __init__(self, model_dir=None):
        """
        Inicializa el predictor de precios con LSTM.
        
        Args:
            model_dir: Directorio donde se guardarán los modelos entrenados.
        """
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent / 'models' / 'lstm'
        else:
            self.model_dir = Path(model_dir)
        
        # Crear el directorio si no existe
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Inicializar el modelo
        self.model = None
        self.scaler = None
    
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
    
    def prepare_data(self, df, target_column='close', sequence_length=60, train_split=0.8):
        """
        Prepara los datos para el entrenamiento del modelo LSTM.
        
        Args:
            df: DataFrame de pandas con los datos.
            target_column: Columna objetivo para la predicción.
            sequence_length: Longitud de la secuencia para el modelo LSTM.
            train_split: Proporción de datos para entrenamiento.
            
        Returns:
            Datos de entrenamiento y prueba preparados para el modelo LSTM.
        """
        if df is None or df.empty:
            return None
        
        # Asegurarse de que el target_column existe
        if target_column not in df.columns:
            print(f"La columna objetivo '{target_column}' no existe en el DataFrame.")
            return None
        
        # Seleccionar solo las columnas numéricas
        df = df.select_dtypes(include=['number'])
        
        # Normalizar los datos
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(df)
        
        # Obtener el índice de la columna objetivo
        target_idx = df.columns.get_loc(target_column)
        
        # Crear secuencias para el modelo LSTM
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # Dividir en conjuntos de entrenamiento y prueba
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'df': df,
            'target_idx': target_idx
        }
    
    def build_model(self, input_shape, units=50, dropout_rate=0.2):
        """
        Construye el modelo LSTM.
        
        Args:
            input_shape: Forma de los datos de entrada (sequence_length, n_features).
            units: Número de unidades en la capa LSTM.
            dropout_rate: Tasa de dropout para regularización.
            
        Returns:
            Modelo LSTM compilado.
        """
        model = Sequential()
        
        # Primera capa LSTM con dropout
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Segunda capa LSTM con dropout
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Capa de salida
        model.add(Dense(units=1))
        
        # Compilar el modelo
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train_model(self, data, epochs=50, batch_size=32, patience=10, model_name='lstm_model'):
        """
        Entrena el modelo LSTM.
        
        Args:
            data: Diccionario con los datos preparados.
            epochs: Número de épocas para el entrenamiento.
            batch_size: Tamaño del lote para el entrenamiento.
            patience: Número de épocas para early stopping.
            model_name: Nombre para guardar el modelo entrenado.
            
        Returns:
            Historial de entrenamiento.
        """
        if data is None:
            return None
        
        # Construir el modelo
        input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
        self.model = self.build_model(input_shape)
        
        # Definir callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(filepath=str(self.model_dir / f"{model_name}.h5"), 
                           save_best_only=True, monitor='val_loss')
        ]
        
        # Entrenar el modelo
        history = self.model.fit(
            data['X_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_test'], data['y_test']),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, data, sequence_length=60):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            data: Datos para realizar predicciones.
            sequence_length: Longitud de la secuencia para el modelo LSTM.
            
        Returns:
            Predicciones y valores reales.
        """
        if self.model is None:
            print("El modelo no ha sido entrenado.")
            return None
        
        # Realizar predicciones
        predictions = self.model.predict(data['X_test'])
        
        # Crear un array para desnormalizar las predicciones
        dummy = np.zeros((len(predictions), data['df'].shape[1]))
        dummy[:, data['target_idx']] = predictions.flatten()
        
        # Desnormalizar las predicciones
        predictions_denorm = self.scaler.inverse_transform(dummy)[:, data['target_idx']]
        
        # Desnormalizar los valores reales
        dummy = np.zeros((len(data['y_test']), data['df'].shape[1]))
        dummy[:, data['target_idx']] = data['y_test']
        real_values_denorm = self.scaler.inverse_transform(dummy)[:, data['target_idx']]
        
        return {
            'predictions': predictions_denorm,
            'real_values': real_values_denorm
        }
    
    def evaluate(self, predictions, real_values):
        """
        Evalúa el rendimiento del modelo.
        
        Args:
            predictions: Predicciones del modelo.
            real_values: Valores reales.
            
        Returns:
            Métricas de evaluación.
        """
        # Calcular el error cuadrático medio (MSE)
        mse = np.mean((predictions - real_values) ** 2)
        
        # Calcular la raíz del error cuadrático medio (RMSE)
        rmse = np.sqrt(mse)
        
        # Calcular el error absoluto medio (MAE)
        mae = np.mean(np.abs(predictions - real_values))
        
        # Calcular el error porcentual absoluto medio (MAPE)
        mape = np.mean(np.abs((real_values - predictions) / real_values)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def plot_results(self, predictions, real_values, title='Predicción de precios', save_path=None):
        """
        Visualiza los resultados de la predicción.
        
        Args:
            predictions: Predicciones del modelo.
            real_values: Valores reales.
            title: Título del gráfico.
            save_path: Ruta para guardar el gráfico.
        """
        plt.figure(figsize=(16, 8))
        plt.plot(real_values, color='blue', label='Valores reales')
        plt.plot(predictions, color='red', label='Predicciones')
        plt.title(title)
        plt.xlabel('Tiempo')
        plt.ylabel('Precio')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def save_model(self, filepath=None):
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo.
        """
        if self.model is None:
            print("No hay modelo para guardar.")
            return
        
        if filepath is None:
            filepath = self.model_dir / 'lstm_model.h5'
        
        self.model.save(filepath)
        print(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            filepath: Ruta al modelo guardado.
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Modelo cargado desde {filepath}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    predictor = LSTMPricePredictor()
    
    # Cargar datos procesados
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'crypto' / 'processed_BTC_USD_1d_2y.csv'
    df = predictor.load_data(data_path)
    
    if df is not None:
        # Preparar datos
        data = predictor.prepare_data(df, target_column='close', sequence_length=60)
        
        # Entrenar modelo
        history = predictor.train_model(data, epochs=50, batch_size=32, model_name='btc_lstm')
        
        # Realizar predicciones
        results = predictor.predict(data)
        
        # Evaluar modelo
        metrics = predictor.evaluate(results['predictions'], results['real_values'])
        print(f"Métricas de evaluación: {metrics}")
        
        # Visualizar resultados
        predictor.plot_results(
            results['predictions'], 
            results['real_values'], 
            title='Predicción de precios de BTC',
            save_path=str(Path(__file__).parent.parent.parent / 'models' / 'lstm' / 'btc_prediction.png')
        )

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import os

class RandomForestPricePredictor:
    """Simple price predictor using RandomForestRegressor."""

    def __init__(self, model_dir=None):
        self.model = None
        self.scaler = None
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent / 'models' / 'rf'
        else:
            self.model_dir = Path(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_data(self, df, target_column='close', train_split=0.8):
        if df is None or df.empty or target_column not in df.columns:
            return None
        df = df.select_dtypes(include=['number'])
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(df)
        target_idx = df.columns.get_loc(target_column)
        X = np.delete(scaled, target_idx, axis=1)
        y = scaled[:, target_idx]
        split = int(len(X) * train_split)
        return {
            'X_train': X[:split],
            'X_test': X[split:],
            'y_train': y[:split],
            'y_test': y[split:],
        }

    def train_model(self, data, n_estimators=100, model_name='rf_model'):
        if data is None:
            return None
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.model.fit(data['X_train'], data['y_train'])
        mse = mean_squared_error(data['y_test'], self.model.predict(data['X_test']))
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_dir / f"{model_name}.joblib")
        return mse

    def load_model(self, model_file):
        if not Path(model_file).exists():
            return False
        bundle = joblib.load(model_file)
        self.model = bundle.get('model')
        self.scaler = bundle.get('scaler')
        return True

    def predict(self, df):
        if self.model is None or self.scaler is None:
            return None
        df = df.select_dtypes(include=['number'])
        scaled = self.scaler.transform(df)
        return self.model.predict(np.delete(scaled, df.columns.get_loc('close'), axis=1))

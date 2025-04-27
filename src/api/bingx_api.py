import requests
import hmac
import hashlib
import time
import json
import pandas as pd
from pathlib import Path
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bingx_api')

class BingXAPI:
    """
    Clase para interactuar con la API de BingX para trading de criptomonedas y forex.
    """
    
    def __init__(self, api_key=None, api_secret=None, config_file=None):
        """
        Inicializa la conexión con la API de BingX.
        
        Args:
            api_key: API Key de BingX.
            api_secret: API Secret de BingX.
            config_file: Ruta al archivo de configuración con las credenciales.
        """
        self.base_url = "https://open-api.bingx.com"
        
        # Cargar credenciales
        if api_key and api_secret:
            self.api_key = api_key
            self.api_secret = api_secret
        elif config_file:
            self._load_config(config_file)
        else:
            # Buscar archivo de configuración por defecto
            default_config = Path.home() / '.bingx' / 'config.json'
            if default_config.exists():
                self._load_config(default_config)
            else:
                logger.warning("No se proporcionaron credenciales de API. Algunas funciones no estarán disponibles.")
                self.api_key = None
                self.api_secret = None
    
    def _load_config(self, config_file):
        """
        Carga las credenciales desde un archivo de configuración.
        
        Args:
            config_file: Ruta al archivo de configuración.
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.api_key = config.get('api_key')
                self.api_secret = config.get('api_secret')
                logger.info(f"Credenciales cargadas desde {config_file}")
        except Exception as e:
            logger.error(f"Error al cargar el archivo de configuración: {e}")
            self.api_key = None
            self.api_secret = None
    
    def _generate_signature(self, params):
        """
        Genera la firma para autenticar las solicitudes a la API.
        
        Args:
            params: Parámetros de la solicitud.
            
        Returns:
            Firma generada.
        """
        # Convertir parámetros a cadena de consulta
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Generar firma HMAC-SHA256
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _make_request(self, method, endpoint, params=None, signed=False):
        """
        Realiza una solicitud a la API de BingX.
        
        Args:
            method: Método HTTP (GET, POST, etc.).
            endpoint: Endpoint de la API.
            params: Parámetros de la solicitud.
            signed: Si es True, la solicitud requiere autenticación.
            
        Returns:
            Respuesta de la API.
        """
        url = f"{self.base_url}{endpoint}"
        
        # Preparar parámetros
        if params is None:
            params = {}
        
        # Añadir timestamp para solicitudes firmadas
        if signed:
            if self.api_key is None or self.api_secret is None:
                raise ValueError("Se requieren credenciales de API para esta operación.")
            
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['apiKey'] = self.api_key
            
            # Generar firma
            params['signature'] = self._generate_signature(params)
        
        # Realizar solicitud
        try:
            if method == 'GET':
                response = requests.get(url, params=params)
            elif method == 'POST':
                response = requests.post(url, json=params)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            # Verificar respuesta
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en la solicitud a la API: {e}")
            return {'code': -1, 'msg': str(e)}
    
    def get_server_time(self):
        """
        Obtiene la hora del servidor de BingX.
        
        Returns:
            Hora del servidor en milisegundos.
        """
        endpoint = "/openApi/swap/v2/server/time"
        response = self._make_request('GET', endpoint)
        
        return response
    
    def get_exchange_info(self):
        """
        Obtiene información sobre los pares de trading disponibles.
        
        Returns:
            Información de los pares de trading.
        """
        endpoint = "/openApi/swap/v2/quote/contracts"
        response = self._make_request('GET', endpoint)
        
        return response
    
    def get_ticker(self, symbol=None):
        """
        Obtiene información de ticker para un símbolo o todos los símbolos.
        
        Args:
            symbol: Símbolo específico (opcional).
            
        Returns:
            Información de ticker.
        """
        endpoint = "/openApi/swap/v2/quote/ticker"
        params = {}
        
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request('GET', endpoint, params)
        
        return response
    
    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """
        Obtiene datos de velas (klines) para un símbolo.
        
        Args:
            symbol: Símbolo del par de trading.
            interval: Intervalo de tiempo (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M).
            start_time: Tiempo de inicio en milisegundos (opcional).
            end_time: Tiempo de fin en milisegundos (opcional).
            limit: Número máximo de velas a devolver (máximo 1000).
            
        Returns:
            Datos de velas.
        """
        endpoint = "/openApi/swap/v2/quote/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        
        if end_time:
            params['endTime'] = end_time
        
        response = self._make_request('GET', endpoint, params)
        
        # Convertir a DataFrame si la respuesta es exitosa
        if response.get('code') == 0 and 'data' in response:
            data = response['data']
            df = pd.DataFrame(data, columns=[
                'openTime', 'open', 'high', 'low', 'close', 'volume', 
                'closeTime', 'quoteAssetVolume', 'trades', 
                'takerBuyBaseAssetVolume', 'takerBuyQuoteAssetVolume', 'ignore'
            ])
            
            # Convertir timestamps a datetime
            df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
            df['closeTime'] = pd.to_datetime(df['closeTime'], unit='ms')
            
            # Convertir columnas numéricas
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'quoteAssetVolume', 'takerBuyBaseAssetVolume', 'takerBuyQuoteAssetVolume']
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            
            return df
        
        return response
    
    def get_account_info(self):
        """
        Obtiene información de la cuenta.
        
        Returns:
            Información de la cuenta.
        """
        endpoint = "/openApi/swap/v2/user/balance"
        response = self._make_request('GET', endpoint, signed=True)
        
        return response
    
    def get_positions(self):
        """
        Obtiene las posiciones abiertas.
        
        Returns:
            Posiciones abiertas.
        """
        endpoint = "/openApi/swap/v2/user/positions"
        response = self._make_request('GET', endpoint, signed=True)
        
        return response
    
    def place_order(self, symbol, side, order_type, quantity, price=None, time_in_force="GTC"):
        """
        Coloca una orden de trading.
        
        Args:
            symbol: Símbolo del par de trading.
            side: Lado de la orden (BUY o SELL).
            order_type: Tipo de orden (LIMIT, MARKET, etc.).
            quantity: Cantidad a comprar/vender.
            price: Precio para órdenes limit (opcional).
            time_in_force: Tiempo en vigor (GTC, IOC, FOK).
            
        Returns:
            Resultado de la orden.
        """
        endpoint = "/openApi/swap/v2/trade/order"
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'timeInForce': time_in_force
        }
        
        if order_type == 'LIMIT' and price is not None:
            params['price'] = price
        
        response = self._make_request('POST', endpoint, params, signed=True)
        
        return response
    
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """
        Cancela una orden existente.
        
        Args:
            symbol: Símbolo del par de trading.
            order_id: ID de la orden (opcional).
            client_order_id: ID de cliente de la orden (opcional).
            
        Returns:
            Resultado de la cancelación.
        """
        endpoint = "/openApi/swap/v2/trade/cancel"
        
        params = {
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['clientOrderId'] = client_order_id
        else:
            raise ValueError("Se debe proporcionar order_id o client_order_id")
        
        response = self._make_request('POST', endpoint, params, signed=True)
        
        return response
    
    def get_order(self, symbol, order_id=None, client_order_id=None):
        """
        Obtiene información sobre una orden específica.
        
        Args:
            symbol: Símbolo del par de trading.
            order_id: ID de la orden (opcional).
            client_order_id: ID de cliente de la orden (opcional).
            
        Returns:
            Información de la orden.
        """
        endpoint = "/openApi/swap/v2/trade/order"
        
        params = {
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['clientOrderId'] = client_order_id
        else:
            raise ValueError("Se debe proporcionar order_id o client_order_id")
        
        response = self._make_request('GET', endpoint, params, signed=True)
        
        return response
    
    def get_open_orders(self, symbol=None):
        """
        Obtiene las órdenes abiertas.
        
        Args:
            symbol: Símbolo del par de trading (opcional).
            
        Returns:
            Órdenes abiertas.
        """
        endpoint = "/openApi/swap/v2/trade/openOrders"
        
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request('GET', endpoint, params, signed=True)
        
        return response
    
    def get_order_history(self, symbol, limit=500):
        """
        Obtiene el historial de órdenes.
        
        Args:
            symbol: Símbolo del par de trading.
            limit: Número máximo de órdenes a devolver.
            
        Returns:
            Historial de órdenes.
        """
        endpoint = "/openApi/swap/v2/trade/allOrders"
        
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        response = self._make_request('GET', endpoint, params, signed=True)
        
        return response
    
    def save_config(self, api_key, api_secret, config_file=None):
        """
        Guarda las credenciales de API en un archivo de configuración.
        
        Args:
            api_key: API Key de BingX.
            api_secret: API Secret de BingX.
            config_file: Ruta al archivo de configuración (opcional).
            
        Returns:
            True si se guardó correctamente, False en caso contrario.
        """
        if config_file is None:
            config_dir = Path.home() / '.bingx'
            os.makedirs(config_dir, exist_ok=True)
            config_file = config_dir / 'config.json'
        
        try:
            config = {
                'api_key': api_key,
                'api_secret': api_secret
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Actualizar credenciales actuales
            self.api_key = api_key
            self.api_secret = api_secret
            
            logger.info(f"Credenciales guardadas en {config_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error al guardar el archivo de configuración: {e}")
            return False

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia de la API
    api = BingXAPI()
    
    # Obtener hora del servidor
    server_time = api.get_server_time()
    print(f"Hora del servidor: {server_time}")
    
    # Obtener información de ticker para BTC/USDT
    ticker = api.get_ticker("BTC-USDT")
    print(f"Ticker BTC/USDT: {ticker}")
    
    # Obtener datos de velas para BTC/USDT (intervalo de 1 día, últimos 30 días)
    klines = api.get_klines("BTC-USDT", "1d", limit=30)
    print(f"Datos de velas BTC/USDT: {klines.head()}")

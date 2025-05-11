import ccxt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

class MarketDataProvider:
    """
    A class to fetch market data from exchanges using CCXT and calculate
    various indicators like moving averages.
    """
    
    def __init__(self, exchange_id='binance', timeframe='1d', limit=1000):
        """
        Initialize the market data provider.
        
        Args:
            exchange_id (str): ID of the exchange to use (default: 'binance')
            timeframe (str): Timeframe for data (default: '1d')
            limit (int): Maximum number of candles to fetch (default: 1000)
        """
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = self._initialize_exchange(exchange_id)
        
        # Ensure the static directory exists
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(static_dir, exist_ok=True)
        self.data_file = os.path.join(static_dir, 'market_data.json')
        
        # Available assets (can be expanded)
        self.crypto_assets = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        
        # Stock symbols (if using stock exchange like NASDAQ through CCXT)
        self.stock_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Available timeframes
        self.available_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    
    def _initialize_exchange(self, exchange_id):
        """
        Initialize the CCXT exchange.
        
        Args:
            exchange_id (str): ID of the exchange to use
            
        Returns:
            ccxt.Exchange: Initialized exchange object
        """
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        return exchange
    
    def get_available_assets(self):
        """
        Get the list of available assets.
        
        Returns:
            dict: Dictionary with crypto and stock assets
        """
        return {
            'crypto': self.crypto_assets,
            'stocks': self.stock_assets
        }
    
    def get_available_timeframes(self):
        """
        Get the list of available timeframes.
        
        Returns:
            list: List of available timeframes
        """
        return self.available_timeframes
    
    def fetch_ohlcv_data(self, symbol, timeframe=None, limit=None):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Args:
            symbol (str): Symbol to fetch data for
            timeframe (str, optional): Timeframe for data
            limit (int, optional): Maximum number of candles to fetch
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if timeframe is None:
            timeframe = self.timeframe
        
        if limit is None:
            limit = self.limit
            
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Round to 2 decimal places
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].round(2)
            
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_moving_averages(self, data, periods=[20, 50, 200]):
        """
        Calculate moving averages for the given data.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            periods (list): List of periods to calculate MAs for
            
        Returns:
            pd.DataFrame: DataFrame with price data and MAs
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        for period in periods:
            column_name = f'ma_{period}'
            df[column_name] = df['close'].rolling(window=period).mean().round(2)
        
        return df
    
    def prepare_data_for_nn(self, symbol, ma_periods=[20, 50, 200], timeframe=None, limit=None):
        """
        Prepare data for neural network training.
        
        Args:
            symbol (str): Symbol to fetch data for
            ma_periods (list): List of periods to calculate MAs for
            timeframe (str, optional): Timeframe for data
            limit (int, optional): Maximum number of candles to fetch
            
        Returns:
            dict: Data ready for neural network consumption
        """
        df = self.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if df.empty:
            return {
                'error': f"Could not fetch data for {symbol}"
            }
        
        df_with_mas = self.calculate_moving_averages(df, ma_periods)
        
        # Convert to format suitable for the neural network
        # Format dates appropriately based on timeframe
        if timeframe and ('m' in timeframe or 'h' in timeframe):
            # For minute/hour timeframes, include the full timestamp
            dates = df_with_mas.index.strftime('%Y-%m-%dT%H:%M:%S').tolist()
        else:
            # For day and above timeframes, just include the date
            dates = df_with_mas.index.strftime('%Y-%m-%d').tolist()
            
        price_data = df_with_mas['close'].tolist()
        
        ma_data = {}
        for period in ma_periods:
            column_name = f'ma_{period}'
            # Fill NaN values with the first available value
            # For short MAs (like 2-period), ensure we handle NaN values properly
            ma_series = df_with_mas[column_name]
            if ma_series.isna().any():
                # Replace NaN with first available value or the price if still NaN
                # Using bfill() instead of fillna(method='bfill') as recommended
                ma_series = ma_series.bfill().fillna(df_with_mas['close'])
            ma_values = ma_series.tolist()
            
            # Handle any remaining NaN values (convert to None for JSON)
            ma_values = [None if isinstance(x, float) and np.isnan(x) else x for x in ma_values]
            ma_data[period] = ma_values
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe or self.timeframe,
            'dates': dates,
            'prices': price_data,
            'moving_averages': ma_data
        }
        
        return result
    
    def save_data_to_json(self, data):
        """
        Save data to JSON file.
        
        Args:
            data (dict): Data to save
        """
        # Custom JSON encoder to handle NaN values
        class NaNEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float) and np.isnan(obj):
                    return None
                return super(NaNEncoder, self).default(obj)
                
        with open(self.data_file, 'w') as f:
            json.dump(data, f, cls=NaNEncoder)
    
    def load_data_from_json(self):
        """
        Load data from JSON file.
        
        Returns:
            dict: Loaded data
        """
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def process_market_data(self, symbol, ma_periods, timeframe=None, limit=None):
        """
        Process market data for a symbol and save it.
        
        Args:
            symbol (str): Symbol to process
            ma_periods (list): List of periods to calculate MAs for
            timeframe (str, optional): Timeframe for data
            limit (int, optional): Maximum number of candles to fetch
            
        Returns:
            dict: Processed market data
        """
        data = self.prepare_data_for_nn(symbol, ma_periods, timeframe, limit)
        
        if 'error' not in data:
            self.save_data_to_json(data)
        
        return data

# Helper functions for use in Flask routes

def get_market_data_provider(exchange='binance', timeframe='1d', limit=1000):
    """
    Get a market data provider instance.
    
    Args:
        exchange (str): Exchange ID
        timeframe (str): Timeframe for data
        limit (int): Maximum number of candles to fetch
        
    Returns:
        MarketDataProvider: Market data provider instance
    """
    return MarketDataProvider(exchange_id=exchange, timeframe=timeframe, limit=limit)

def get_data_for_asset(symbol, ma_periods=[20, 50, 200], timeframe='1d', limit=1000, exchange='binance'):
    """
    Process market data for an asset.
    
    Args:
        symbol (str): Symbol to process
        ma_periods (list): List of periods to calculate MAs for
        timeframe (str): Timeframe for data
        limit (int): Maximum number of candles to fetch
        exchange (str): Exchange ID
        
    Returns:
        dict: Processed market data
    """
    provider = get_market_data_provider(exchange, timeframe, limit)
    return provider.process_market_data(symbol, ma_periods, timeframe, limit)

# Example usage
if __name__ == "__main__":
    # Example of how to use the class
    provider = MarketDataProvider()
    btc_data = provider.process_market_data('BTC/USDT', [20, 50, 200])
    print(f"Processed data for BTC/USDT with {len(btc_data['prices'])} data points") 
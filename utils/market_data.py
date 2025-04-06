import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class MarketDataUtil:
    """
    Utility class for market data operations and technical indicators.
    """
    
    def __init__(self, mt5_connector):
        """
        Initialize the market data utility.
        
        Args:
            mt5_connector: MT5Connector instance for data access.
        """
        self.mt5_connector = mt5_connector
        
        # Cache for historical data
        self._data_cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(minutes=5)  # Cache duration
    
    def get_historical_data(self, symbol, timeframe, count=500, use_cache=True):
        """
        Get historical price data with optional caching.
        
        Args:
            symbol (str): Trading symbol.
            timeframe (str): Timeframe (e.g., 'M1', 'H1', 'D1').
            count (int): Number of bars to retrieve.
            use_cache (bool): Whether to use cached data if available.
            
        Returns:
            DataFrame: Historical price data or None if failed.
        """
        cache_key = f"{symbol}_{timeframe}_{count}"
        
        # Check cache if enabled
        if use_cache and cache_key in self._data_cache:
            cache_time = self._cache_expiry.get(cache_key)
            if cache_time and datetime.now() < cache_time:
                logger.debug(f"Using cached data for {cache_key}")
                return self._data_cache[cache_key]
        
        # Get data from MT5
        data = self.mt5_connector.get_historical_data(symbol, timeframe, count)
        
        # Cache the data
        if data is not None and use_cache:
            self._data_cache[cache_key] = data
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
            logger.debug(f"Cached data for {cache_key}")
        
        return data
    
    def clear_cache(self, symbol=None, timeframe=None):
        """
        Clear the data cache, optionally filtered by symbol and timeframe.
        
        Args:
            symbol (str, optional): Symbol to clear cache for.
            timeframe (str, optional): Timeframe to clear cache for.
            
        Returns:
            int: Number of cache entries cleared.
        """
        if symbol is None and timeframe is None:
            # Clear all cache
            count = len(self._data_cache)
            self._data_cache.clear()
            self._cache_expiry.clear()
            return count
        
        # Selective clearing
        keys_to_remove = []
        for key in list(self._data_cache.keys()):
            parts = key.split('_')
            if len(parts) >= 2:
                key_symbol, key_timeframe = parts[0], parts[1]
                if (symbol is None or key_symbol == symbol) and (timeframe is None or key_timeframe == timeframe):
                    keys_to_remove.append(key)
        
        # Remove matched keys
        for key in keys_to_remove:
            del self._data_cache[key]
            if key in self._cache_expiry:
                del self._cache_expiry[key]
                
        return len(keys_to_remove)
    
    def calculate_indicators(self, data):
        """
        Calculate technical indicators for the given data.
        
        Args:
            data (DataFrame): Historical price data.
            
        Returns:
            dict: Calculated indicators.
        """
        if data is None or len(data) < 20:
            logger.warning("Insufficient data for calculating indicators")
            return {}
        
        indicators = {}
        
        # Moving Averages
        for period in [20, 50, 200]:
            indicators[f'ma{period}'] = self._calculate_sma(data['close'], period)
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(data['close'], 14)
        
        # MACD
        macd, signal = self._calculate_macd(data['close'], 12, 26, 9)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = macd - signal
        
        # Bollinger Bands
        middle, upper, lower = self._calculate_bollinger_bands(data['close'], 20, 2.0)
        indicators['bb_middle'] = middle
        indicators['bb_upper'] = upper
        indicators['bb_lower'] = lower
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(data, 14)
        
        return indicators
    
    def find_support_resistance_levels(self, data, window=50, threshold=0.01):
        """
        Find support and resistance levels in the historical data.
        
        Args:
            data (DataFrame): Historical price data.
            window (int): Window size for finding pivots.
            threshold (float): Threshold for grouping similar levels.
            
        Returns:
            dict: Support and resistance levels.
        """
        if data is None or len(data) < window * 2:
            return {'support': [], 'resistance': []}
        
        # Use recent data only
        recent_data = data.iloc[-min(len(data), window*3):]
        
        # Find pivot highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(recent_data) - window):
            is_pivot_high = True
            is_pivot_low = True
            
            current_high = recent_data['high'].iloc[i]
            current_low = recent_data['low'].iloc[i]
            
            # Check if it's a pivot high
            for j in range(i - window, i + window + 1):
                if j == i:
                    continue
                if recent_data['high'].iloc[j] > current_high:
                    is_pivot_high = False
                    break
            
            # Check if it's a pivot low
            for j in range(i - window, i + window + 1):
                if j == i:
                    continue
                if recent_data['low'].iloc[j] < current_low:
                    is_pivot_low = False
                    break
            
            if is_pivot_high:
                pivot_highs.append(current_high)
            
            if is_pivot_low:
                pivot_lows.append(current_low)
        
        # Group similar levels
        resistance_levels = self._group_levels(pivot_highs, threshold)
        support_levels = self._group_levels(pivot_lows, threshold)
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels)
        }
    
    def _group_levels(self, levels, threshold):
        """
        Group similar price levels.
        
        Args:
            levels (list): List of price levels.
            threshold (float): Threshold for grouping.
            
        Returns:
            list: Grouped price levels.
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Initialize groups
        grouped = []
        current_group = [sorted_levels[0]]
        
        # Group similar levels
        for i in range(1, len(sorted_levels)):
            current = sorted_levels[i]
            prev_avg = sum(current_group) / len(current_group)
            
            # Check if current level is similar to previous group
            if abs(current - prev_avg) / prev_avg <= threshold:
                current_group.append(current)
            else:
                # Add average of current group and start new group
                grouped.append(sum(current_group) / len(current_group))
                current_group = [current]
        
        # Add the last group
        if current_group:
            grouped.append(sum(current_group) / len(current_group))
        
        return grouped
    
    def _calculate_sma(self, data, period):
        """
        Calculate Simple Moving Average.
        
        Args:
            data (Series): Price data.
            period (int): MA period.
            
        Returns:
            Series: SMA values.
        """
        return data.rolling(window=period).mean().values
    
    def _calculate_rsi(self, data, period):
        """
        Calculate Relative Strength Index.
        
        Args:
            data (Series): Price data.
            period (int): RSI period.
            
        Returns:
            Series: RSI values.
        """
        delta = data.diff().dropna()
        
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        down = down.abs()
        
        roll_up = up.rolling(window=period).mean()
        roll_down = down.rolling(window=period).mean()
        
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi.values
    
    def _calculate_macd(self, data, fast_period, slow_period, signal_period):
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            data (Series): Price data.
            fast_period (int): Fast EMA period.
            slow_period (int): Slow EMA period.
            signal_period (int): Signal EMA period.
            
        Returns:
            tuple: (MACD line, Signal line)
        """
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return macd_line.values, signal_line.values
    
    def _calculate_bollinger_bands(self, data, period, std_dev):
        """
        Calculate Bollinger Bands.
        
        Args:
            data (Series): Price data.
            period (int): SMA period.
            std_dev (float): Standard deviation multiplier.
            
        Returns:
            tuple: (Middle band, Upper band, Lower band)
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return middle_band.values, upper_band.values, lower_band.values
    
    def _calculate_atr(self, data, period):
        """
        Calculate Average True Range.
        
        Args:
            data (DataFrame): Price data with OHLC.
            period (int): ATR period.
            
        Returns:
            Series: ATR values.
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.values
    
    def generate_simulated_prices(self, symbol, timeframe, count=100, base_price=None, volatility=0.01):
        """
        Generate simulated price data when MT5 is not available.
        
        Args:
            symbol (str): Symbol to simulate.
            timeframe (str): Timeframe to simulate.
            count (int): Number of candles to generate.
            base_price (float, optional): Starting price. Defaults to random value.
            volatility (float, optional): Price volatility factor.
            
        Returns:
            DataFrame: Simulated price data.
        """
        if base_price is None:
            # Generate random base price based on symbol
            if 'BTC' in symbol or 'XBT' in symbol:
                base_price = 35000 + random.uniform(-5000, 5000)
            elif 'ETH' in symbol:
                base_price = 2500 + random.uniform(-300, 300)
            elif 'USD' in symbol or 'EUR' in symbol:
                base_price = random.uniform(0.8, 1.2)
            else:
                base_price = random.uniform(50, 200)
        
        # Adjust volatility based on symbol
        if 'BTC' in symbol or 'XBT' in symbol:
            volatility *= 10
        elif 'ETH' in symbol:
            volatility *= 5
        
        # Generate time series
        end_time = datetime.now()
        
        # Determine time step based on timeframe
        if timeframe == 'M1':
            time_step = timedelta(minutes=1)
        elif timeframe == 'M5':
            time_step = timedelta(minutes=5)
        elif timeframe == 'M15':
            time_step = timedelta(minutes=15)
        elif timeframe == 'M30':
            time_step = timedelta(minutes=30)
        elif timeframe == 'H1':
            time_step = timedelta(hours=1)
        elif timeframe == 'H4':
            time_step = timedelta(hours=4)
        elif timeframe == 'D1':
            time_step = timedelta(days=1)
        elif timeframe == 'W1':
            time_step = timedelta(weeks=1)
        elif timeframe == 'MN1':
            time_step = timedelta(days=30)
        else:
            time_step = timedelta(minutes=5)
        
        # Generate timestamps
        timestamps = [end_time - i * time_step for i in range(count)]
        timestamps.reverse()  # Earliest first
        
        # Generate prices using random walk
        prices = [base_price]
        for i in range(1, count):
            # Random price change with momentum
            change = random.normalvariate(0, 1) * volatility * prices[-1]
            new_price = prices[-1] + change
            prices.append(max(0.001, new_price))  # Ensure positive price
        
        # Generate OHLC data
        data = []
        for i in range(count):
            price = prices[i]
            high = price * (1 + random.uniform(0, volatility))
            low = price * (1 - random.uniform(0, volatility))
            
            # Ensure high >= open/close >= low
            high = max(high, price)
            low = min(low, price)
            
            # Generate open price
            if i == 0:
                open_price = price * (1 + random.uniform(-volatility, volatility))
            else:
                # Open near previous close
                open_price = prices[i-1] * (1 + random.uniform(-volatility/2, volatility/2))
            
            # Ensure proper ordering
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            volume = random.randint(100, 1000)
            
            data.append({
                'time': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        
        return df
    
    def generate_simulated_tick(self, symbol, base_price=None, spread_pips=10):
        """
        Generate a simulated tick when MT5 is not available.
        
        Args:
            symbol (str): Symbol to simulate.
            base_price (float, optional): Base price. Defaults to random value.
            spread_pips (int, optional): Spread in pips.
            
        Returns:
            dict: Simulated tick data.
        """
        if base_price is None:
            # Use recent close price if available
            cache_key = f"{symbol}_M1_10"
            if cache_key in self._data_cache:
                data = self._data_cache[cache_key]
                base_price = data['close'].iloc[-1]
            else:
                # Generate random base price
                if 'BTC' in symbol or 'XBT' in symbol:
                    base_price = 35000 + random.uniform(-50, 50)
                elif 'ETH' in symbol:
                    base_price = 2500 + random.uniform(-20, 20)
                elif 'USD' in symbol or 'EUR' in symbol:
                    base_price = random.uniform(0.8, 1.2)
                else:
                    base_price = random.uniform(50, 200)
        
        # Calculate spread
        pip_value = 0.0001 if ('USD' in symbol or 'EUR' in symbol) and 'JPY' not in symbol else 0.01
        spread = spread_pips * pip_value
        
        bid = base_price
        ask = bid + spread
        
        # Current time
        time = datetime.now()
        
        # Create tick data
        tick = {
            'time': time,
            'bid': bid,
            'ask': ask,
            'last': bid,
            'volume': random.randint(1, 100),
            'time_msc': int(time.timestamp() * 1000),
            'flags': 0,
            'volume_real': random.random() * 10
        }
        
        return tick
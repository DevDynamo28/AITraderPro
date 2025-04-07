import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime

class MT5APIConnector:
    """
    A class to handle the connection to a MetaTrader 5 API server.
    This allows real MT5 data access without requiring the MT5 package on this server.
    """
    
    def __init__(self, config):
        """
        Initialize the connector with the given configuration.
        
        Args:
            config (dict): The configuration parameters
        """
        self.config = config
        self.initialized = False
        self.logger = logging.getLogger('MT5APIConnector')
        self.simulation_mode = False
        
        # Check environment variables for MT5 credentials
        self.mt5_login = os.environ.get('MT5_LOGIN') or self.config.get('login')
        self.mt5_password = os.environ.get('MT5_PASSWORD') or self.config.get('password')
        self.mt5_server = os.environ.get('MT5_SERVER') or self.config.get('server')
        
        # API Configuration
        self.api_url = self.config.get('api_url', 'https://mt5api.example.com')
        self.api_port = self.config.get('api_port', 443)
        self.api_key = self.config.get('api_key')
        
        # Convert login to int if it's a string containing only digits
        if isinstance(self.mt5_login, str) and self.mt5_login.isdigit():
            self.mt5_login = int(self.mt5_login)
    
    def initialize(self):
        """
        Initialize the connection to the MT5 API server.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        if self.initialized:
            return True
        
        try:
            # Test API connection
            response = self.make_api_call('/ping', {})
            
            if response and response.get('status') == 'success':
                self.initialized = True
                self.logger.info("MT5 API connection initialized successfully.")
                return True
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response from API'
                self.logger.error(f"MT5 API initialization failed: {error_msg}")
                
                # Fall back to simulation mode on error if enabled
                if self.config.get('simulation_fallback', True):
                    self.logger.info("Falling back to simulation mode due to API error")
                    self.simulation_mode = True
                    self.initialized = True
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing MT5 API: {str(e)}")
            
            # Fall back to simulation mode on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.info("Falling back to simulation mode due to error")
                self.simulation_mode = True
                self.initialized = True
                return True
                
            return False
    
    def make_api_call(self, endpoint, params):
        """
        Make an API call to the MT5 API server.
        
        Args:
            endpoint (str): The API endpoint.
            params (dict): Parameters for the API call.
            
        Returns:
            dict: The API response or None if failed.
        """
        try:
            # Add authentication parameters
            auth_params = {
                'login': self.mt5_login,
                'password': self.mt5_password,
                'server': self.mt5_server,
                'api_key': self.api_key
            }
            
            # Merge with endpoint-specific parameters
            request_params = {**auth_params, **params}
            
            # Make the API call
            url = f"{self.api_url}:{self.api_port}{endpoint}"
            response = requests.post(url, json=request_params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API call failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error making API call: {str(e)}")
            return None
    
    def get_account_info(self):
        """
        Get the account information.
        
        Returns:
            dict: The account information or None if failed.
        """
        # If in simulation mode, return simulated data
        if self.simulation_mode:
            return self._get_simulated_account_info()
        
        # Ensure API is initialized
        if not self.initialized and not self.initialize():
            return None
        
        try:
            response = self.make_api_call('/account_info', {})
            
            if response and response.get('status') == 'success':
                account_data = response.get('data', {})
                account_data['simulated'] = False  # Flag to indicate this is real data
                return account_data
            else:
                self.logger.error(f"Failed to get account info from API")
                
                # Fall back to simulation if enabled
                if self.config.get('simulation_fallback', True):
                    self.logger.warning("Falling back to simulation mode for account info")
                    self.simulation_mode = True
                    return self._get_simulated_account_info()
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting account info from API: {str(e)}")
            
            # Fall back to simulation on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning("Error getting account info, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_account_info()
            
            return None
    
    def get_symbol_info(self, symbol):
        """
        Get information about a specific symbol.
        
        Args:
            symbol (str): The symbol to get information for.
            
        Returns:
            dict: The symbol information or None if failed.
        """
        # If in simulation mode, return simulated data
        if self.simulation_mode:
            return self._get_simulated_symbol_info(symbol)
        
        # Ensure API is initialized
        if not self.initialized and not self.initialize():
            return None
        
        try:
            response = self.make_api_call('/symbol_info', {'symbol': symbol})
            
            if response and response.get('status') == 'success':
                symbol_data = response.get('data', {})
                symbol_data['simulated'] = False  # Flag to indicate this is real data
                return symbol_data
            else:
                self.logger.error(f"Failed to get symbol info for {symbol} from API")
                
                # Fall back to simulation if enabled
                if self.config.get('simulation_fallback', True):
                    self.logger.warning(f"Falling back to simulation mode for {symbol}")
                    self.simulation_mode = True
                    return self._get_simulated_symbol_info(symbol)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol} from API: {str(e)}")
            
            # Fall back to simulation on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning(f"Error getting symbol info for {symbol}, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_symbol_info(symbol)
            
            return None
    
    def get_historical_data(self, symbol, timeframe, count=100, start_pos=0):
        """
        Get historical price data for a symbol.
        
        Args:
            symbol (str): The symbol to get data for.
            timeframe (str): The timeframe to get data for (e.g., "M5" for 5-minute).
            count (int): The number of bars to get.
            start_pos (int): The position to start from.
            
        Returns:
            DataFrame: The historical data as a pandas DataFrame or None if failed.
        """
        # If in simulation mode, return simulated data
        if self.simulation_mode:
            return self._get_simulated_historical_data(symbol, timeframe, count, start_pos)
        
        # Ensure API is initialized
        if not self.initialized and not self.initialize():
            return None
        
        try:
            response = self.make_api_call('/historical_data', {
                'symbol': symbol,
                'timeframe': timeframe,
                'count': count,
                'start_pos': start_pos
            })
            
            if response and response.get('status') == 'success':
                # Convert API response to DataFrame
                data = response.get('data', [])
                if data:
                    df = pd.DataFrame(data)
                    # Make sure we have all the required columns
                    for col in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
                        if col not in df.columns:
                            if col == 'time':
                                df[col] = [datetime.now() for _ in range(len(df))]
                            elif col in ['open', 'high', 'low', 'close']:
                                df[col] = df.get(df.columns[0], 0)
                            else:
                                df[col] = 0
                    
                    # Add simulated flag (this is real data)
                    df['simulated'] = False
                    return df
                else:
                    self.logger.error(f"Empty historical data returned for {symbol}")
                    
                    # Fall back to simulation if enabled
                    if self.config.get('simulation_fallback', True):
                        self.logger.warning(f"Falling back to simulation mode for historical data of {symbol}")
                        self.simulation_mode = True
                        return self._get_simulated_historical_data(symbol, timeframe, count, start_pos)
                    
                    return None
            else:
                self.logger.error(f"Failed to get historical data for {symbol} from API")
                
                # Fall back to simulation if enabled
                if self.config.get('simulation_fallback', True):
                    self.logger.warning(f"Falling back to simulation mode for historical data of {symbol}")
                    self.simulation_mode = True
                    return self._get_simulated_historical_data(symbol, timeframe, count, start_pos)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol} from API: {str(e)}")
            
            # Fall back to simulation on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning(f"Error getting historical data for {symbol}, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_historical_data(symbol, timeframe, count, start_pos)
            
            return None
    
    def place_order(self, symbol, order_type, volume, price=0.0, sl=0.0, tp=0.0, comment=""):
        """
        Place an order with MT5.
        
        Args:
            symbol (str): The symbol to trade.
            order_type (str): The order type ("BUY" or "SELL").
            volume (float): The volume to trade.
            price (float): The price to trade at (0 for market orders).
            sl (float): The stop loss price.
            tp (float): The take profit price.
            comment (str): A comment for the order.
            
        Returns:
            dict: The order result information or None if failed.
        """
        # If in simulation mode, return simulated data
        if self.simulation_mode:
            return self._get_simulated_order_result(symbol, order_type, volume, price, sl, tp, comment)
        
        # Ensure API is initialized
        if not self.initialized and not self.initialize():
            return None
        
        try:
            response = self.make_api_call('/place_order', {
                'symbol': symbol,
                'order_type': order_type,
                'volume': volume,
                'price': price,
                'sl': sl,
                'tp': tp,
                'comment': comment
            })
            
            if response and response.get('status') == 'success':
                order_data = response.get('data', {})
                order_data['simulated'] = False  # Flag to indicate this is real data
                return order_data
            else:
                self.logger.error(f"Failed to place order for {symbol} through API")
                
                # Fall back to simulation if enabled
                if self.config.get('simulation_fallback', True):
                    self.logger.warning(f"Falling back to simulation mode for placing order on {symbol}")
                    self.simulation_mode = True
                    return self._get_simulated_order_result(symbol, order_type, volume, price, sl, tp, comment)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol} through API: {str(e)}")
            
            # Fall back to simulation on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning(f"Error placing order for {symbol}, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_order_result(symbol, order_type, volume, price, sl, tp, comment)
            
            return None
    
    def get_positions(self, symbol=None):
        """
        Get all open positions, optionally filtered by symbol.
        
        Args:
            symbol (str, optional): The symbol to filter by. Defaults to None.
            
        Returns:
            list: The positions as a list or empty list if none.
        """
        # If in simulation mode, return simulated data
        if self.simulation_mode:
            return self._get_simulated_positions(symbol)
        
        # Ensure API is initialized
        if not self.initialized and not self.initialize():
            return []
        
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
                
            response = self.make_api_call('/get_positions', params)
            
            if response and response.get('status') == 'success':
                positions = response.get('data', [])
                # Add simulated flag (this is real data)
                for pos in positions:
                    pos['simulated'] = False
                return positions
            else:
                self.logger.error(f"Failed to get positions from API")
                
                # Fall back to simulation if enabled
                if self.config.get('simulation_fallback', True):
                    self.logger.warning("Falling back to simulation mode for positions")
                    self.simulation_mode = True
                    return self._get_simulated_positions(symbol)
                
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting positions from API: {str(e)}")
            
            # Fall back to simulation on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning("Error getting positions, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_positions(symbol)
            
            return []
    
    def close_position(self, position_id):
        """
        Close a specific position.
        
        Args:
            position_id (int): The position ID to close.
            
        Returns:
            dict: The close result information or None if failed.
        """
        # If in simulation mode, return simulated data
        if self.simulation_mode:
            return self._get_simulated_close_result(position_id)
        
        # Ensure API is initialized
        if not self.initialized and not self.initialize():
            return None
        
        try:
            response = self.make_api_call('/close_position', {
                'position_id': position_id
            })
            
            if response and response.get('status') == 'success':
                close_data = response.get('data', {})
                close_data['simulated'] = False  # Flag to indicate this is real data
                return close_data
            else:
                self.logger.error(f"Failed to close position {position_id} through API")
                
                # Fall back to simulation if enabled
                if self.config.get('simulation_fallback', True):
                    self.logger.warning(f"Falling back to simulation mode for closing position {position_id}")
                    self.simulation_mode = True
                    return self._get_simulated_close_result(position_id)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error closing position {position_id} through API: {str(e)}")
            
            # Fall back to simulation on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning(f"Error closing position {position_id}, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_close_result(position_id)
            
            return None
            
    # Simulated data methods for fallback
    def _get_simulated_account_info(self):
        """
        Get simulated account information.
        
        Returns:
            dict: Simulated account information
        """
        return {
            'balance': 10000.0,
            'equity': 10000.0,
            'profit': 0.0,
            'margin': 0.0,
            'margin_free': 10000.0,
            'margin_level': 100.0,
            'leverage': 100,
            'currency': 'USD',
            'simulated': True  # Flag to indicate this is simulated data
        }
    
    def _get_simulated_symbol_info(self, symbol):
        """
        Get simulated symbol information for a specific symbol.
        
        Args:
            symbol (str): The symbol to get information for.
            
        Returns:
            dict: Simulated symbol information
        """
        # Default simulated values
        simulated_data = {
            'bid': 60000.0,
            'ask': 60050.0,
            'point': 0.01,
            'digits': 2,
            'spread': 50,
            'volume_min': 0.01,
            'volume_max': 10.0,
            'volume_step': 0.01,
            'trade_contract_size': 1.0,
            'trade_tick_value': 1.0,
            'trade_tick_size': 0.01,
            'simulated': True  # Flag to indicate this is simulated data
        }
        
        # Customize for different symbols
        if symbol == "EURUSD":
            simulated_data.update({
                'bid': 1.0890,
                'ask': 1.0892,
                'digits': 5,
                'spread': 2,
                'point': 0.00001
            })
        elif symbol == "ETHUSD":
            simulated_data.update({
                'bid': 3500.0,
                'ask': 3505.0,
                'digits': 2,
                'spread': 50
            })
        elif symbol == "XRPUSD":
            simulated_data.update({
                'bid': 0.5245,
                'ask': 0.5250,
                'digits': 4,
                'spread': 5,
                'point': 0.0001
            })
        
        return simulated_data
    
    def _get_simulated_historical_data(self, symbol, timeframe, count=100, start_pos=0):
        """
        Get simulated historical price data for a symbol.
        
        Args:
            symbol (str): The symbol to get data for.
            timeframe (str): The timeframe to get data for (e.g., "M5" for 5-minute).
            count (int): The number of bars to get.
            start_pos (int): The position to start from.
            
        Returns:
            DataFrame: Simulated historical data as a pandas DataFrame
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create realistic time intervals based on timeframe
        current_time = datetime.now()
        times = []
        
        # Determine time delta based on timeframe
        if timeframe == 'M1':
            delta = timedelta(minutes=1)
        elif timeframe == 'M5':
            delta = timedelta(minutes=5)
        elif timeframe == 'M15':
            delta = timedelta(minutes=15)
        elif timeframe == 'M30':
            delta = timedelta(minutes=30)
        elif timeframe == 'H1':
            delta = timedelta(hours=1)
        elif timeframe == 'H4':
            delta = timedelta(hours=4)
        elif timeframe == 'D1':
            delta = timedelta(days=1)
        elif timeframe == 'W1':
            delta = timedelta(weeks=1)
        elif timeframe == 'MN1':
            delta = timedelta(days=30)  # Approximate month
        else:
            delta = timedelta(minutes=5)  # Default to M5
        
        # Generate timestamps
        for i in range(count):
            # We need to go back in time, so the latest data is the most recent
            # start_pos is how many bars to skip from current time
            times.append(current_time - delta * (i + start_pos))
        
        # Sort times in ascending order (oldest first)
        times.sort()
        
        # Get base price based on symbol
        if symbol == "EURUSD":
            base_price = 1.0890
            volatility = 0.0010
        elif symbol == "ETHUSD":
            base_price = 3500.0
            volatility = 50.0
        elif symbol == "XRPUSD":
            base_price = 0.5245
            volatility = 0.01
        else:  # Default BTCUSD
            base_price = 60000.0
            volatility = 500.0
        
        # Generate price data with realistic trends
        opens = []
        highs = []
        lows = []
        closes = []
        tick_volumes = []
        
        # Create a slight upward/downward trend with random wobble
        trend = np.random.choice([-1, 1]) * 0.0001 * base_price  # Small trend direction
        current_price = base_price
        
        for i in range(count):
            # Add some randomness to current price
            price_change = np.random.normal(trend, volatility)
            open_price = current_price
            
            # High is usually the larger of the random movements from open
            high_rand = abs(np.random.normal(0, volatility))
            # Low is usually the smaller of the random movements from open
            low_rand = -abs(np.random.normal(0, volatility))
            
            high_price = open_price + high_rand
            low_price = open_price + low_rand
            
            # Ensure high is always highest and low is always lowest
            if high_price < open_price:
                high_price = open_price + abs(high_rand / 2)
            if low_price > open_price:
                low_price = open_price - abs(low_rand / 2)
            
            # Close can be anywhere between high and low
            close_price = open_price + np.random.uniform(low_rand, high_rand)
            
            # Enforce strict high/low boundaries
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Update current price for the next iteration
            current_price = close_price
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            tick_volumes.append(int(np.random.uniform(100, 1000)))
        
        # Create DataFrame
        data = {
            'time': times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'tick_volume': tick_volumes,
            'spread': [int(volatility * 100) for _ in range(count)],
            'real_volume': tick_volumes,
            'simulated': [True for _ in range(count)]  # Mark as simulated data
        }
        
        df = pd.DataFrame(data)
        
        return df
    
    def _get_simulated_order_result(self, symbol, order_type, volume, price, sl, tp, comment):
        """
        Get simulated order result.
        
        Args:
            symbol (str): The symbol to trade.
            order_type (str): The order type ("BUY" or "SELL").
            volume (float): The volume to trade.
            price (float): The price to trade at.
            sl (float): The stop loss price.
            tp (float): The take profit price.
            comment (str): A comment for the order.
            
        Returns:
            dict: Simulated order result
        """
        import random
        return {
            'order': random.randint(10000, 99999),
            'volume': volume,
            'price': price if price > 0 else self._get_simulated_symbol_info(symbol)['ask'],
            'symbol': symbol,
            'type': order_type,
            'sl': sl,
            'tp': tp,
            'comment': comment,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'retcode': 10009,  # MT5 success code
            'simulated': True  # Flag to indicate this is simulated data
        }
    
    def _get_simulated_positions(self, symbol=None):
        """
        Get simulated positions.
        
        Args:
            symbol (str, optional): The symbol to filter by. Defaults to None.
            
        Returns:
            list: Simulated positions
        """
        # In simulation mode, return an empty list (no open positions)
        return []
    
    def _get_simulated_close_result(self, position_id):
        """
        Get simulated close position result.
        
        Args:
            position_id (int): The position ID to close.
            
        Returns:
            dict: Simulated close result
        """
        return {
            'position_id': position_id,
            'success': True,
            'message': 'Position closed successfully (simulation)',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'simulated': True  # Flag to indicate this is simulated data
        }
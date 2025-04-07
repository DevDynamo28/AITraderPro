import os
import time
import logging
import pandas as pd
from datetime import datetime

# Try to import MetaTrader5, but handle if it's not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    # Create mock constants for simulation mode
    class MT5Constants:
        TIMEFRAME_M1 = "M1"
        TIMEFRAME_M5 = "M5"
        TIMEFRAME_M15 = "M15"
        TIMEFRAME_M30 = "M30"
        TIMEFRAME_H1 = "H1"
        TIMEFRAME_H4 = "H4"
        TIMEFRAME_D1 = "D1"
        TIMEFRAME_W1 = "W1"
        TIMEFRAME_MN1 = "MN1"
        
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        ORDER_TYPE_BUY_LIMIT = 2
        ORDER_TYPE_SELL_LIMIT = 3
        ORDER_TYPE_BUY_STOP = 4
        ORDER_TYPE_SELL_STOP = 5
        
        TRADE_ACTION_DEAL = 1
        ORDER_TIME_GTC = 1
        ORDER_FILLING_IOC = 1
        
        TRADE_RETCODE_DONE = 10009
        
        POSITION_TYPE_BUY = 0
    
    # Create a complete mock class for MT5
    class MockMT5:
        def __init__(self):
            # Constants
            self.TIMEFRAME_M1 = MT5Constants.TIMEFRAME_M1
            self.TIMEFRAME_M5 = MT5Constants.TIMEFRAME_M5
            self.TIMEFRAME_M15 = MT5Constants.TIMEFRAME_M15
            self.TIMEFRAME_M30 = MT5Constants.TIMEFRAME_M30
            self.TIMEFRAME_H1 = MT5Constants.TIMEFRAME_H1
            self.TIMEFRAME_H4 = MT5Constants.TIMEFRAME_H4
            self.TIMEFRAME_D1 = MT5Constants.TIMEFRAME_D1
            self.TIMEFRAME_W1 = MT5Constants.TIMEFRAME_W1
            self.TIMEFRAME_MN1 = MT5Constants.TIMEFRAME_MN1
            
            self.ORDER_TYPE_BUY = MT5Constants.ORDER_TYPE_BUY
            self.ORDER_TYPE_SELL = MT5Constants.ORDER_TYPE_SELL
            self.ORDER_TYPE_BUY_LIMIT = MT5Constants.ORDER_TYPE_BUY_LIMIT
            self.ORDER_TYPE_SELL_LIMIT = MT5Constants.ORDER_TYPE_SELL_LIMIT
            self.ORDER_TYPE_BUY_STOP = MT5Constants.ORDER_TYPE_BUY_STOP
            self.ORDER_TYPE_SELL_STOP = MT5Constants.ORDER_TYPE_SELL_STOP
            
            self.TRADE_ACTION_DEAL = MT5Constants.TRADE_ACTION_DEAL
            self.ORDER_TIME_GTC = MT5Constants.ORDER_TIME_GTC
            self.ORDER_FILLING_IOC = MT5Constants.ORDER_FILLING_IOC
            self.TRADE_RETCODE_DONE = MT5Constants.TRADE_RETCODE_DONE
            self.POSITION_TYPE_BUY = MT5Constants.POSITION_TYPE_BUY
            
            # Mock data
            from collections import namedtuple
            self.AccountInfo = namedtuple('AccountInfo', ['balance', 'equity', 'profit', 'margin', 'margin_free', 'margin_level', 'leverage'])
            self.SymbolInfo = namedtuple('SymbolInfo', ['bid', 'ask', 'point', 'digits', 'spread', 'volume_min', 'volume_max', 'volume_step', 'trade_contract_size', 'trade_tick_value', 'trade_tick_size'])
            self.OrderSendResult = namedtuple('OrderSendResult', ['order', 'volume', 'price', 'retcode', 'comment'])
            
        def initialize(self):
            return True
            
        def shutdown(self):
            return True
            
        def login(self, login=None, password=None, server=None):
            return True
            
        def last_error(self):
            return (0, "No error")
            
        def terminal_info(self):
            return True
            
        def account_info(self):
            # Return mock account info
            return self.AccountInfo(
                balance=10000.0,
                equity=10000.0,
                profit=0.0,
                margin=0.0,
                margin_free=10000.0,
                margin_level=100.0,
                leverage=100
            )
            
        def symbol_info(self, symbol):
            # Return mock symbol info
            if symbol == "BTCUSD":
                return self.SymbolInfo(
                    bid=60000.0,
                    ask=60050.0,
                    point=0.01,
                    digits=2,
                    spread=50,
                    volume_min=0.01,
                    volume_max=10.0,
                    volume_step=0.01,
                    trade_contract_size=1.0,
                    trade_tick_value=1.0,
                    trade_tick_size=0.01
                )
            return None
            
        def symbol_info_tick(self, symbol):
            # Return the same as symbol_info but as a tick
            return self.symbol_info(symbol)
            
        def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
            # Generate some mock price data
            import numpy as np
            from datetime import datetime, timedelta
            
            if symbol != "BTCUSD":
                return None
                
            current_time = datetime.now()
            times = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            base_price = 60000.0
            volatility = 500.0
            
            # Generate random price data
            for i in range(count):
                time_offset = i
                if timeframe == self.TIMEFRAME_M1:
                    dt = current_time - timedelta(minutes=time_offset)
                elif timeframe == self.TIMEFRAME_M5:
                    dt = current_time - timedelta(minutes=5*time_offset)
                elif timeframe == self.TIMEFRAME_M15:
                    dt = current_time - timedelta(minutes=15*time_offset)
                elif timeframe == self.TIMEFRAME_M30:
                    dt = current_time - timedelta(minutes=30*time_offset)
                elif timeframe == self.TIMEFRAME_H1:
                    dt = current_time - timedelta(hours=time_offset)
                elif timeframe == self.TIMEFRAME_H4:
                    dt = current_time - timedelta(hours=4*time_offset)
                elif timeframe == self.TIMEFRAME_D1:
                    dt = current_time - timedelta(days=time_offset)
                else:
                    dt = current_time - timedelta(minutes=5*time_offset)
                
                times.append(int(dt.timestamp()))
                
                # Generate random price movement
                price_change = np.random.normal(0, volatility)
                open_price = base_price + price_change
                high_price = open_price + abs(np.random.normal(0, volatility/2))
                low_price = open_price - abs(np.random.normal(0, volatility/2))
                close_price = (open_price + high_price + low_price) / 3 + np.random.normal(0, volatility/4)
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(np.random.randint(100, 1000))
            
            # Create structured array for MT5 compatibility
            data = np.zeros(count, dtype=[
                ('time', np.int64),
                ('open', np.float64),
                ('high', np.float64),
                ('low', np.float64),
                ('close', np.float64),
                ('tick_volume', np.int64),
                ('spread', np.int32),
                ('real_volume', np.int64)
            ])
            
            data['time'] = times
            data['open'] = opens
            data['high'] = highs
            data['low'] = lows
            data['close'] = closes
            data['tick_volume'] = volumes
            data['spread'] = 5
            data['real_volume'] = volumes
            
            return data
            
        def order_send(self, request):
            # Mock successful order send
            return self.OrderSendResult(
                order=12345,
                volume=request['volume'],
                price=request['price'],
                retcode=self.TRADE_RETCODE_DONE,
                comment="Simulation Order"
            )
            
        def positions_get(self, symbol=None, ticket=None):
            # In simulation mode, return an empty list for positions_get
            # If a specific ticket is requested, return None
            if ticket is not None:
                return None
                
            # Create a Position namedtuple for simulation purposes
            from collections import namedtuple
            import time
            Position = namedtuple('Position', [
                'ticket', 'time', 'type', 'volume', 'price_open', 'price_current', 
                'sl', 'tp', 'profit', 'symbol', 'comment'
            ])
            
            # Return an empty list (no positions by default in simulation)
            return []
    
    # Create mock MT5 module
    mt5 = MockMT5()

class MT5Connector:
    """
    A class to handle the connection and interaction with MetaTrader 5 platform.
    """
    
    def __init__(self, config):
        """
        Initialize the connector with the given configuration.
        
        Args:
            config (dict): The configuration parameters
        """
        self.config = config
        self.initialized = False
        self.logger = logging.getLogger('MT5Connector')
        
        # Check environment variables for MT5 credentials
        mt5_login = os.environ.get('MT5_LOGIN') or self.config.get('login')
        # Convert login to int if it's a string containing only digits
        if isinstance(mt5_login, str) and mt5_login.isdigit():
            mt5_login = int(mt5_login)
        self.config['login'] = mt5_login
        self.config['password'] = os.environ.get('MT5_PASSWORD') or self.config.get('password')
        self.config['server'] = os.environ.get('MT5_SERVER') or self.config.get('server')
        
        # Check if simulation mode is explicitly set
        self.simulation_mode = self.config.get('simulation', False) or not MT5_AVAILABLE
    
    def initialize(self):
        """
        Initialize the connection to the MT5 platform.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        if self.initialized:
            return True
        
        # Check if we're in simulation mode (by config or MT5 not available)
        if self.simulation_mode:
            self.logger.info("Running in simulation mode (as configured in settings)")
            self.initialized = True
            return True
            
        # Check if MT5 is available
        if not MT5_AVAILABLE:
            self.logger.warning("Running in simulation mode - MetaTrader5 package not available")
            self.initialized = True
            return True
            
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.error(f"MT5 initialization failed with error code {error[0]}: {error[1]}")
                return False
            
            # Attempt login if credentials are provided
            if self.config.get('login') and self.config.get('password'):
                login_result = mt5.login(
                    login=self.config.get('login'),
                    password=self.config.get('password'),
                    server=self.config.get('server', '')
                )
                
                if not login_result:
                    error = mt5.last_error()
                    self.logger.error(f"MT5 login failed with error code {error[0]}: {error[1]}")
                    mt5.shutdown()
                    
                    # Log more details about the login attempt (without showing the password)
                    self.logger.info(f"Login details: login={type(self.config.get('login'))}, server={self.config.get('server', '')}")
                    
                    # Continue in simulation mode after failed login if enabled
                    if self.config.get('simulation_fallback', True):
                        self.logger.info("Falling back to simulation mode due to login failure")
                        self.simulation_mode = True
                        self.initialized = True
                        return True
                    
                    return False
            
            self.initialized = True
            self.logger.info("MT5 connection initialized successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            
            # Fall back to simulation mode on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.info("Falling back to simulation mode due to error")
                self.simulation_mode = True
                self.initialized = True
                return True
                
            return False
    
    def shutdown(self):
        """
        Shutdown the connection to the MT5 platform.
        """
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            self.logger.info("MT5 connection shutdown.")
    
    def _get_simulated_account_info(self):
        """
        Get simulated account information when MT5 is not available.
        
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
    
    def get_account_info(self):
        """
        Get the account information.
        
        Returns:
            dict: The account information or None if failed.
        """
        # If in simulation mode, return simulated data
        if self.simulation_mode:
            return self._get_simulated_account_info()
        
        # Ensure MT5 is initialized
        if not self.ensure_initialized():
            return self._get_simulated_account_info() if self.simulation_mode else None
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get account info. Error: {error[0]} - {error[1]}")
                
                # If we get a "No IPC connection" error, switch to simulation mode
                if error[0] == -10004 and self.config.get('simulation_fallback', True):
                    self.logger.warning("MT5 IPC connection lost, switching to simulation mode")
                    self.simulation_mode = True
                    return self._get_simulated_account_info()
                
                return None
            
            # Convert MT5 account info to a dictionary
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.profit,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'leverage': account_info.leverage,
                'currency': getattr(account_info, 'currency', 'USD'),
                'simulated': False  # Flag to indicate this is real data
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            
            # Fall back to simulation mode on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning("Error getting account info, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_account_info()
            
            return None
    
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
        
        # Ensure MT5 is initialized
        if not self.ensure_initialized():
            return self._get_simulated_symbol_info(symbol) if self.simulation_mode else None
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get symbol info for {symbol}. Error: {error[0]} - {error[1]}")
                
                # If we get a "No IPC connection" error, switch to simulation mode
                if error[0] == -10004 and self.config.get('simulation_fallback', True):
                    self.logger.warning(f"MT5 IPC connection lost for {symbol}, switching to simulation mode")
                    self.simulation_mode = True
                    return self._get_simulated_symbol_info(symbol)
                
                return None
            
            # Convert MT5 symbol info to a dictionary
            return {
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_contract_size': symbol_info.trade_contract_size,
                'trade_tick_value': symbol_info.trade_tick_value,
                'trade_tick_size': symbol_info.trade_tick_size,
                'simulated': False  # Flag to indicate this is real data
            }
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            
            # Fall back to simulation mode on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.warning(f"Error getting symbol info for {symbol}, switching to simulation mode")
                self.simulation_mode = True
                return self._get_simulated_symbol_info(symbol)
            
            return None
    
    def _get_simulated_historical_data(self, symbol, timeframe, count=100, start_pos=0):
        """
        Get simulated historical price data for a symbol when MT5 is not available.
        
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
        
        # Ensure MT5 is initialized
        if not self.ensure_initialized():
            return self._get_simulated_historical_data(symbol, timeframe, count, start_pos) if self.simulation_mode else None
        
        try:
            # Map timeframe string to MT5 timeframe constants
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }
            
            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Get the historical data
            rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                self.logger.error(f"Failed to get historical data for {symbol}. Error: {error[0]} - {error[1]}")
                
                # If we get a "No IPC connection" error, switch to simulation mode
                if error[0] == -10004 and self.config.get('simulation_fallback', True):
                    self.logger.warning(f"MT5 IPC connection lost when getting historical data for {symbol}, switching to simulation mode")
                    self.simulation_mode = True
                    return self._get_simulated_historical_data(symbol, timeframe, count, start_pos)
                
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['simulated'] = False  # Mark as real data
            
            return df
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            
            # Fall back to simulation mode on exception if enabled
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
        if not self.ensure_initialized():
            return None
        
        # Map order type string to MT5 order type constants
        order_type_map = {
            "BUY": mt5.ORDER_TYPE_BUY,
            "SELL": mt5.ORDER_TYPE_SELL,
            "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
            "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
            "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
            "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
        }
        
        mt5_order_type = order_type_map.get(order_type)
        if mt5_order_type is None:
            self.logger.error(f"Invalid order type: {order_type}")
            return None
        
        # Get current price if not specified for market orders
        if price == 0.0 and mt5_order_type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return None
            
            if mt5_order_type == mt5.ORDER_TYPE_BUY:
                price = symbol_info['ask']
            else:
                price = symbol_info['bid']
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,  # Maximum deviation from requested price
            "magic": 12345,   # Magic number, unique identifier
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send the order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed with error code: {result.retcode}")
            return None
        
        # Convert order result to a dictionary
        return {
            'order_id': result.order,
            'volume': result.volume,
            'price': result.price,
            'retcode': result.retcode,
            'comment': result.comment,
        }
    
    def get_positions(self, symbol=None):
        """
        Get all open positions, optionally filtered by symbol.
        
        Args:
            symbol (str, optional): The symbol to filter by. Defaults to None.
            
        Returns:
            list: The positions as a list or empty list if none.
        """
        if not self.ensure_initialized():
            return []
        
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                # Check if there's an error or if there are just no positions
                if MT5_AVAILABLE:
                    error = mt5.last_error()
                    if error[0] != 0:
                        self.logger.error(f"Failed to get positions. Error: {error[0]} - {error[1]}")
                return []
            
            if len(positions) == 0:
                return []
            
            # Convert to a list of dictionaries for easier handling
            result = []
            for pos in positions:
                pos_dict = pos._asdict() if hasattr(pos, '_asdict') else vars(pos)
                # Convert time to datetime if it's an integer
                if 'time' in pos_dict and isinstance(pos_dict['time'], (int, float)):
                    pos_dict['time'] = datetime.fromtimestamp(pos_dict['time'])
                result.append(pos_dict)
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def close_position(self, position_id):
        """
        Close a specific position.
        
        Args:
            position_id (int): The position ID to close.
            
        Returns:
            dict: The close result information or None if failed.
        """
        if not self.ensure_initialized():
            return None
        
        try:
            positions = mt5.positions_get(ticket=position_id)
            if positions is None or len(positions) == 0:
                self.logger.error(f"Position with ID {position_id} not found.")
                
                # In simulation mode, return a mock successful close
                if not MT5_AVAILABLE:
                    self.logger.info(f"Simulation: Closing position {position_id}")
                    return {
                        'order_id': 12345,
                        'volume': 0.1,
                        'price': 60000.0,
                        'retcode': mt5.TRADE_RETCODE_DONE,
                        'comment': "Simulated position close",
                    }
                
                return None
            
            position = positions[0]
            
            # Get symbol info for price
            symbol_info = self.get_symbol_info(getattr(position, 'symbol', 'BTCUSD'))
            if symbol_info is None:
                if not MT5_AVAILABLE:
                    # Return mock data in simulation mode
                    price = 60000.0
                else:
                    self.logger.error(f"Failed to get symbol info for position close")
                    return None
            else:
                # Determine close operation based on position type
                if getattr(position, 'type', 0) == mt5.POSITION_TYPE_BUY:
                    price = symbol_info['bid']
                else:
                    price = symbol_info['ask']
            
            # Prepare the close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": getattr(position, 'symbol', 'BTCUSD'),
                "volume": getattr(position, 'volume', 0.1),
                "type": mt5.ORDER_TYPE_SELL if getattr(position, 'type', 0) == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position_id,
                "price": price,
                "deviation": 10,
                "magic": 12345,
                "comment": "Position close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send the close order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close position failed with error code: {result.retcode}")
                return None
            
            # Convert close result to a dictionary
            return {
                'order_id': result.order,
                'volume': result.volume,
                'price': result.price,
                'retcode': result.retcode,
                'comment': result.comment,
            }
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            # In simulation mode, return a mock successful close
            if not MT5_AVAILABLE:
                self.logger.info(f"Simulation: Closing position {position_id} after error")
                return {
                    'order_id': 12345,
                    'volume': 0.1,
                    'price': 60000.0,
                    'retcode': mt5.TRADE_RETCODE_DONE,
                    'comment': "Simulated position close after error",
                }
            return None
    
    def ensure_initialized(self):
        """
        Ensure that MT5 is initialized, try to initialize if not.
        
        Returns:
            bool: True if initialized, False otherwise.
        """
        if not self.initialized:
            return self.initialize()
        
        # In simulation mode, we're always initialized
        if self.simulation_mode or not MT5_AVAILABLE:
            return True
            
        # Check if MT5 is still running
        try:
            if not mt5.terminal_info():
                return self.initialize()
        except Exception as e:
            self.logger.error(f"Error checking MT5 terminal info: {str(e)}")
            
            # Fall back to simulation mode on exception if enabled
            if self.config.get('simulation_fallback', True):
                self.logger.info("Falling back to simulation mode due to terminal info error")
                self.simulation_mode = True
                return True
            
            return self.initialize()
        
        return True

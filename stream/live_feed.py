import os
import time
import json
import logging
import threading
import asyncio
import websockets
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Set

# Try to import MetaTrader5, but handle if it's not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

class LiveFeed:
    """
    Class to provide real-time price streaming for trading symbols.
    Supports both polling and WebSocket methods for data distribution.
    """
    
    def __init__(self, config, mt5_connector):
        """
        Initialize the live feed.
        
        Args:
            config (dict): Configuration for the live feed.
            mt5_connector: MT5Connector instance for data access.
        """
        self.config = config
        self.mt5 = mt5_connector
        self.logger = logging.getLogger('LiveFeed')
        
        # Stream settings
        self.update_interval = config.get('update_interval', 1.0)  # seconds
        self.symbols = config.get('symbols', ['BTCUSD'])
        self.enable_websocket = config.get('enable_websocket', True)
        self.websocket_port = config.get('websocket_port', 8765)
        
        # Check simulation mode
        self.simulation_mode = mt5_connector.simulation_mode if hasattr(mt5_connector, 'simulation_mode') else False
        if self.simulation_mode:
            self.logger.info("LiveFeed operating in simulation mode")
        
        # State variables
        self.running = False
        self.last_prices = {}
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.clients_lock = threading.Lock()
        
        # Callback handlers
        self.price_callbacks = []  # Functions to call when prices update
        
        # Simulation price variables
        self.last_sim_update = time.time()
        self.sim_price_base = {
            'BTCUSD': 60000.0,
            'ETHUSD': 4000.0,
            'XRPUSD': 1.0,
        }
        
        # Thread control
        self.poll_thread = None
        self.websocket_thread = None
    
    def start(self):
        """
        Start the live feed service.
        
        Returns:
            bool: True if started successfully, False otherwise.
        """
        if self.running:
            self.logger.warning("Live feed already running")
            return True
        
        try:
            # Start MT5 connection if not in simulation mode and not already initialized
            if not self.simulation_mode:
                if not self.mt5.ensure_initialized():
                    self.logger.warning("Failed to initialize MT5 connection, falling back to simulation mode")
                    self.simulation_mode = True
            else:
                self.logger.info("Starting live feed in simulation mode")
            
            # Start price polling thread
            self.running = True
            self.poll_thread = threading.Thread(target=self._price_polling_loop)
            self.poll_thread.daemon = True
            self.poll_thread.start()
            
            # Start WebSocket server if enabled
            if self.enable_websocket:
                self.websocket_thread = threading.Thread(target=self._start_websocket_server)
                self.websocket_thread.daemon = True
                self.websocket_thread.start()
            
            mode_str = "simulation mode" if self.simulation_mode else "live mode"
            self.logger.info(f"Live feed started with symbols: {self.symbols} in {mode_str}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting live feed: {str(e)}")
            self.running = False
            return False
    
    def stop(self):
        """
        Stop the live feed service.
        """
        if not self.running:
            return
        
        self.running = False
        
        # Wait for threads to stop
        if self.poll_thread:
            self.poll_thread.join(timeout=2.0)
        
        self.logger.info("Live feed stopped")
    
    def add_symbol(self, symbol):
        """
        Add a symbol to the live feed.
        
        Args:
            symbol (str): The symbol to add.
            
        Returns:
            bool: True if added, False if already present.
        """
        if symbol in self.symbols:
            return False
        
        self.symbols.append(symbol)
        self.logger.info(f"Added symbol to live feed: {symbol}")
        return True
    
    def remove_symbol(self, symbol):
        """
        Remove a symbol from the live feed.
        
        Args:
            symbol (str): The symbol to remove.
            
        Returns:
            bool: True if removed, False if not found.
        """
        if symbol not in self.symbols:
            return False
        
        self.symbols.remove(symbol)
        
        # Remove from last prices
        if symbol in self.last_prices:
            del self.last_prices[symbol]
        
        self.logger.info(f"Removed symbol from live feed: {symbol}")
        return True
    
    def get_last_price(self, symbol):
        """
        Get the last known price for a symbol.
        
        Args:
            symbol (str): The symbol to get price for.
            
        Returns:
            dict: Price information or None if not available.
        """
        return self.last_prices.get(symbol)
    
    def get_all_prices(self):
        """
        Get all last known prices.
        
        Returns:
            dict: Dictionary of all symbol prices.
        """
        return self.last_prices.copy()
    
    def register_price_callback(self, callback_fn):
        """
        Register a callback function to be called when prices update.
        
        Args:
            callback_fn: Function to call with price updates.
            
        Returns:
            bool: True if registered, False if already registered.
        """
        if callback_fn in self.price_callbacks:
            return False
        
        self.price_callbacks.append(callback_fn)
        return True
    
    def unregister_price_callback(self, callback_fn):
        """
        Unregister a price update callback function.
        
        Args:
            callback_fn: Function to unregister.
            
        Returns:
            bool: True if unregistered, False if not found.
        """
        if callback_fn not in self.price_callbacks:
            return False
        
        self.price_callbacks.remove(callback_fn)
        return True
    
    def _price_polling_loop(self):
        """
        Main loop for polling prices from MT5.
        """
        self.logger.info("Price polling thread started")
        
        while self.running:
            try:
                self._update_prices()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in price polling loop: {str(e)}")
                time.sleep(2.0)  # Longer delay on error
    
    def _update_prices(self):
        """
        Update prices for all symbols and notify callbacks.
        """
        for symbol in self.symbols:
            try:
                # Check if we're using simulation mode
                if self.simulation_mode:
                    # Generate simulated price data
                    price_data = self._generate_simulated_price(symbol)
                else:
                    # Get current symbol price from MT5
                    symbol_info = self.mt5.get_symbol_info(symbol)
                    
                    if symbol_info is None:
                        # If MT5 can't provide data, fallback to simulation for this symbol
                        price_data = self._generate_simulated_price(symbol)
                    else:
                        # Format the price data from MT5
                        price_data = {
                            'symbol': symbol,
                            'bid': symbol_info['bid'],
                            'ask': symbol_info['ask'],
                            'spread': symbol_info['spread'],
                            'time': datetime.now().isoformat()
                        }
                
                # Check if price has changed
                if symbol in self.last_prices:
                    old_price = self.last_prices[symbol]
                    if (old_price['bid'] == price_data['bid'] and 
                        old_price['ask'] == price_data['ask']):
                        continue  # No change, skip update
                
                # Update last price
                self.last_prices[symbol] = price_data
                
                # Notify callbacks
                for callback_fn in self.price_callbacks:
                    try:
                        callback_fn(symbol, price_data)
                    except Exception as e:
                        self.logger.error(f"Error in price callback: {str(e)}")
                
                # Send to WebSocket clients
                if self.enable_websocket and self.connected_clients:
                    message = json.dumps({
                        'type': 'price_update',
                        'data': price_data
                    })
                    self._broadcast_message(message)
            
            except Exception as e:
                self.logger.error(f"Error updating price for {symbol}: {str(e)}")
                
    def _generate_simulated_price(self, symbol):
        """
        Generate simulated price data for a symbol.
        
        Args:
            symbol (str): The symbol to generate price for.
            
        Returns:
            dict: Simulated price data.
        """
        import random
        
        # Get base price for this symbol, default to 1000.0 if not defined
        base_price = self.sim_price_base.get(symbol, 1000.0)
        
        # Calculate time since last update to scale volatility
        now = time.time()
        time_diff = now - self.last_sim_update
        self.last_sim_update = now
        
        # Add some simulated volatility (more realistic price movement)
        volatility = base_price * 0.001  # 0.1% base volatility
        
        # Generate a random price movement
        # Use time difference to scale movement (larger movements over longer periods)
        movement = random.uniform(-1, 1) * volatility * min(time_diff, 10)
        
        # Calculate new price
        new_price = base_price + movement
        
        # Update the base price for next time
        self.sim_price_base[symbol] = new_price
        
        # Calculate spread (0.01% to 0.05% of price)
        spread_pct = random.uniform(0.0001, 0.0005)
        spread_amount = new_price * spread_pct
        
        # Return formatted price data
        return {
            'symbol': symbol,
            'bid': max(0.1, round(new_price - (spread_amount/2), 2)),
            'ask': max(0.1, round(new_price + (spread_amount/2), 2)),
            'spread': round(spread_amount * 100, 2),  # convert to points
            'time': datetime.now().isoformat(),
            'simulated': True  # Mark as simulated data
        }
    
    def _start_websocket_server(self):
        """
        Start WebSocket server for streaming prices.
        """
        self.logger.info(f"Starting WebSocket server on port {self.websocket_port}")
        
        try:
            # Create event loop in the new thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Set up the WebSocket server
            start_server = websockets.serve(
                self._websocket_handler,
                '0.0.0.0',
                self.websocket_port
            )
            
            # Run the server until the thread is stopped
            loop.run_until_complete(start_server)
            loop.run_forever()
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {str(e)}")
            # Safely continue even if WebSocket fails
            pass
    
    async def _websocket_handler(self, websocket, path):
        """
        Handle WebSocket client connections.
        
        Args:
            websocket: WebSocket client connection.
            path: Connection path.
        """
        # Register the client
        with self.clients_lock:
            self.connected_clients.add(websocket)
            client_id = id(websocket)
            self.logger.info(f"WebSocket client connected: {client_id}")
        
        try:
            # Send current prices to the new client
            await self._send_initial_prices(websocket)
            
            # Keep the connection open and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Received invalid JSON from client {client_id}")
                except Exception as e:
                    self.logger.error(f"Error handling message from client {client_id}: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"Error in WebSocket handler: {str(e)}")
        finally:
            # Unregister the client
            with self.clients_lock:
                self.connected_clients.remove(websocket)
    
    async def _send_initial_prices(self, websocket):
        """
        Send initial prices to a newly connected WebSocket client.
        
        Args:
            websocket: WebSocket client connection.
        """
        prices = self.get_all_prices()
        if prices:
            message = json.dumps({
                'type': 'initial_prices',
                'data': prices
            })
            await websocket.send(message)
    
    async def _handle_client_message(self, websocket, data):
        """
        Handle messages from WebSocket clients.
        
        Args:
            websocket: WebSocket client connection.
            data: Message data from client.
        """
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            # Handle subscription to specific symbols
            symbols = data.get('symbols', [])
            for symbol in symbols:
                if symbol not in self.symbols:
                    self.add_symbol(symbol)
        
        elif message_type == 'unsubscribe':
            # Handle unsubscription from specific symbols
            symbols = data.get('symbols', [])
            for symbol in symbols:
                # Only remove if no other clients are interested
                self.remove_symbol(symbol)
        
        elif message_type == 'ping':
            # Respond to ping messages
            await websocket.send(json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}))
    
    def _broadcast_message(self, message):
        """
        Broadcast a message to all connected WebSocket clients.
        
        Args:
            message (str): Message to broadcast.
        """
        with self.clients_lock:
            # Make a copy of the set to avoid modification during iteration
            clients = self.connected_clients.copy()
        
        # Use asyncio to send messages
        for websocket in clients:
            asyncio.run_coroutine_threadsafe(
                self._safe_send(websocket, message),
                asyncio.get_event_loop()
            )
    
    async def _safe_send(self, websocket, message):
        """
        Safely send a message to a WebSocket client.
        
        Args:
            websocket: WebSocket client connection.
            message (str): Message to send.
        """
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            # Connection is closed, will be removed on next iteration
            pass
        except Exception as e:
            self.logger.error(f"Error sending message to WebSocket client: {str(e)}")

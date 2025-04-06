import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import time
import hashlib

class ExecutorAgent:
    """
    Agent responsible for executing trades via MT5 based on signals and risk parameters.
    """
    
    def __init__(self, config, mt5_connector):
        """
        Initialize the executor agent.
        
        Args:
            config (dict): Configuration for the agent.
            mt5_connector: MT5Connector instance for trade execution.
        """
        self.config = config
        self.mt5 = mt5_connector
        self.logger = logging.getLogger('ExecutorAgent')
        self.execution_memory = []
        self.max_memory_items = config.get('max_memory_items', 20)
        self.execution_delay = config.get('execution_delay', 0.5)  # Delay in seconds between actions
        self.max_retries = config.get('max_retries', 3)
        self.require_confirmation = config.get('require_confirmation', True)
        self.confirmation_callback = None
    
    def set_confirmation_callback(self, callback_function):
        """
        Set a callback function for trade confirmation.
        
        Args:
            callback_function: Function to call for trade confirmation.
        """
        self.confirmation_callback = callback_function
    
    def execute_trade(self, trade_plan, risk_parameters, market_conditions=None, require_confirmation=None):
        """
        Execute a trade based on the trade plan and risk parameters.
        
        Args:
            trade_plan (dict): Trade plan with signal and symbol information.
            risk_parameters (dict): Risk parameters including position size and price levels.
            market_conditions (dict, optional): Current market conditions.
            require_confirmation (bool, optional): Whether to require confirmation before execution.
            
        Returns:
            dict: Trade execution result.
        """
        # Use instance setting if not specified
        if require_confirmation is None:
            require_confirmation = self.require_confirmation
        
        try:
            # Extract trade parameters
            symbol = trade_plan.get('symbol', '')
            signal = trade_plan.get('signal', '')
            position_size = risk_parameters.get('position_size', 0)
            entry_price = risk_parameters.get('entry_price', 0)
            stop_loss = risk_parameters.get('stop_loss', 0)
            take_profit = risk_parameters.get('take_profit', 0)
            
            # Validate parameters
            if not symbol or not signal or signal not in ['buy', 'sell'] or position_size <= 0:
                error_message = "Invalid trade parameters"
                self.logger.error(error_message)
                return {"success": False, "error": error_message}
            
            # Generate unique trade ID
            trade_id = self._generate_trade_id(symbol, signal, position_size, entry_price)
            
            # Log the trade plan
            self.logger.info(f"Trade plan: {symbol} {signal} {position_size} lots at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
            
            # If confirmation is required, get confirmation
            if require_confirmation and self.confirmation_callback:
                confirmation = self.confirmation_callback(
                    symbol=symbol,
                    signal=signal,
                    position_size=position_size,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trade_id=trade_id
                )
                
                if not confirmation:
                    result = {"success": False, "error": "Trade not confirmed by user", "trade_id": trade_id}
                    self._add_to_memory({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'trade_execution',
                        'inputs': {
                            'trade_plan': trade_plan,
                            'risk_parameters': risk_parameters
                        },
                        'output': result
                    })
                    return result
            
            # Execute the trade
            order_type = "BUY" if signal == "buy" else "SELL"
            
            # Add a small comment with the strategy info
            comment = f"AI Trader {order_type} {trade_id[-6:]}"
            
            # Execute with retry logic
            for retry in range(self.max_retries):
                try:
                    # Get current price if needed
                    if entry_price <= 0:
                        symbol_info = self.mt5.get_symbol_info(symbol)
                        if symbol_info:
                            entry_price = symbol_info['ask'] if signal == 'buy' else symbol_info['bid']
                    
                    # Place the order
                    order_result = self.mt5.place_order(
                        symbol=symbol,
                        order_type=order_type,
                        volume=position_size,
                        price=entry_price,
                        sl=stop_loss,
                        tp=take_profit,
                        comment=comment
                    )
                    
                    # Check if the order was successful
                    if order_result is not None:
                        # Format the result
                        result = {
                            "success": True,
                            "order_id": order_result.get('order_id', 0),
                            "volume": order_result.get('volume', position_size),
                            "price": order_result.get('price', entry_price),
                            "trade_id": trade_id,
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "type": order_type,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        }
                        
                        # Store in memory
                        self._add_to_memory({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'trade_execution',
                            'inputs': {
                                'trade_plan': trade_plan,
                                'risk_parameters': risk_parameters
                            },
                            'output': result
                        })
                        
                        self.logger.info(f"Trade executed successfully: {symbol} {order_type} {position_size} lots at {order_result.get('price')}")
                        return result
                    else:
                        self.logger.warning(f"Order failed, retry {retry+1}/{self.max_retries}")
                        time.sleep(self.execution_delay)  # Wait before retry
                
                except Exception as e:
                    self.logger.error(f"Error in trade execution attempt {retry+1}: {str(e)}")
                    time.sleep(self.execution_delay)  # Wait before retry
            
            # If we get here, all retries failed
            error_message = f"Failed to execute trade after {self.max_retries} attempts"
            result = {"success": False, "error": error_message, "trade_id": trade_id}
            
            # Store in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'trade_execution',
                'inputs': {
                    'trade_plan': trade_plan,
                    'risk_parameters': risk_parameters
                },
                'output': result
            })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def close_position(self, position_id, partial=False, percent=100):
        """
        Close a specific position by ID.
        
        Args:
            position_id (int): Position ID to close.
            partial (bool, optional): Whether to close only part of the position.
            percent (float, optional): Percentage of position to close if partial.
            
        Returns:
            dict: Position close result.
        """
        try:
            # If not partial, close the entire position
            if not partial or percent >= 100:
                result = self.mt5.close_position(position_id)
                
                if result is not None:
                    close_result = {
                        "success": True,
                        "position_id": position_id,
                        "order_id": result.get('order_id', 0),
                        "volume": result.get('volume', 0),
                        "price": result.get('price', 0),
                        "timestamp": datetime.now().isoformat(),
                        "partial": False
                    }
                    
                    # Store in memory
                    self._add_to_memory({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'position_close',
                        'inputs': {
                            'position_id': position_id,
                            'partial': False
                        },
                        'output': close_result
                    })
                    
                    self.logger.info(f"Position {position_id} closed successfully at {result.get('price')}")
                    return close_result
                else:
                    error_message = "Failed to close position"
                    self.logger.error(f"{error_message}: {position_id}")
                    return {"success": False, "error": error_message, "position_id": position_id}
            
            # Handle partial close
            else:
                positions = self.mt5.get_positions()
                if positions is None:
                    return {"success": False, "error": "Failed to get positions", "position_id": position_id}
                
                position_data = positions[positions['ticket'] == position_id]
                if len(position_data) == 0:
                    return {"success": False, "error": "Position not found", "position_id": position_id}
                
                # Calculate volume to close
                total_volume = position_data.iloc[0]['volume']
                volume_to_close = total_volume * (percent / 100.0)
                
                # TODO: Implement partial close logic
                # This would require custom implementation depending on the broker's API
                # For now, return not supported
                return {"success": False, "error": "Partial close not supported yet", "position_id": position_id}
        
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {str(e)}")
            return {"success": False, "error": str(e), "position_id": position_id}
    
    def modify_position(self, position_id, new_sl=None, new_tp=None):
        """
        Modify stop loss and/or take profit for an existing position.
        
        Args:
            position_id (int): Position ID to modify.
            new_sl (float, optional): New stop loss level.
            new_tp (float, optional): New take profit level.
            
        Returns:
            dict: Position modification result.
        """
        try:
            # Get position information
            positions = self.mt5.get_positions()
            if positions is None:
                return {"success": False, "error": "Failed to get positions", "position_id": position_id}
            
            position_data = positions[positions['ticket'] == position_id]
            if len(position_data) == 0:
                return {"success": False, "error": "Position not found", "position_id": position_id}
            
            position = position_data.iloc[0]
            
            # Use current SL/TP if new values not provided
            if new_sl is None:
                new_sl = position['sl']
            
            if new_tp is None:
                new_tp = position['tp']
            
            # TODO: Implement position modification
            # This would require additional MT5 connector methods
            # For now, return not supported
            return {"success": False, "error": "Position modification not implemented yet", "position_id": position_id}
        
        except Exception as e:
            self.logger.error(f"Error modifying position {position_id}: {str(e)}")
            return {"success": False, "error": str(e), "position_id": position_id}
    
    def close_all_positions(self, symbol=None):
        """
        Close all open positions, optionally filtered by symbol.
        
        Args:
            symbol (str, optional): Symbol to filter positions by.
            
        Returns:
            dict: Result of closing all positions.
        """
        try:
            # Get all open positions
            positions = self.mt5.get_positions(symbol=symbol)
            if positions is None or len(positions) == 0:
                return {"success": True, "message": "No open positions to close", "count": 0}
            
            # Close each position
            results = []
            success_count = 0
            
            for _, position in positions.iterrows():
                position_id = position['ticket']
                result = self.close_position(position_id)
                
                results.append(result)
                if result.get('success', False):
                    success_count += 1
            
            # Summarize results
            close_all_result = {
                "success": success_count > 0,
                "total_positions": len(positions),
                "successful_closes": success_count,
                "failed_closes": len(positions) - success_count,
                "results": results
            }
            
            # Store in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'close_all_positions',
                'inputs': {
                    'symbol': symbol
                },
                'output': close_all_result
            })
            
            return close_all_result
        
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def one_click_trade(self, symbol, signal, risk_level="medium", market_order=True):
        """
        Execute a one-click trade with simplified parameters.
        
        Args:
            symbol (str): The trading symbol.
            signal (str): Trade direction ("buy" or "sell").
            risk_level (str, optional): Risk level ("low", "medium", "high").
            market_order (bool, optional): Whether to use market order.
            
        Returns:
            dict: Trade execution result.
        """
        try:
            # Validate parameters
            if signal not in ["buy", "sell"]:
                return {"success": False, "error": "Invalid signal, must be 'buy' or 'sell'"}
            
            if risk_level not in ["low", "medium", "high"]:
                risk_level = "medium"
            
            # Risk multipliers for different risk levels
            risk_multipliers = {
                "low": 0.5,
                "medium": 1.0,
                "high": 2.0
            }
            
            # Get account info
            account_info = self.mt5.get_account_info()
            if account_info is None:
                return {"success": False, "error": "Failed to get account info"}
            
            # Get symbol info
            symbol_info = self.mt5.get_symbol_info(symbol)
            if symbol_info is None:
                return {"success": False, "error": f"Failed to get symbol info for {symbol}"}
            
            # Determine price
            price = 0.0 if market_order else (symbol_info['ask'] if signal == 'buy' else symbol_info['bid'])
            
            # Get default risk percentage based on risk level
            base_risk = self.config.get('default_risk_per_trade', 0.01)  # 1% default
            risk_percent = base_risk * risk_multipliers[risk_level]
            
            # Calculate position size based on account balance and risk
            balance = account_info['balance']
            risk_amount = balance * risk_percent
            
            # Use simple calculation based on 2% price movement for stop loss
            current_price = symbol_info['ask'] if signal == 'buy' else symbol_info['bid']
            
            # Calculate stop loss and take profit
            if signal == 'buy':
                stop_loss = current_price * 0.98  # 2% below for buy
                take_profit = current_price * 1.04  # 4% above for buy
            else:
                stop_loss = current_price * 1.02  # 2% above for sell
                take_profit = current_price * 0.96  # 4% below for sell
            
            # Calculate position size (simplified)
            price_risk = abs(current_price - stop_loss)
            tick_size = symbol_info['trade_tick_size']
            tick_value = symbol_info['trade_tick_value']
            
            ticks_at_risk = price_risk / tick_size if tick_size > 0 else 100
            risk_per_lot = ticks_at_risk * tick_value
            
            position_size = risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01
            
            # Ensure position size is within limits
            position_size = max(symbol_info['volume_min'], min(position_size, symbol_info['volume_max']))
            
            # Round to volume step
            volume_step = symbol_info['volume_step']
            position_size = math.floor(position_size / volume_step) * volume_step
            
            # Prepare trade plan and risk parameters
            trade_plan = {
                'symbol': symbol,
                'signal': signal,
                'market_order': market_order
            }
            
            risk_parameters = {
                'position_size': position_size,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_percent': risk_percent * 100,  # to percentage
            }
            
            # Execute the trade
            result = self.execute_trade(trade_plan, risk_parameters)
            
            # Add one-click info to the result
            result['risk_level'] = risk_level
            result['one_click'] = True
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in one-click trade: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_open_positions(self, symbol=None):
        """
        Get all open positions, optionally filtered by symbol.
        
        Args:
            symbol (str, optional): Symbol to filter positions by.
            
        Returns:
            list: Open positions.
        """
        try:
            positions_df = self.mt5.get_positions(symbol=symbol)
            if positions_df is None:
                return []
            
            # Convert DataFrame to list of dictionaries
            positions = positions_df.to_dict('records')
            
            # Clean up datetime objects for JSON serialization
            for position in positions:
                if 'time' in position and isinstance(position['time'], datetime):
                    position['time'] = position['time'].isoformat()
            
            return positions
        
        except Exception as e:
            self.logger.error(f"Error getting open positions: {str(e)}")
            return []
    
    def _generate_trade_id(self, symbol, signal, position_size, price):
        """
        Generate a unique trade ID.
        
        Args:
            symbol (str): The trading symbol.
            signal (str): Trade direction.
            position_size (float): Position size.
            price (float): Entry price.
            
        Returns:
            str: Unique trade ID.
        """
        timestamp = datetime.now().isoformat()
        input_string = f"{symbol}_{signal}_{position_size}_{price}_{timestamp}"
        
        # Generate hash
        hash_object = hashlib.md5(input_string.encode())
        trade_id = hash_object.hexdigest()
        
        return trade_id
    
    def _add_to_memory(self, item):
        """
        Add an item to the agent's memory, maintaining max size.
        
        Args:
            item (dict): The memory item to add.
        """
        self.execution_memory.append(item)
        
        # Keep memory size limited
        if len(self.execution_memory) > self.max_memory_items:
            self.execution_memory.pop(0)
    
    def get_memory(self):
        """
        Get the agent's memory.
        
        Returns:
            list: The agent's memory items.
        """
        return self.execution_memory

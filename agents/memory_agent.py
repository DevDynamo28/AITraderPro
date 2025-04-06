import os
import json
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryAgent:
    """
    Agent responsible for storing and retrieving trading data.
    Acts as a persistent memory for the trading system.
    """
    
    def __init__(self, config):
        """
        Initialize the memory agent.
        
        Args:
            config (dict): Configuration for the agent.
        """
        self.config = config
        self.data_dir = Path("data")
        self.trades_file = self.data_dir / "trades.json"
        self.memory_file = self.data_dir / "memory.json"
        self.reflections_file = self.data_dir / "reflections.json"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize empty data structures
        self.trades = self._load_json(self.trades_file, [])
        self.memory = self._load_json(self.memory_file, {
            "market_insights": [],
            "performance_metrics": {},
            "system_events": []
        })
        self.reflections = self._load_json(self.reflections_file, [])
    
    def _load_json(self, file_path, default=None):
        """
        Load data from a JSON file or return default if file doesn't exist.
        
        Args:
            file_path (Path): Path to the JSON file.
            default: Default value to return if file doesn't exist.
            
        Returns:
            The loaded JSON data or the default value.
        """
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return default
    
    def _save_json(self, file_path, data):
        """
        Save data to a JSON file.
        
        Args:
            file_path (Path): Path to the JSON file.
            data: Data to save.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {str(e)}")
            return False
    
    def store_trade(self, trade_data):
        """
        Store a trade record.
        
        Args:
            trade_data (dict): Trade data to store.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Ensure trade has a timestamp
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        # Add trade to list
        self.trades.append(trade_data)
        
        # Save to file
        return self._save_json(self.trades_file, self.trades)
    
    def update_trade(self, trade_id, updates):
        """
        Update an existing trade record.
        
        Args:
            trade_id (str): ID of the trade to update.
            updates (dict): Updates to apply to the trade.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        for i, trade in enumerate(self.trades):
            if trade.get('trade_id') == trade_id:
                self.trades[i].update(updates)
                return self._save_json(self.trades_file, self.trades)
        
        logger.warning(f"Trade with ID {trade_id} not found for update.")
        return False
    
    def get_trade(self, trade_id):
        """
        Get a specific trade by ID.
        
        Args:
            trade_id (str): ID of the trade to retrieve.
            
        Returns:
            dict: The trade data or None if not found.
        """
        for trade in self.trades:
            if trade.get('trade_id') == trade_id:
                return trade
        return None
    
    def get_recent_trades(self, limit=50):
        """
        Get recent trades, sorted by timestamp.
        
        Args:
            limit (int): Maximum number of trades to return.
            
        Returns:
            list: Recent trades.
        """
        # Sort trades by timestamp (newest first)
        sorted_trades = sorted(
            self.trades,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return sorted_trades[:limit]
    
    def get_trade_statistics(self):
        """
        Calculate and return trading statistics.
        
        Returns:
            dict: Trading statistics.
        """
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        # No trades yet
        if not self.trades:
            # Set profit factor to 1.0 (neutral) when no trades
            stats['profit_factor'] = 1.0
            return stats
        
        # Calculate statistics
        for trade in self.trades:
            profit = trade.get('profit', 0.0)
            
            # Skip trades without profit data or still open
            if profit is None:
                continue
            
            stats['total_profit'] += profit if profit > 0 else 0
            stats['total_loss'] += abs(profit) if profit < 0 else 0
            
            if profit > 0:
                stats['winning_trades'] += 1
                stats['best_trade'] = max(stats['best_trade'], profit)
            elif profit < 0:
                stats['losing_trades'] += 1
                stats['worst_trade'] = min(stats['worst_trade'], profit)
        
        # Calculate averages and ratios
        closed_trades = stats['winning_trades'] + stats['losing_trades']
        
        if closed_trades > 0:
            stats['win_rate'] = stats['winning_trades'] / closed_trades
            
            if stats['winning_trades'] > 0:
                stats['avg_profit'] = stats['total_profit'] / stats['winning_trades']
            
            if stats['losing_trades'] > 0:
                stats['avg_loss'] = stats['total_loss'] / stats['losing_trades']
            
            if stats['total_loss'] > 0:
                stats['profit_factor'] = stats['total_profit'] / stats['total_loss']
            else:
                # If no losses, profit factor is very good
                stats['profit_factor'] = 100.0
        
        return stats
    
    def store_market_insight(self, insight):
        """
        Store a market insight.
        
        Args:
            insight (dict): Market insight data.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Ensure insight has a timestamp
        if 'timestamp' not in insight:
            insight['timestamp'] = datetime.now().isoformat()
        
        # Add insight to memory
        self.memory['market_insights'].append(insight)
        
        # Trim to keep only recent insights (last 100)
        if len(self.memory['market_insights']) > 100:
            self.memory['market_insights'] = self.memory['market_insights'][-100:]
        
        # Save to file
        return self._save_json(self.memory_file, self.memory)
    
    def get_market_insights(self, limit=10):
        """
        Get recent market insights.
        
        Args:
            limit (int): Maximum number of insights to return.
            
        Returns:
            list: Recent market insights.
        """
        # Sort insights by timestamp (newest first)
        sorted_insights = sorted(
            self.memory['market_insights'],
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return sorted_insights[:limit]
    
    def store_system_event(self, event):
        """
        Store a system event.
        
        Args:
            event (dict): System event data.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Ensure event has a timestamp
        if 'timestamp' not in event:
            event['timestamp'] = datetime.now().isoformat()
        
        # Add event to memory
        self.memory['system_events'].append(event)
        
        # Trim to keep only recent events (last 1000)
        if len(self.memory['system_events']) > 1000:
            self.memory['system_events'] = self.memory['system_events'][-1000:]
        
        # Save to file
        return self._save_json(self.memory_file, self.memory)
    
    def get_system_events(self, limit=50, event_type=None):
        """
        Get recent system events, optionally filtered by type.
        
        Args:
            limit (int): Maximum number of events to return.
            event_type (str, optional): Type of events to filter by.
            
        Returns:
            list: Recent system events.
        """
        # Filter by event type if specified
        events = self.memory['system_events']
        if event_type:
            events = [e for e in events if e.get('type') == event_type]
        
        # Sort events by timestamp (newest first)
        sorted_events = sorted(
            events,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return sorted_events[:limit]
    
    def update_performance_metrics(self, metrics):
        """
        Update performance metrics.
        
        Args:
            metrics (dict): Performance metrics to update.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Update metrics in memory
        self.memory['performance_metrics'].update(metrics)
        
        # Save to file
        return self._save_json(self.memory_file, self.memory)
    
    def get_performance_metrics(self):
        """
        Get all performance metrics.
        
        Returns:
            dict: Performance metrics.
        """
        return self.memory['performance_metrics']
    
    def store_reflection(self, reflection):
        """
        Store a trading reflection.
        
        Args:
            reflection (dict): Reflection data.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Ensure reflection has a timestamp
        if 'timestamp' not in reflection:
            reflection['timestamp'] = datetime.now().isoformat()
        
        # Add reflection to list
        self.reflections.append(reflection)
        
        # Save to file
        return self._save_json(self.reflections_file, self.reflections)
    
    def get_reflections(self, limit=10, period=None):
        """
        Get recent reflections, optionally filtered by period.
        
        Args:
            limit (int): Maximum number of reflections to return.
            period (str, optional): Period to filter by ('daily', 'weekly', 'monthly').
            
        Returns:
            list: Recent reflections.
        """
        # Filter by period if specified
        reflections = self.reflections
        if period:
            reflections = [r for r in reflections if r.get('period') == period]
        
        # Sort reflections by timestamp (newest first)
        sorted_reflections = sorted(
            reflections,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return sorted_reflections[:limit]
    
    def get_latest_reflection(self, period=None):
        """
        Get the latest reflection, optionally for a specific period.
        
        Args:
            period (str, optional): Period to filter by ('daily', 'weekly', 'monthly').
            
        Returns:
            dict: Latest reflection or None if not found.
        """
        reflections = self.get_reflections(limit=1, period=period)
        return reflections[0] if reflections else None
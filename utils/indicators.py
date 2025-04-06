import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import Dict, Any, Union

class Indicators:
    """
    Class for calculating technical indicators on OHLCV data.
    """
    
    def __init__(self, df=None):
        """
        Initialize with optional dataframe.
        
        Args:
            df (DataFrame, optional): Pandas DataFrame with OHLCV data.
        """
        self.df = df
        self.logger = logging.getLogger('Indicators')
    
    def set_data(self, df):
        """
        Set the dataframe to use for calculations.
        
        Args:
            df (DataFrame): Pandas DataFrame with OHLCV data.
        """
        self.df = df
    
    def _validate_data(self):
        """
        Validate that we have dataframe with required columns.
        
        Returns:
            bool: True if data is valid, False otherwise.
        """
        if self.df is None:
            self.logger.error("No data provided for indicator calculation")
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns for indicator calculation: {missing_columns}")
            return False
        
        return True
    
    def rsi(self, length=14, scalar=None, drift=None, offset=None):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            length (int): The period length.
            scalar (float, optional): A scale factor.
            drift (int, optional): The difference period.
            offset (int, optional): Number of periods to offset the result.
            
        Returns:
            Series: RSI values or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            return ta.rsi(self.df['close'], length=length, scalar=scalar, drift=drift, offset=offset)
        except Exception as e:
            self.logger.error(f"Failed to calculate RSI: {str(e)}")
            return None
    
    def macd(self, fast=12, slow=26, signal=9, offset=None):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            fast (int): The short period.
            slow (int): The long period.
            signal (int): The signal period.
            offset (int, optional): Number of periods to offset the result.
            
        Returns:
            DataFrame: MACD DataFrame with columns ['MACD', 'MACD_signal', 'MACD_hist'] or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            return ta.macd(self.df['close'], fast=fast, slow=slow, signal=signal, offset=offset)
        except Exception as e:
            self.logger.error(f"Failed to calculate MACD: {str(e)}")
            return None
    
    def bollinger_bands(self, length=20, std=2, mamode='sma', offset=None):
        """
        Calculate Bollinger Bands.
        
        Args:
            length (int): The period length.
            std (float): The standard deviation multiplier.
            mamode (str): Moving average type.
            offset (int, optional): Number of periods to offset the result.
            
        Returns:
            DataFrame: Bollinger Bands DataFrame with columns ['BBL', 'BBM', 'BBU', 'BBB', 'BBP'] or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            return ta.bbands(self.df['close'], length=length, std=std, mamode=mamode, offset=offset)
        except Exception as e:
            self.logger.error(f"Failed to calculate Bollinger Bands: {str(e)}")
            return None
    
    def stochastic(self, k=14, d=3, smooth_k=3, offset=None):
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k (int): The %K period.
            d (int): The %D period.
            smooth_k (int): The %K smoothing period.
            offset (int, optional): Number of periods to offset the result.
            
        Returns:
            DataFrame: Stochastic Oscillator DataFrame with columns ['STOCHk', 'STOCHd'] or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            return ta.stoch(self.df['high'], self.df['low'], self.df['close'], k=k, d=d, smooth_k=smooth_k, offset=offset)
        except Exception as e:
            self.logger.error(f"Failed to calculate Stochastic Oscillator: {str(e)}")
            return None
    
    def ema(self, length=10, offset=None):
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            length (int): The period length.
            offset (int, optional): Number of periods to offset the result.
            
        Returns:
            Series: EMA values or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            return ta.ema(self.df['close'], length=length, offset=offset)
        except Exception as e:
            self.logger.error(f"Failed to calculate EMA: {str(e)}")
            return None
    
    def sma(self, length=10, offset=None):
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            length (int): The period length.
            offset (int, optional): Number of periods to offset the result.
            
        Returns:
            Series: SMA values or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            return ta.sma(self.df['close'], length=length, offset=offset)
        except Exception as e:
            self.logger.error(f"Failed to calculate SMA: {str(e)}")
            return None
    
    def atr(self, length=14, mamode='sma', offset=None):
        """
        Calculate Average True Range (ATR).
        
        Args:
            length (int): The period length.
            mamode (str): Moving average type.
            offset (int, optional): Number of periods to offset the result.
            
        Returns:
            Series: ATR values or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            return ta.atr(self.df['high'], self.df['low'], self.df['close'], length=length, mamode=mamode, offset=offset)
        except Exception as e:
            self.logger.error(f"Failed to calculate ATR: {str(e)}")
            return None
    
    def custom_rsi_strategy(self, rsi_length=14, rsi_overbought=70, rsi_oversold=30):
        """
        Custom RSI-based trading strategy.
        
        Args:
            rsi_length (int): RSI period length.
            rsi_overbought (int): RSI overbought threshold.
            rsi_oversold (int): RSI oversold threshold.
            
        Returns:
            DataFrame: DataFrame with signals or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            # Calculate RSI
            rsi_values = self.rsi(length=rsi_length)
            if rsi_values is None:
                return None
            
            # Create DataFrame for signals
            signals = pd.DataFrame(index=self.df.index)
            signals['rsi'] = rsi_values
            
            # Generate signals
            signals['signal'] = 0  # 0: no signal, 1: buy, -1: sell
            signals.loc[signals['rsi'] < rsi_oversold, 'signal'] = 1
            signals.loc[signals['rsi'] > rsi_overbought, 'signal'] = -1
            
            return signals
        except Exception as e:
            self.logger.error(f"Failed to calculate RSI strategy: {str(e)}")
            return None
    
    def custom_macd_strategy(self, fast=12, slow=26, signal=9):
        """
        Custom MACD-based trading strategy.
        
        Args:
            fast (int): MACD fast period.
            slow (int): MACD slow period.
            signal (int): MACD signal period.
            
        Returns:
            DataFrame: DataFrame with signals or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            # Calculate MACD
            macd_values = self.macd(fast=fast, slow=slow, signal=signal)
            if macd_values is None:
                return None
            
            # Create DataFrame for signals
            signals = pd.DataFrame(index=self.df.index)
            signals['macd'] = macd_values['MACD']
            signals['macd_signal'] = macd_values['MACD_signal']
            signals['macd_hist'] = macd_values['MACD_hist']
            
            # Generate signals based on MACD crossover
            signals['signal'] = 0  # 0: no signal, 1: buy, -1: sell
            
            # Buy signal: MACD crosses above signal line
            buy_signal = (signals['macd'] > signals['macd_signal']) & (signals['macd'].shift(1) <= signals['macd_signal'].shift(1))
            signals.loc[buy_signal, 'signal'] = 1
            
            # Sell signal: MACD crosses below signal line
            sell_signal = (signals['macd'] < signals['macd_signal']) & (signals['macd'].shift(1) >= signals['macd_signal'].shift(1))
            signals.loc[sell_signal, 'signal'] = -1
            
            return signals
        except Exception as e:
            self.logger.error(f"Failed to calculate MACD strategy: {str(e)}")
            return None
    
    def rsi_macd_combined_strategy(self, rsi_length=14, rsi_overbought=70, rsi_oversold=30, 
                                  macd_fast=12, macd_slow=26, macd_signal=9):
        """
        Custom combined RSI and MACD trading strategy.
        
        Args:
            rsi_length (int): RSI period length.
            rsi_overbought (int): RSI overbought threshold.
            rsi_oversold (int): RSI oversold threshold.
            macd_fast (int): MACD fast period.
            macd_slow (int): MACD slow period.
            macd_signal (int): MACD signal period.
            
        Returns:
            DataFrame: DataFrame with signals or None if calculation fails.
        """
        if not self._validate_data():
            return None
        
        try:
            # Calculate RSI
            rsi_values = self.rsi(length=rsi_length)
            if rsi_values is None:
                return None
            
            # Calculate MACD
            macd_values = self.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd_values is None:
                return None
            
            # Create DataFrame for signals
            signals = pd.DataFrame(index=self.df.index)
            signals['rsi'] = rsi_values
            signals['macd'] = macd_values['MACD']
            signals['macd_signal'] = macd_values['MACD_signal']
            signals['macd_hist'] = macd_values['MACD_hist']
            
            # Generate RSI signals
            signals['rsi_signal'] = 0
            signals.loc[signals['rsi'] < rsi_oversold, 'rsi_signal'] = 1
            signals.loc[signals['rsi'] > rsi_overbought, 'rsi_signal'] = -1
            
            # Generate MACD signals
            signals['macd_signal_line'] = 0
            
            # Buy signal: MACD crosses above signal line
            buy_signal = (signals['macd'] > signals['macd_signal']) & (signals['macd'].shift(1) <= signals['macd_signal'].shift(1))
            signals.loc[buy_signal, 'macd_signal_line'] = 1
            
            # Sell signal: MACD crosses below signal line
            sell_signal = (signals['macd'] < signals['macd_signal']) & (signals['macd'].shift(1) >= signals['macd_signal'].shift(1))
            signals.loc[sell_signal, 'macd_signal_line'] = -1
            
            # Combined signal
            signals['combined_signal'] = 0
            
            # Strong buy: Both RSI and MACD indicate buy
            strong_buy = (signals['rsi_signal'] == 1) & (signals['macd_signal_line'] == 1)
            signals.loc[strong_buy, 'combined_signal'] = 2
            
            # Weak buy: Either RSI or MACD indicates buy
            weak_buy = ((signals['rsi_signal'] == 1) & (signals['macd_signal_line'] == 0)) | \
                       ((signals['rsi_signal'] == 0) & (signals['macd_signal_line'] == 1))
            signals.loc[weak_buy, 'combined_signal'] = 1
            
            # Strong sell: Both RSI and MACD indicate sell
            strong_sell = (signals['rsi_signal'] == -1) & (signals['macd_signal_line'] == -1)
            signals.loc[strong_sell, 'combined_signal'] = -2
            
            # Weak sell: Either RSI or MACD indicates sell
            weak_sell = ((signals['rsi_signal'] == -1) & (signals['macd_signal_line'] == 0)) | \
                        ((signals['rsi_signal'] == 0) & (signals['macd_signal_line'] == -1))
            signals.loc[weak_sell, 'combined_signal'] = -1
            
            return signals
        except Exception as e:
            self.logger.error(f"Failed to calculate combined RSI MACD strategy: {str(e)}")
            return None
    
    def calculate_all_indicators(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Calculate multiple indicators based on configuration.
        
        Args:
            config (Dict[str, Dict[str, Any]]): Dictionary where keys are indicator names and values are parameter dictionaries.
            
        Returns:
            Dict[str, Union[pd.Series, pd.DataFrame]]: Dictionary of calculated indicators.
        """
        if not self._validate_data():
            return {}
        
        results = {}
        
        for indicator_name, params in config.items():
            if hasattr(self, indicator_name):
                try:
                    indicator_method = getattr(self, indicator_name)
                    result = indicator_method(**params)
                    results[indicator_name] = result
                except Exception as e:
                    self.logger.error(f"Error calculating indicator {indicator_name}: {str(e)}")
            else:
                self.logger.warning(f"Indicator {indicator_name} not found")
        
        return results

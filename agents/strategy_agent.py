import json
import logging
import os
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger(__name__)

# For OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class StrategyAgent:
    """
    Agent responsible for analyzing market data and generating trading signals.
    Uses GPT-4o to analyze market data and indicators.
    """
    
    def __init__(self, config, market_data_util):
        """
        Initialize the strategy agent.
        
        Args:
            config (dict): Configuration for the agent.
            market_data_util: MarketDataUtil instance for data access.
        """
        self.config = config
        self.market_data_util = market_data_util
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        
        # Supported strategies
        self.supported_strategies = {
            'rsi_macd': self._rsi_macd_strategy,
            'trend_following': self._trend_following_strategy,
            'support_resistance': self._support_resistance_strategy,
            'breakout': self._breakout_strategy,
            'ai_analysis': self._ai_analysis
        }
        
        # Default strategy
        self.default_strategy = 'rsi_macd'
    
    def analyze_market(self, symbol=None, timeframe=None, strategies=None):
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol (str, optional): Symbol to analyze. Defaults to configured symbol.
            timeframe (str, optional): Timeframe to analyze. Defaults to configured timeframe.
            strategies (list, optional): List of strategies to use. Defaults to all strategies.
            
        Returns:
            dict: Analysis results including signals and reasoning.
        """
        # Use default values from config if not specified
        symbol = symbol or self.config['trading']['symbol']
        timeframe = timeframe or self.config['trading']['timeframe']
        
        # Get market data
        historical_data = self.market_data_util.get_historical_data(symbol, timeframe)
        
        if historical_data is None or len(historical_data) < 50:
            logger.warning(f"Insufficient data for {symbol} on {timeframe} timeframe")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': 'neutral',
                'confidence': 0.0,
                'reasoning': 'Insufficient data for analysis',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate indicators
        indicators = self.market_data_util.calculate_indicators(historical_data)
        
        # Collect analysis from each strategy
        strategy_results = []
        
        # Use specified strategies or all supported strategies
        strategies_to_use = strategies or list(self.supported_strategies.keys())
        
        for strategy in strategies_to_use:
            if strategy in self.supported_strategies:
                strategy_fn = self.supported_strategies[strategy]
                result = strategy_fn(historical_data, indicators)
                result['strategy'] = strategy
                strategy_results.append(result)
        
        # Combine strategy results (prioritizing AI analysis)
        combined_result = self._combine_strategy_results(strategy_results)
        
        # Format the final result
        analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': combined_result['signal'],
            'confidence': combined_result['confidence'],
            'reasoning': combined_result['reasoning'],
            'summary': combined_result.get('summary', ''),
            'timestamp': datetime.now().isoformat(),
            'strategy_results': strategy_results
        }
        
        return analysis
    
    def _rsi_macd_strategy(self, historical_data, indicators):
        """
        Apply RSI and MACD strategy.
        
        Args:
            historical_data (DataFrame): Historical market data.
            indicators (dict): Pre-calculated indicators.
            
        Returns:
            dict: Strategy result.
        """
        # Get the latest indicator values
        rsi = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
        macd = indicators['macd'][-1] if len(indicators['macd']) > 0 else 0
        macd_signal = indicators['macd_signal'][-1] if len(indicators['macd_signal']) > 0 else 0
        
        # RSI configuration
        rsi_overbought = self.config['indicators']['rsi']['overbought']
        rsi_oversold = self.config['indicators']['rsi']['oversold']
        
        # Determine signal based on RSI and MACD
        signal = 'neutral'
        confidence = 0.5
        reasoning = []
        
        # RSI signals
        if rsi > rsi_overbought:
            signal = 'sell'
            confidence = 0.6
            reasoning.append(f"RSI is overbought at {rsi:.2f} (above {rsi_overbought})")
        elif rsi < rsi_oversold:
            signal = 'buy'
            confidence = 0.6
            reasoning.append(f"RSI is oversold at {rsi:.2f} (below {rsi_oversold})")
        else:
            reasoning.append(f"RSI is neutral at {rsi:.2f}")
        
        # MACD signals
        if macd > macd_signal:
            if macd > 0 and macd_signal > 0:
                # Strong bullish
                if signal == 'buy':
                    confidence = 0.8
                elif signal == 'neutral':
                    signal = 'buy'
                    confidence = 0.6
                else:  # signal == 'sell'
                    confidence = 0.3  # Conflicting signals
            reasoning.append(f"MACD ({macd:.4f}) is above signal line ({macd_signal:.4f}), indicating bullish momentum")
        else:
            if macd < 0 and macd_signal < 0:
                # Strong bearish
                if signal == 'sell':
                    confidence = 0.8
                elif signal == 'neutral':
                    signal = 'sell'
                    confidence = 0.6
                else:  # signal == 'buy'
                    confidence = 0.3  # Conflicting signals
            reasoning.append(f"MACD ({macd:.4f}) is below signal line ({macd_signal:.4f}), indicating bearish momentum")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': ' '.join(reasoning)
        }
    
    def _trend_following_strategy(self, historical_data, indicators):
        """
        Apply trend following strategy using moving averages.
        
        Args:
            historical_data (DataFrame): Historical market data.
            indicators (dict): Pre-calculated indicators.
            
        Returns:
            dict: Strategy result.
        """
        # Get the latest moving averages
        ma20 = indicators['ma20'][-1] if len(indicators['ma20']) > 0 else 0
        ma50 = indicators['ma50'][-1] if len(indicators['ma50']) > 0 else 0
        ma200 = indicators['ma200'][-1] if len(indicators['ma200']) > 0 else 0
        
        # Get latest close price
        close = historical_data['close'].iloc[-1]
        
        # Determine the trend
        signal = 'neutral'
        confidence = 0.5
        reasoning = []
        
        # Short-term trend (MA20 vs price)
        if close > ma20:
            reasoning.append(f"Price ({close:.2f}) is above MA20 ({ma20:.2f}), indicating short-term bullish trend")
            confidence = 0.55
            signal = 'buy'
        else:
            reasoning.append(f"Price ({close:.2f}) is below MA20 ({ma20:.2f}), indicating short-term bearish trend")
            confidence = 0.55
            signal = 'sell'
        
        # Medium-term trend (MA20 vs MA50)
        if ma20 > ma50:
            reasoning.append(f"MA20 ({ma20:.2f}) is above MA50 ({ma50:.2f}), indicating medium-term bullish trend")
            if signal == 'buy':
                confidence = 0.7
            elif signal == 'sell':
                confidence = 0.4  # Conflicting signals
        else:
            reasoning.append(f"MA20 ({ma20:.2f}) is below MA50 ({ma50:.2f}), indicating medium-term bearish trend")
            if signal == 'sell':
                confidence = 0.7
            elif signal == 'buy':
                confidence = 0.4  # Conflicting signals
        
        # Long-term trend (MA50 vs MA200)
        if ma50 > ma200:
            reasoning.append(f"MA50 ({ma50:.2f}) is above MA200 ({ma200:.2f}), indicating long-term bullish trend")
            if signal == 'buy':
                confidence = 0.8
            elif signal == 'sell' and confidence < 0.6:
                signal = 'neutral'
                confidence = 0.5
        else:
            reasoning.append(f"MA50 ({ma50:.2f}) is below MA200 ({ma200:.2f}), indicating long-term bearish trend")
            if signal == 'sell':
                confidence = 0.8
            elif signal == 'buy' and confidence < 0.6:
                signal = 'neutral'
                confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': ' '.join(reasoning)
        }
    
    def _support_resistance_strategy(self, historical_data, indicators):
        """
        Apply support and resistance strategy.
        
        Args:
            historical_data (DataFrame): Historical market data.
            indicators (dict): Pre-calculated indicators.
            
        Returns:
            dict: Strategy result.
        """
        # Identify support and resistance levels
        levels = self.market_data_util.find_support_resistance_levels(historical_data)
        
        # Get current price
        current_price = historical_data['close'].iloc[-1]
        
        # Find nearest support and resistance
        nearest_support = None
        nearest_resistance = None
        
        for level in levels['support']:
            if level < current_price and (nearest_support is None or level > nearest_support):
                nearest_support = level
        
        for level in levels['resistance']:
            if level > current_price and (nearest_resistance is None or level < nearest_resistance):
                nearest_resistance = level
        
        # Determine signal
        signal = 'neutral'
        confidence = 0.5
        reasoning = []
        
        if nearest_support is not None and nearest_resistance is not None:
            # Calculate distances
            distance_to_support = current_price - nearest_support
            distance_to_resistance = nearest_resistance - current_price
            
            # Calculate price range
            price_range = nearest_resistance - nearest_support
            support_percentage = (distance_to_support / price_range) * 100
            resistance_percentage = (distance_to_resistance / price_range) * 100
            
            if support_percentage < 5:
                # Price is near support
                signal = 'buy'
                confidence = 0.7
                reasoning.append(f"Price ({current_price:.2f}) is very close to support level ({nearest_support:.2f}), bounce likely")
            elif resistance_percentage < 5:
                # Price is near resistance
                signal = 'sell'
                confidence = 0.7
                reasoning.append(f"Price ({current_price:.2f}) is very close to resistance level ({nearest_resistance:.2f}), reversal likely")
            elif support_percentage < 20:
                # Price is near-ish to support
                signal = 'buy'
                confidence = 0.6
                reasoning.append(f"Price ({current_price:.2f}) is approaching support level ({nearest_support:.2f})")
            elif resistance_percentage < 20:
                # Price is near-ish to resistance
                signal = 'sell'
                confidence = 0.6
                reasoning.append(f"Price ({current_price:.2f}) is approaching resistance level ({nearest_resistance:.2f})")
            else:
                reasoning.append(f"Price ({current_price:.2f}) is in the middle range between support ({nearest_support:.2f}) and resistance ({nearest_resistance:.2f})")
        else:
            reasoning.append("Could not identify clear support and resistance levels")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': ' '.join(reasoning)
        }
    
    def _breakout_strategy(self, historical_data, indicators):
        """
        Apply breakout strategy.
        
        Args:
            historical_data (DataFrame): Historical market data.
            indicators (dict): Pre-calculated indicators.
            
        Returns:
            dict: Strategy result.
        """
        # Get Bollinger Bands
        upper_band = indicators['bb_upper'][-1] if len(indicators['bb_upper']) > 0 else 0
        lower_band = indicators['bb_lower'][-1] if len(indicators['bb_lower']) > 0 else 0
        
        # Get current and previous close prices
        close = historical_data['close'].iloc[-1]
        prev_close = historical_data['close'].iloc[-2] if len(historical_data) > 1 else close
        
        # Get volume
        volume = historical_data['volume'].iloc[-1] if 'volume' in historical_data else 0
        avg_volume = historical_data['volume'].mean() if 'volume' in historical_data else 0
        
        # Determine signal
        signal = 'neutral'
        confidence = 0.5
        reasoning = []
        
        # Check for breakouts
        if close > upper_band and prev_close <= upper_band:
            signal = 'buy'
            confidence = 0.7
            reasoning.append(f"Price ({close:.2f}) broke above upper Bollinger Band ({upper_band:.2f}), indicating potential bullish breakout")
            
            # Confirm with volume
            if volume > avg_volume * 1.5:
                confidence = 0.8
                reasoning.append(f"Breakout confirmed with high volume ({volume:.0f} vs avg {avg_volume:.0f})")
        
        elif close < lower_band and prev_close >= lower_band:
            signal = 'sell'
            confidence = 0.7
            reasoning.append(f"Price ({close:.2f}) broke below lower Bollinger Band ({lower_band:.2f}), indicating potential bearish breakdown")
            
            # Confirm with volume
            if volume > avg_volume * 1.5:
                confidence = 0.8
                reasoning.append(f"Breakdown confirmed with high volume ({volume:.0f} vs avg {avg_volume:.0f})")
        
        else:
            # Check for tests of bands
            if close > 0.98 * upper_band:
                signal = 'sell'
                confidence = 0.6
                reasoning.append(f"Price ({close:.2f}) is testing upper Bollinger Band ({upper_band:.2f}), potential reversal point")
            
            elif close < 1.02 * lower_band:
                signal = 'buy'
                confidence = 0.6
                reasoning.append(f"Price ({close:.2f}) is testing lower Bollinger Band ({lower_band:.2f}), potential reversal point")
            
            else:
                reasoning.append(f"Price ({close:.2f}) is within normal Bollinger Band range ({lower_band:.2f} - {upper_band:.2f})")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': ' '.join(reasoning)
        }
    
    def _ai_analysis(self, historical_data, indicators):
        """
        Use GPT-4o to analyze market data and generate trading signals.
        
        Args:
            historical_data (DataFrame): Historical market data.
            indicators (dict): Pre-calculated indicators.
            
        Returns:
            dict: Strategy result.
        """
        try:
            # Prepare data for the prompt
            recent_data = historical_data.tail(20).reset_index()
            recent_data_str = recent_data.to_string()
            
            # Prepare indicators
            indicator_values = {}
            for name, values in indicators.items():
                if len(values) > 0:
                    # Convert numpy arrays to Python lists for JSON serialization
                    indicator_values[name] = [float(val) for val in values[-5:]]  # Last 5 values
            
            indicator_str = json.dumps(indicator_values, indent=2)
            
            # Symbol from index name or column
            symbol = historical_data.index.name or "SYMBOL"
            
            # Build the prompt
            prompt = f"""
You are a professional trading analyst. Analyze the following market data and technical indicators for {symbol} and provide a trading signal.

RECENT PRICE DATA:
{recent_data_str}

TECHNICAL INDICATORS:
{indicator_str}

Based on this data, determine if there's a trading opportunity. Provide your analysis in JSON format with the following fields:
1. signal: Either "buy", "sell", or "neutral"
2. confidence: A float from 0.0 to 1.0 indicating confidence level
3. reasoning: A detailed explanation of your trading recommendation
4. summary: A brief one-sentence summary of your analysis

Your response should be ONLY the JSON object, with no additional text.
"""
            
            # Call GPT-4o
            response = self.openai.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a professional trading analyst assistant. You provide analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            # Ensure required fields are present
            if 'signal' not in analysis or 'confidence' not in analysis or 'reasoning' not in analysis:
                raise ValueError("Missing required fields in AI analysis")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {
                'signal': 'neutral',
                'confidence': 0.5,
                'reasoning': f"AI analysis failed: {str(e)}",
                'summary': "Unable to perform AI analysis due to an error."
            }
    
    def _combine_strategy_results(self, results):
        """
        Combine results from multiple strategies.
        
        Args:
            results (list): List of strategy results.
            
        Returns:
            dict: Combined result.
        """
        if not results:
            return {
                'signal': 'neutral',
                'confidence': 0.0,
                'reasoning': 'No strategy results available'
            }
        
        # Check if AI analysis is available
        ai_results = [r for r in results if r.get('strategy') == 'ai_analysis']
        
        # If AI analysis is available and confident, use it
        if ai_results and ai_results[0]['confidence'] >= 0.7:
            return ai_results[0]
        
        # Count signals
        buy_count = sum(1 for r in results if r['signal'] == 'buy')
        sell_count = sum(1 for r in results if r['signal'] == 'sell')
        neutral_count = sum(1 for r in results if r['signal'] == 'neutral')
        
        # Calculate weighted confidence
        buy_confidence = sum(r['confidence'] for r in results if r['signal'] == 'buy')
        sell_confidence = sum(r['confidence'] for r in results if r['signal'] == 'sell')
        
        # Determine final signal
        if buy_count > sell_count and buy_count > neutral_count:
            signal = 'buy'
            confidence = buy_confidence / buy_count if buy_count > 0 else 0.5
        elif sell_count > buy_count and sell_count > neutral_count:
            signal = 'sell'
            confidence = sell_confidence / sell_count if sell_count > 0 else 0.5
        else:
            signal = 'neutral'
            confidence = 0.5
        
        # Generate reasoning based on all strategies
        all_reasons = [f"{r.get('strategy', 'Unknown')}: {r['signal']} (conf: {r['confidence']:.2f})" for r in results]
        reasoning = "Combined analysis from multiple strategies: " + "; ".join(all_reasons)
        
        # Generate a summary
        if signal == 'buy':
            summary = f"Bullish signal with {confidence:.0%} confidence based on {len(results)} strategies."
        elif signal == 'sell':
            summary = f"Bearish signal with {confidence:.0%} confidence based on {len(results)} strategies."
        else:
            summary = f"No clear signal from {len(results)} strategies, recommend waiting for stronger indication."
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'summary': summary
        }
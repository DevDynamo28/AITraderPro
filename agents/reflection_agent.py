import json
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

# For OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class ReflectionAgent:
    """
    Agent responsible for periodic reflection on trading performance.
    Uses GPT-4o to analyze trading history and market conditions to improve strategies.
    """
    
    def __init__(self, config, memory_agent):
        """
        Initialize the reflection agent.
        
        Args:
            config (dict): Configuration for the agent.
            memory_agent: MemoryAgent instance for data access.
        """
        self.config = config
        self.memory_agent = memory_agent
        self.has_openai = False
        
        # Try to initialize OpenAI client if API key is available
        if OPENAI_API_KEY:
            try:
                self.openai = OpenAI(api_key=OPENAI_API_KEY)
                self.has_openai = True
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
                logger.warning("Reflection agent will operate in limited mode without AI capabilities")
        else:
            logger.warning("No OpenAI API key found. Reflection agent will operate in limited mode")
        
        # Default periods for reflection (in days)
        self.periods = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30
        }
    
    def reflect_on_recent_trades(self, period='daily'):
        """
        Generate a reflection on recent trading performance.
        
        Args:
            period (str): Period to reflect on ('daily', 'weekly', or 'monthly').
            
        Returns:
            dict: Reflection data including insights and recommendations.
        """
        # Check if reflection for this period already exists and is recent
        existing_reflection = self._get_recent_reflection(period)
        if existing_reflection:
            return existing_reflection
            
        # Get trades for the specified period
        trades = self._get_trades_for_period(period)
        
        # Get market insights for the period
        insights = self._get_insights_for_period(period)
        
        # Get overall statistics
        stats = self.memory_agent.get_trade_statistics()
        
        # Generate reflection using GPT-4o
        reflection_data = self._generate_reflection(trades, insights, stats, period)
        
        # Store the reflection
        self.memory_agent.store_reflection(reflection_data)
        
        return reflection_data
    
    def _get_recent_reflection(self, period):
        """
        Check if a recent reflection already exists for the period.
        
        Args:
            period (str): Period to check ('daily', 'weekly', or 'monthly').
            
        Returns:
            dict: Existing reflection or None.
        """
        latest = self.memory_agent.get_latest_reflection(period)
        
        if not latest:
            return None
            
        # Check if reflection is still fresh
        try:
            reflection_time = datetime.fromisoformat(latest.get('timestamp', ''))
            now = datetime.now()
            
            # Define freshness thresholds by period
            thresholds = {
                'daily': timedelta(hours=12),
                'weekly': timedelta(days=3),
                'monthly': timedelta(days=7)
            }
            
            if now - reflection_time < thresholds.get(period, timedelta(days=1)):
                return latest
        except:
            pass
            
        return None
    
    def _get_trades_for_period(self, period):
        """
        Get trades for the specified period.
        
        Args:
            period (str): Period to get trades for.
            
        Returns:
            list: Trades within the period.
        """
        all_trades = self.memory_agent.get_recent_trades(500)
        
        # Calculate period start time
        days = self.periods.get(period, 1)
        start_time = datetime.now() - timedelta(days=days)
        
        # Filter trades by timestamp
        period_trades = []
        for trade in all_trades:
            try:
                trade_time = datetime.fromisoformat(trade.get('timestamp', ''))
                if trade_time >= start_time:
                    period_trades.append(trade)
            except:
                # Skip trades with invalid timestamps
                continue
                
        return period_trades
    
    def _get_insights_for_period(self, period):
        """
        Get market insights for the specified period.
        
        Args:
            period (str): Period to get insights for.
            
        Returns:
            list: Market insights within the period.
        """
        all_insights = self.memory_agent.get_market_insights(50)
        
        # Calculate period start time
        days = self.periods.get(period, 1)
        start_time = datetime.now() - timedelta(days=days)
        
        # Filter insights by timestamp
        period_insights = []
        for insight in all_insights:
            try:
                insight_time = datetime.fromisoformat(insight.get('timestamp', ''))
                if insight_time >= start_time:
                    period_insights.append(insight)
            except:
                # Skip insights with invalid timestamps
                continue
                
        return period_insights
    
    def _generate_reflection(self, trades, insights, stats, period):
        """
        Generate a reflection using GPT-4o.
        
        Args:
            trades (list): Trades to reflect on.
            insights (list): Market insights for context.
            stats (dict): Overall trading statistics.
            period (str): Period of reflection.
            
        Returns:
            dict: Generated reflection.
        """
        # If no trades or insights for the period, generate a simple reflection
        if not trades:
            return self._generate_empty_reflection(period)
            
        # Check if OpenAI is available
        if not self.has_openai or not hasattr(self, 'openai'):
            logger.warning("OpenAI API is not available. Generating basic reflection.")
            return self._generate_basic_reflection(trades, insights, stats, period)
            
        # Prepare data for the prompt
        trades_json = json.dumps(trades, indent=2)
        insights_json = json.dumps(insights, indent=2)
        stats_json = json.dumps(stats, indent=2)
        
        # Build the prompt
        prompt = f"""
You are a professional trading performance analyst. Your task is to analyze trading data and generate a reflection. The reflection should include insights on performance, market conditions, and specific recommendations for improvement.

Period: {period.capitalize()}

TRADING DATA:
{trades_json}

MARKET INSIGHTS:
{insights_json}

OVERALL STATISTICS:
{stats_json}

Based on this information, generate a detailed trading reflection in JSON format with the following structure:
{{
  "period": "{period}",
  "timestamp": "current ISO timestamp",
  "summary": "A concise summary of trading performance for the period",
  "statistics": {{
    "total_trades": number,
    "win_rate": decimal,
    "total_profit": number,
    "avg_profit": number
  }},
  "market_conditions": "Analysis of market conditions during this period",
  "best_strategy": "Most effective strategy during this period",
  "strengths": ["List of trading strengths"],
  "weaknesses": ["List of trading weaknesses"],
  "opportunities": ["List of trading opportunities"],
  "recommendations": ["List of actionable recommendations for improvement"]
}}

Your response should be ONLY the JSON object, with no additional text.
"""
        
        try:
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
            reflection = json.loads(response.choices[0].message.content)
            
            # Ensure timestamp is present and valid
            if not reflection.get('timestamp'):
                reflection['timestamp'] = datetime.now().isoformat()
                
            # Ensure period is present
            if not reflection.get('period'):
                reflection['period'] = period
                
            return reflection
            
        except Exception as e:
            logger.error(f"Error generating reflection: {str(e)}")
            return self._generate_basic_reflection(trades, insights, stats, period)
            
    def _generate_basic_reflection(self, trades, insights, stats, period):
        """
        Generate a basic reflection when OpenAI is not available.
        
        Args:
            trades (list): Trades to reflect on.
            insights (list): Market insights for context.
            stats (dict): Overall trading statistics.
            period (str): Period of reflection.
            
        Returns:
            dict: Basic reflection based on simple calculations.
        """
        # Calculate basic statistics
        total_trades = len(trades)
        
        # Calculate profit and win rate
        total_profit = 0
        winning_trades = 0
        
        for trade in trades:
            profit = trade.get('profit', 0)
            total_profit += profit
            if profit > 0:
                winning_trades += 1
                
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Determine best strategy with simple heuristic
        strategy_profits = {}
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            profit = trade.get('profit', 0)
            
            if strategy not in strategy_profits:
                strategy_profits[strategy] = {'total': 0, 'count': 0}
                
            strategy_profits[strategy]['total'] += profit
            strategy_profits[strategy]['count'] += 1
            
        best_strategy = 'unknown'
        best_profit = -float('inf')
        
        for strategy, data in strategy_profits.items():
            avg_strategy_profit = data['total'] / data['count'] if data['count'] > 0 else 0
            if avg_strategy_profit > best_profit:
                best_profit = avg_strategy_profit
                best_strategy = strategy
        
        return {
            "period": period,
            "timestamp": datetime.now().isoformat(),
            "summary": f"{total_trades} trades executed with a win rate of {win_rate:.2%} and total profit of {total_profit:.2f}.",
            "statistics": {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "avg_profit": avg_profit
            },
            "market_conditions": "Market analysis not available without OpenAI API.",
            "best_strategy": best_strategy,
            "strengths": [
                "System continued to execute trades as programmed",
                f"Win rate: {win_rate:.2%}"
            ],
            "weaknesses": [
                "Limited analysis capabilities due to missing OpenAI API"
            ],
            "opportunities": [
                "Add OpenAI API key to enable advanced analysis",
                "Consider increasing trade frequency if win rate is above 50%"
            ],
            "recommendations": [
                "Configure your OpenAI API key in environment variables",
                "Review your trading strategies for the next period",
                f"Focus on the {best_strategy} strategy which performed best"
            ]
        }
    
    def _generate_empty_reflection(self, period):
        """
        Generate a simple reflection when no trades are available.
        
        Args:
            period (str): Period of reflection.
            
        Returns:
            dict: Simple reflection.
        """
        return {
            "period": period,
            "timestamp": datetime.now().isoformat(),
            "summary": "No trades were executed during this period.",
            "statistics": {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "avg_profit": 0.0
            },
            "market_conditions": "Insufficient data to analyze market conditions.",
            "best_strategy": "No strategies were employed during this period.",
            "strengths": ["Not enough data to identify strengths."],
            "weaknesses": ["Lack of trading activity."],
            "opportunities": ["Consider setting up more automated trading rules."],
            "recommendations": [
                "Set up at least one active trading strategy.",
                "Configure more aggressive trading parameters.",
                "Consider trading during more active market hours."
            ]
        }
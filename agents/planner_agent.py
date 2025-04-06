import os
import json
import logging
from typing import Dict, List, Any
from openai import OpenAI
from datetime import datetime

class PlannerAgent:
    """
    Agent responsible for breaking down trading goals and planning the overall strategy.
    """
    
    def __init__(self, config):
        """
        Initialize the planner agent.
        
        Args:
            config (dict): Configuration for the agent.
        """
        self.config = config
        self.logger = logging.getLogger('PlannerAgent')
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = config.get('model', 'gpt-4o')  # Use gpt-4o by default
        self.planner_memory = []
        self.max_memory_items = config.get('max_memory_items', 10)
    
    def create_plan(self, market_conditions, account_info, current_positions=None):
        """
        Create a trading plan based on market conditions and account information.
        
        Args:
            market_conditions (dict): Current market conditions.
            account_info (dict): Account information.
            current_positions (dict, optional): Current open positions.
            
        Returns:
            dict: The trading plan.
        """
        try:
            # Format the input for the API
            prompt = self._create_planning_prompt(market_conditions, account_info, current_positions)
            
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            plan = json.loads(response_text)
            
            # Store the plan in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'plan',
                'input': {
                    'market_conditions': market_conditions,
                    'account_info': account_info,
                    'current_positions': current_positions
                },
                'output': plan
            })
            
            return plan
        
        except Exception as e:
            self.logger.error(f"Error creating trading plan: {str(e)}")
            # Return a safe default plan
            return {
                "action": "hold",
                "reason": f"Error in planning: {str(e)}",
                "indicators_to_use": ["rsi", "macd"],
                "risk_assessment": "high",
                "error": True
            }
    
    def evaluate_market_condition(self, market_data):
        """
        Evaluate overall market condition based on various data points.
        
        Args:
            market_data (dict): Market data including indicators.
            
        Returns:
            dict: Market condition evaluation.
        """
        try:
            # Format the input for the API
            prompt = self._create_market_evaluation_prompt(market_data)
            
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency market analyst. Evaluate the market conditions based on the provided data and return a JSON response with your analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            evaluation = json.loads(response_text)
            
            # Store the evaluation in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'market_evaluation',
                'input': market_data,
                'output': evaluation
            })
            
            return evaluation
        
        except Exception as e:
            self.logger.error(f"Error evaluating market condition: {str(e)}")
            # Return a safe default evaluation
            return {
                "market_trend": "unknown",
                "volatility": "unknown",
                "recommendation": "hold",
                "confidence": 0.0,
                "error": True,
                "error_message": str(e)
            }
    
    def adjust_strategy_weights(self, strategies, performance_data):
        """
        Adjust the weights of different strategies based on performance data.
        
        Args:
            strategies (list): List of available strategies.
            performance_data (dict): Performance data for each strategy.
            
        Returns:
            dict: Adjusted weights for each strategy.
        """
        try:
            # Format the input for the API
            prompt = self._create_weight_adjustment_prompt(strategies, performance_data)
            
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert trading system optimizer. Adjust the weights of different trading strategies based on their past performance. Return a JSON object with the strategies and their new weights (0.0 to 1.0) that sum to 1.0."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            adjusted_weights = json.loads(response_text)
            
            # Store the adjustment in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'weight_adjustment',
                'input': {
                    'strategies': strategies,
                    'performance_data': performance_data
                },
                'output': adjusted_weights
            })
            
            return adjusted_weights
        
        except Exception as e:
            self.logger.error(f"Error adjusting strategy weights: {str(e)}")
            # Return equal weights as a fallback
            equal_weight = 1.0 / len(strategies)
            return {strategy: equal_weight for strategy in strategies}
    
    def _get_system_prompt(self):
        """
        Get the system prompt for the planner agent.
        
        Returns:
            str: The system prompt.
        """
        return """You are an expert cryptocurrency trading planner. Your role is to create detailed trading plans based on market conditions and account information. Follow these guidelines:

1. Analyze the provided market data and indicators
2. Consider the account balance and current open positions
3. Recommend appropriate trading strategies and indicators to use
4. Assess risk level and suggest position sizing
5. Provide clear reasoning for your recommendations
6. Always be conservative with risk management
7. Return your response in JSON format with the following structure:
   {
     "action": "buy|sell|hold",
     "reason": "detailed explanation",
     "indicators_to_use": ["list", "of", "indicators"],
     "risk_assessment": "low|medium|high",
     "time_horizon": "short|medium|long",
     "suggested_position_size": "percentage of account",
     "price_targets": {
       "entry": "suggested entry price or range",
       "stop_loss": "suggested stop loss price",
       "take_profit": "suggested take profit price"
     },
     "confidence": "0.0 to 1.0"
   }

Be precise and conservative with your recommendations, prioritizing capital preservation."""
    
    def _create_planning_prompt(self, market_conditions, account_info, current_positions):
        """
        Create a prompt for the planning task.
        
        Args:
            market_conditions (dict): Current market conditions.
            account_info (dict): Account information.
            current_positions (dict, optional): Current open positions.
            
        Returns:
            str: The formatted prompt.
        """
        prompt = f"""Please create a trading plan for BTCUSD based on the following information:

## Market Conditions
```json
{json.dumps(market_conditions, indent=2)}

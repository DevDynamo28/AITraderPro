import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI
from datetime import datetime
import math

class RiskAgent:
    """
    Agent responsible for risk management, setting position sizes, stop losses, and take profits.
    """
    
    def __init__(self, config):
        """
        Initialize the risk agent.
        
        Args:
            config (dict): Configuration for the agent.
        """
        self.config = config
        self.logger = logging.getLogger('RiskAgent')
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = config.get('model', 'gpt-4o')  # Use gpt-4o by default
        self.risk_memory = []
        self.max_memory_items = config.get('max_memory_items', 10)
        
        # Default risk parameters
        self.default_risk_per_trade = config.get('default_risk_per_trade', 0.01)  # 1% of account
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% of account
        self.default_risk_reward_ratio = config.get('default_risk_reward_ratio', 2.0)  # 1:2
    
    def calculate_position_size(self, account_info, signal, symbol_info, adaptive=True):
        """
        Calculate appropriate position size based on account balance and risk parameters.
        
        Args:
            account_info (dict): Account information including balance.
            signal (dict): Trading signal with confidence and price targets.
            symbol_info (dict): Symbol information including tick value and contract size.
            adaptive (bool, optional): Whether to adapt position size based on signal confidence.
            
        Returns:
            dict: Position size information.
        """
        try:
            # Extract necessary information
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', balance)
            
            # Use the smaller of balance or equity for conservative sizing
            available_capital = min(balance, equity)
            
            # Default risk amount (e.g., 1% of account)
            risk_percent = self.default_risk_per_trade
            
            # If adaptive sizing is enabled, adjust based on signal confidence
            if adaptive and 'confidence' in signal:
                confidence = signal.get('confidence', 0.5)
                
                # Scale risk based on confidence (higher confidence = higher risk)
                risk_percent = self.default_risk_per_trade * confidence
                
                # Cap at max risk percentage
                risk_percent = min(risk_percent, self.max_risk_per_trade)
            
            # Calculate risk amount in account currency
            risk_amount = available_capital * risk_percent
            
            # Calculate position size based on stop loss
            entry_price = float(signal.get('price_targets', {}).get('entry', 0))
            stop_loss = float(signal.get('price_targets', {}).get('stop_loss', 0))
            
            # If entry or stop loss are missing or invalid, use a default risk distance
            if entry_price <= 0 or stop_loss <= 0 or abs(entry_price - stop_loss) < 0.000001:
                # Use a default percentage of price as risk (e.g., 1% of price)
                current_price = symbol_info.get('bid', 0) if signal.get('signal') == 'sell' else symbol_info.get('ask', 0)
                if current_price <= 0:
                    self.logger.warning("Invalid price information for position sizing")
                    return {"error": "Invalid price information"}
                
                risk_distance = current_price * 0.01  # 1% of price as default risk distance
            else:
                risk_distance = abs(entry_price - stop_loss)
            
            # Calculate position size
            tick_size = symbol_info.get('trade_tick_size', 0.01)
            tick_value = symbol_info.get('trade_tick_value', 1)
            contract_size = symbol_info.get('trade_contract_size', 1)
            
            if tick_size <= 0 or risk_distance <= 0:
                self.logger.warning("Invalid tick size or risk distance for position sizing")
                return {"error": "Invalid parameters for position sizing calculation"}
            
            # Calculate ticks at risk
            ticks_at_risk = risk_distance / tick_size
            
            # Calculate risk per contract
            risk_per_contract = ticks_at_risk * tick_value
            
            if risk_per_contract <= 0:
                self.logger.warning("Invalid risk per contract for position sizing")
                return {"error": "Invalid risk per contract calculation"}
            
            # Calculate position size in lots
            position_size_raw = risk_amount / risk_per_contract
            
            # Adjust for contract size
            position_size_lots = position_size_raw / contract_size
            
            # Round to valid lot size
            volume_step = symbol_info.get('volume_step', 0.01)
            volume_min = symbol_info.get('volume_min', 0.01)
            volume_max = symbol_info.get('volume_max', 100)
            
            position_size_lots = math.floor(position_size_lots / volume_step) * volume_step
            
            # Ensure position size is within min/max limits
            position_size_lots = max(volume_min, min(position_size_lots, volume_max))
            
            # Calculate notional value
            notional_value = position_size_lots * contract_size * entry_price
            
            # Calculate expected risk and reward
            expected_risk = risk_amount
            expected_reward = self.default_risk_reward_ratio * expected_risk
            
            # Position sizing result
            result = {
                "position_size": position_size_lots,
                "risk_percent": risk_percent * 100,  # Convert to percentage
                "risk_amount": risk_amount,
                "notional_value": notional_value,
                "expected_risk": expected_risk,
                "expected_reward": expected_reward,
                "risk_reward_ratio": self.default_risk_reward_ratio,
            }
            
            # Store in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'position_size_calculation',
                'inputs': {
                    'account_info': account_info,
                    'signal': signal,
                    'symbol_info': symbol_info,
                    'adaptive': adaptive
                },
                'output': result
            })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return {"error": str(e)}
    
    def calculate_risk_parameters(self, account_info, signal, symbol_info, market_conditions):
        """
        Calculate risk parameters including stop loss and take profit levels.
        
        Args:
            account_info (dict): Account information.
            signal (dict): Trading signal with confidence and initial price targets.
            symbol_info (dict): Symbol information.
            market_conditions (dict): Market conditions data.
            
        Returns:
            dict: Risk parameters including optimized stop loss and take profit levels.
        """
        try:
            # If signal already has price targets, use them as a starting point
            price_targets = signal.get('price_targets', {})
            entry_price = price_targets.get('entry', 0)
            stop_loss_initial = price_targets.get('stop_loss', 0)
            take_profit_initial = price_targets.get('take_profit', 0)
            
            # If entry price is missing, use current market price
            if entry_price <= 0:
                entry_price = symbol_info.get('ask', 0) if signal.get('signal') == 'buy' else symbol_info.get('bid', 0)
            
            # If we still don't have a valid entry price, return error
            if entry_price <= 0:
                self.logger.warning("Invalid entry price for risk parameter calculation")
                return {"error": "Invalid entry price"}
            
            # If AI-suggested stops are provided and valid, use them as starting point
            valid_initial_targets = (
                stop_loss_initial > 0 and take_profit_initial > 0 and
                ((signal.get('signal') == 'buy' and stop_loss_initial < entry_price < take_profit_initial) or
                (signal.get('signal') == 'sell' and stop_loss_initial > entry_price > take_profit_initial))
            )
            
            if not valid_initial_targets:
                # Use volatility-based method for initial targets
                atr = market_conditions.get('atr', entry_price * 0.01)  # Default to 1% if ATR not available
                
                if signal.get('signal') == 'buy':
                    stop_loss_initial = entry_price - (atr * 1.5)
                    take_profit_initial = entry_price + (atr * 3.0)
                else:  # sell
                    stop_loss_initial = entry_price + (atr * 1.5)
                    take_profit_initial = entry_price - (atr * 3.0)
            
            # For further optimization, use the LLM to refine the levels
            if self.config.get('use_ai_risk_optimization', True):
                optimized_levels = self._optimize_risk_levels({
                    'account_info': account_info,
                    'signal': signal,
                    'symbol_info': symbol_info,
                    'market_conditions': market_conditions,
                    'initial_levels': {
                        'entry': entry_price,
                        'stop_loss': stop_loss_initial,
                        'take_profit': take_profit_initial
                    }
                })
                
                # Extract the optimized levels
                entry_price = optimized_levels.get('entry', entry_price)
                stop_loss = optimized_levels.get('stop_loss', stop_loss_initial)
                take_profit = optimized_levels.get('take_profit', take_profit_initial)
            else:
                # Use the initial calculations
                stop_loss = stop_loss_initial
                take_profit = take_profit_initial
            
            # Calculate risk metrics
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(entry_price - take_profit)
            risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
            
            # Calculate position size
            position_size = self.calculate_position_size(account_info, 
                                                        {**signal, 'price_targets': {'entry': entry_price, 'stop_loss': stop_loss}}, 
                                                        symbol_info)
            
            # Final risk parameters
            result = {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_distance": risk_distance,
                "reward_distance": reward_distance,
                "risk_reward_ratio": risk_reward_ratio,
                "position_size": position_size.get('position_size', 0),
                "risk_amount": position_size.get('risk_amount', 0),
                "expected_reward": position_size.get('expected_reward', 0)
            }
            
            # Store in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'risk_parameters_calculation',
                'inputs': {
                    'account_info': account_info,
                    'signal': signal,
                    'symbol_info': symbol_info,
                    'market_conditions': market_conditions
                },
                'output': result
            })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {str(e)}")
            return {"error": str(e)}
    
    def evaluate_trade_risk(self, trade_plan, account_info, open_positions, market_conditions):
        """
        Evaluate the overall risk of a trade plan given account and market conditions.
        
        Args:
            trade_plan (dict): The planned trade with position size and risk parameters.
            account_info (dict): Account information.
            open_positions (list): Currently open positions.
            market_conditions (dict): Market conditions data.
            
        Returns:
            dict: Risk evaluation including risk score and recommendations.
        """
        try:
            # Calculate key risk metrics
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', balance)
            available_margin = account_info.get('margin_free', equity)
            
            # Calculate exposure from open positions
            total_exposure = sum(pos.get('volume', 0) * pos.get('price_current', 0) for pos in open_positions) if open_positions else 0
            
            # Calculate new trade exposure
            new_trade_size = trade_plan.get('position_size', 0)
            entry_price = trade_plan.get('entry_price', 0)
            new_trade_exposure = new_trade_size * entry_price
            
            # Calculate risk metrics
            total_exposure_with_new_trade = total_exposure + new_trade_exposure
            exposure_to_equity_ratio = total_exposure_with_new_trade / equity if equity > 0 else 999
            
            # Risk from correlated positions
            # Count positions in same direction for the same symbol
            symbol = trade_plan.get('symbol', '')
            direction = trade_plan.get('signal', '')
            correlated_positions = [pos for pos in open_positions 
                                   if pos.get('symbol', '') == symbol and 
                                   (pos.get('type', 0) == 0 and direction == 'buy' or 
                                    pos.get('type', 0) == 1 and direction == 'sell')]
            
            correlation_risk = len(correlated_positions) / 10  # Scale from 0 to 1
            
            # Calculate volatility risk
            market_volatility = market_conditions.get('volatility', 0.01)
            volatility_risk = market_volatility / 0.05  # Scale to 1 for 5% volatility
            
            # Calculate drawdown risk
            account_drawdown = 1 - (equity / balance) if balance > 0 else 0
            drawdown_risk = account_drawdown / 0.1  # Scale to 1 for 10% drawdown
            
            # Calculate overall risk score (0-100, higher is riskier)
            risk_score = min(100, max(0, 20 * exposure_to_equity_ratio + 
                                     15 * correlation_risk + 
                                     25 * volatility_risk + 
                                     40 * drawdown_risk))
            
            # Determine risk level
            risk_level = "low" if risk_score < 30 else "medium" if risk_score < 60 else "high"
            
            # Generate recommendations
            recommendations = []
            
            if exposure_to_equity_ratio > 2:
                recommendations.append("High overall exposure. Consider reducing position size.")
            
            if correlation_risk > 0.3:
                recommendations.append("Multiple positions on same symbol. Consider diversifying.")
            
            if volatility_risk > 0.8:
                recommendations.append("High market volatility. Consider wider stop loss.")
            
            if drawdown_risk > 0.7:
                recommendations.append("Significant account drawdown. Consider trading smaller size.")
            
            if risk_score > 70:
                recommendations.append("Overall risk is too high. Reconsider trade or reduce position size.")
            
            # Format the result
            result = {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "exposure_ratio": exposure_to_equity_ratio,
                "correlation_risk": correlation_risk,
                "volatility_risk": volatility_risk,
                "drawdown_risk": drawdown_risk,
                "recommendations": recommendations,
                "proceed": risk_score < 80  # Flag if the trade should proceed
            }
            
            # Store in memory
            self._add_to_memory({
                'timestamp': datetime.now().isoformat(),
                'type': 'trade_risk_evaluation',
                'inputs': {
                    'trade_plan': trade_plan,
                    'account_info': account_info,
                    'open_positions': open_positions,
                    'market_conditions': market_conditions
                },
                'output': result
            })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error evaluating trade risk: {str(e)}")
            return {
                "risk_score": 100,
                "risk_level": "unknown",
                "recommendations": [f"Error in risk evaluation: {str(e)}"],
                "proceed": False,
                "error": str(e)
            }
    
    def _optimize_risk_levels(self, input_data):
        """
        Use the LLM to optimize entry, stop loss, and take profit levels.
        
        Args:
            input_data (dict): Input data including initial levels and market conditions.
            
        Returns:
            dict: Optimized price levels.
        """
        try:
            # Format the input for the API
            prompt = self._create_risk_optimization_prompt(input_data)
            
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert risk manager for cryptocurrency trading. Your task is to optimize entry, stop loss, and take profit levels based on market conditions and technical analysis. Focus on preserving capital while maintaining a good risk-reward ratio."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            optimized_levels = json.loads(response_text)
            
            return optimized_levels.get('optimized_levels', {})
        
        except Exception as e:
            self.logger.error(f"Error optimizing risk levels: {str(e)}")
            return input_data.get('initial_levels', {})
    
    def _create_risk_optimization_prompt(self, input_data):
        """
        Create a prompt for the risk level optimization task.
        
        Args:
            input_data (dict): Input data including initial levels and market conditions.
            
        Returns:
            str: The formatted prompt.
        """
        signal = input_data.get('signal', {})
        market_conditions = input_data.get('market_conditions', {})
        initial_levels = input_data.get('initial_levels', {})
        
        prompt = f"""Please optimize the entry, stop loss, and take profit levels for a {signal.get('signal', 'trade')} trade based on the following information:

## Initial Price Levels
```json
{json.dumps(initial_levels, indent=2)}

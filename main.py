import os
import sys
import yaml
import logging
import time
import json
import threading
import schedule
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Try to import MetaTrader5, but handle if it's not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 package not available. Running in simulation mode.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utility modules
from utils.mt5_connector import MT5Connector
from utils.market_data import MarketDataUtil

# Import stream module
from stream.live_feed import LiveFeed

# Import agent modules
from agents.memory_agent import MemoryAgent
from agents.reflection_agent import ReflectionAgent
from agents.strategy_agent import StrategyAgent

# Define Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "agentic-ai-trader-secret")

# Global variables
is_running = False
scheduler_thread_obj = None

# Initialize configuration
def load_config():
    """Load configuration from YAML file."""
    try:
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        # Return default configuration
        return {
            'mt5': {
                'server': 'MetaQuotes-Demo',
                'login': None,
                'password': None,
                'timeout': 60000,
                'simulation': not MT5_AVAILABLE
            },
            'trading': {
                'symbol': 'BTCUSD',
                'timeframe': 'M5',
                'risk_per_trade': 0.01,
                'enable_auto_trading': False,
                'max_open_positions': 3
            },
            'indicators': {
                'rsi': {'length': 14, 'overbought': 70, 'oversold': 30},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            'openai': {
                'model': 'gpt-4o',  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                'temperature': 0.2
            }
        }

# Load configuration
config = load_config()

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
os.makedirs('data/logs', exist_ok=True)

# Initialize MT5 connector
mt5_connector = MT5Connector(config['mt5'])

# Initialize market data utilities
market_data_util = MarketDataUtil(mt5_connector)

# Initialize memory agent
memory_agent = MemoryAgent(config)

# Initialize reflectoin agent
reflection_agent = ReflectionAgent(config, memory_agent)

# Initialize strategy agent
strategy_agent = StrategyAgent(config, market_data_util)

# Initialize live feed
live_feed = LiveFeed(config, mt5_connector)

# Initialize MT5 connection
def initialize_mt5():
    """Initialize MT5 connection."""
    if mt5_connector.initialize():
        logger.info("MT5 connection initialized successfully")
        return True
    else:
        logger.warning("MT5 connection failed, running in simulation mode")
        return False

# Trading system functions
def perform_market_analysis():
    """Perform market analysis using strategy agent."""
    try:
        symbol = config['trading']['symbol']
        timeframe = config['trading']['timeframe']
        
        # Get market analysis
        analysis = strategy_agent.analyze_market(symbol, timeframe)
        
        # Store analysis in memory
        if analysis and memory_agent:
            # Use appropriate method based on what's implemented in MemoryAgent
            if hasattr(memory_agent, 'store_market_insight'):
                memory_agent.store_market_insight({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis
                })
            elif hasattr(memory_agent, 'store_analysis'):
                memory_agent.store_analysis({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis
                })
        
        logger.info(f"Market analysis completed for {symbol} {timeframe}")
        return analysis
    except Exception as e:
        logger.error(f"Error in market analysis: {str(e)}")
        return None

def on_price_update(symbol, price_data):
    """Callback function for price updates."""
    # Currently just for demonstration
    # This could trigger actions on price movements
    pass

def setup_scheduler():
    """Set up scheduled tasks."""
    schedule.clear()
    
    # Schedule market analysis
    analysis_interval = config.get('scheduler', {}).get('analysis_interval', 5)
    schedule.every(analysis_interval).minutes.do(perform_market_analysis)
    
    # Schedule reflection
    reflection_interval = config.get('scheduler', {}).get('reflection_interval', 60)
    schedule.every(reflection_interval).minutes.do(
        lambda: reflection_agent.reflect_on_recent_trades(period='daily')
    )
    
    logger.info("Scheduler setup complete")

def scheduler_thread():
    """Thread function for running scheduled tasks."""
    logger.info("Scheduler thread started")
    while is_running:
        schedule.run_pending()
        time.sleep(1)

def start_system():
    """Start the trading system."""
    global is_running, scheduler_thread_obj
    
    if is_running:
        logger.warning("System is already running")
        return False
    
    # Initialize MT5 connection
    initialize_mt5()
    
    # Start price streaming
    if not live_feed.start():
        logger.error("Failed to start price streaming")
        return False
    
    # Register price callback
    live_feed.register_price_callback(on_price_update)
    
    # Setup scheduler
    setup_scheduler()
    
    # Start scheduler thread
    is_running = True
    scheduler_thread_obj = threading.Thread(target=scheduler_thread)
    scheduler_thread_obj.daemon = True
    scheduler_thread_obj.start()
    
    logger.info("Trading system started")
    return True

def stop_system():
    """Stop the trading system."""
    global is_running
    
    if not is_running:
        logger.warning("System is not running")
        return False
    
    # Stop the scheduler thread
    is_running = False
    
    # Stop price streaming
    live_feed.stop()
    
    # Shutdown MT5 connection
    mt5_connector.shutdown()
    
    logger.info("Trading system stopped")
    return True

# Flask routes
@app.route('/')
def index():
    """Render the main dashboard."""
    account_info = mt5_connector.get_account_info() if mt5_connector.initialized else None
    symbol_info = mt5_connector.get_symbol_info(config['trading']['symbol']) if mt5_connector.initialized else None
    open_positions = []  # We'll implement this when the ExecutorAgent is implemented
    
    # Get trade statistics
    trade_stats = memory_agent.get_trade_statistics()
    
    return render_template(
        'index.html',
        system_status=is_running,
        account_info=account_info,
        symbol_info=symbol_info,
        open_positions=open_positions,
        config=config,
        trade_stats=trade_stats,
        MT5_AVAILABLE=MT5_AVAILABLE
    )

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Perform and display market analysis."""
    if request.method == 'POST':
        # Get parameters from form
        symbol = request.form.get('symbol', config['trading']['symbol'])
        timeframe = request.form.get('timeframe', config['trading']['timeframe'])
        
        # Perform analysis
        analysis = strategy_agent.analyze_market(symbol, timeframe)
        
        return render_template(
            'analysis.html',
            symbol=symbol,
            timeframe=timeframe,
            analysis=analysis,
            config=config
        )
    
    # GET request - show analysis form
    return render_template(
        'analysis.html',
        symbol=config['trading']['symbol'],
        timeframe=config['trading']['timeframe'],
        analysis=None,
        config=config
    )

@app.route('/history')
def history():
    """View trading history."""
    # Get trade history
    trades = memory_agent.get_recent_trades(50)
    
    # Get overall statistics
    stats = memory_agent.get_trade_statistics()
    
    return render_template('history.html', trades=trades, stats=stats)

@app.route('/reflection')
def reflection():
    """View trading reflections."""
    period = request.args.get('period', 'daily')
    
    # Perform reflection
    reflection_data = reflection_agent.reflect_on_recent_trades(period=period)
    
    return render_template('reflection.html', reflection=reflection_data, period=period)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """View and update system settings."""
    global config
    
    if request.method == 'POST':
        # Update settings from form
        try:
            # MT5 settings
            config['mt5']['server'] = request.form.get('mt5_server', config['mt5']['server'])
            config['mt5']['login'] = request.form.get('mt5_login', None)
            if request.form.get('mt5_password'):
                config['mt5']['password'] = request.form.get('mt5_password')
            
            # Trading settings
            config['trading']['symbol'] = request.form.get('trading_symbol', config['trading']['symbol'])
            config['trading']['timeframe'] = request.form.get('trading_timeframe', config['trading']['timeframe'])
            config['trading']['risk_per_trade'] = float(request.form.get('risk_per_trade', config['trading']['risk_per_trade']))
            config['trading']['enable_auto_trading'] = 'enable_auto_trading' in request.form
            
            # Indicator settings
            config['indicators']['rsi']['length'] = int(request.form.get('rsi_length', config['indicators']['rsi']['length']))
            config['indicators']['rsi']['overbought'] = int(request.form.get('rsi_overbought', config['indicators']['rsi']['overbought']))
            config['indicators']['rsi']['oversold'] = int(request.form.get('rsi_oversold', config['indicators']['rsi']['oversold']))
            config['indicators']['macd']['fast'] = int(request.form.get('macd_fast', config['indicators']['macd']['fast']))
            config['indicators']['macd']['slow'] = int(request.form.get('macd_slow', config['indicators']['macd']['slow']))
            config['indicators']['macd']['signal'] = int(request.form.get('macd_signal', config['indicators']['macd']['signal']))
            
            # Save the configuration
            with open('config.yaml', 'w') as config_file:
                yaml.dump(config, config_file, default_flow_style=False)
            
            # Restart system if running
            if is_running:
                stop_system()
                start_system()
            
            return redirect(url_for('settings', updated=True))
        
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return render_template('settings.html', config=config, error=str(e))
    
    # GET request - show settings form
    return render_template('settings.html', config=config, updated=request.args.get('updated', False))

@app.route('/api/prices')
def api_prices():
    """API endpoint for current prices."""
    symbol = request.args.get('symbol', config['trading']['symbol'])
    
    # Get price data
    price = live_feed.get_last_price(symbol)
    
    if price:
        return jsonify(price)
    else:
        return jsonify({'error': 'Price not available'})

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    mt5_connected = mt5_connector.initialized
    
    # Get basic status info
    status = {
        'running': is_running,
        'mt5_connected': mt5_connected,
        'auto_trading': config['trading'].get('enable_auto_trading', False),
        'symbol': config['trading']['symbol'],
        'timeframe': config['trading']['timeframe']
    }
    
    # Add account info if connected
    if mt5_connected:
        account_info = mt5_connector.get_account_info()
        if account_info:
            status['account'] = account_info
    
    return jsonify(status)

@app.route('/control/start', methods=['POST'])
def control_start():
    """Start the trading system."""
    result = start_system()
    return jsonify({'success': result})

@app.route('/control/stop', methods=['POST'])
def control_stop():
    """Stop the trading system."""
    result = stop_system()
    return jsonify({'success': result})

@app.route('/control/reset', methods=['POST'])
def control_reset():
    """Reset the trading system (stop and start)."""
    stop_system()
    result = start_system()
    return jsonify({'success': result})

# Create default config file if it doesn't exist
if not os.path.exists('config.yaml'):
    with open('config.yaml', 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

# Initialize the system when the module loads
if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
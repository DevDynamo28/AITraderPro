#!/usr/bin/env python3
import logging
import yaml
from utils.mt5_connector import MT5Connector
from stream.live_feed import LiveFeed

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_mt5_connector')

# Load config
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Config loaded successfully")
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    config = {
        'mt5': {
            'simulation': True,
            'login': '12345',
            'password': 'password',
            'server': 'Demo'
        },
        'stream': {
            'symbols': ['BTCUSD', 'ETHUSD', 'XRPUSD'],
            'update_interval': 1.0
        }
    }

# Test MT5 connector
logger.info("Testing MT5Connector with simulation=True...")
mt5_config_sim = config.get('mt5', {})
mt5_config_sim['simulation'] = True  # Ensure simulation is enabled
mt5_connector_sim = MT5Connector(mt5_config_sim)

# Initialize and display status
sim_init_result = mt5_connector_sim.initialize()
logger.info(f"Simulation mode initialization: {sim_init_result}")
logger.info(f"Simulation mode: {mt5_connector_sim.simulation_mode}")

# Test getting symbol info in simulation mode
btc_info = mt5_connector_sim.get_symbol_info('BTCUSD')
logger.info(f"BTCUSD info in simulation mode: {btc_info}")

# Test LiveFeed
logger.info("\nTesting LiveFeed with simulation mode...")
live_feed = LiveFeed(config.get('stream', {}), mt5_connector_sim)
logger.info(f"LiveFeed simulation mode: {live_feed.simulation_mode}")

# Start live feed (should fallback to simulation even if MT5 not available)
start_result = live_feed.start()
logger.info(f"LiveFeed start result: {start_result}")

# Get some price updates
import time
logger.info("Waiting for price updates...")
time.sleep(3)

# Get last prices 
prices = live_feed.get_all_prices()
logger.info(f"Current prices: {prices}")

# Clean up
live_feed.stop()
mt5_connector_sim.shutdown()
logger.info("Test completed.")
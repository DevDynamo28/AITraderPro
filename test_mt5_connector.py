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

# Test MT5 connector with real credentials
import os
logger.info("Checking for MT5 environment variables...")
mt5_login = os.environ.get('MT5_LOGIN')
mt5_password = os.environ.get('MT5_PASSWORD')
mt5_server = os.environ.get('MT5_SERVER')
mt5_api_url = os.environ.get('MT5_API_URL')
mt5_api_key = os.environ.get('MT5_API_KEY')

logger.info(f"MT5_LOGIN available: {mt5_login is not None}")
logger.info(f"MT5_PASSWORD available: {mt5_password is not None}")
logger.info(f"MT5_SERVER available: {mt5_server is not None}")
logger.info(f"MT5_API_URL available: {mt5_api_url is not None}")
logger.info(f"MT5_API_KEY available: {mt5_api_key is not None}")

# Load configuration from environment
mt5_config = config.get('mt5', {})
mt5_config['simulation'] = False  # Try to use real MT5 connection
mt5_config['login'] = mt5_login
mt5_config['password'] = mt5_password
mt5_config['server'] = mt5_server
mt5_config['api_url'] = mt5_api_url
mt5_config['api_key'] = mt5_api_key
mt5_connector_sim = MT5Connector(mt5_config)

# First test MT5APIConnector directly
from utils.mt5_api_connector import MT5APIConnector
logger.info("\nTesting MT5APIConnector directly...")
mt5_api = MT5APIConnector(mt5_config)
api_init_result = mt5_api.initialize()
logger.info(f"MT5APIConnector initialization: {api_init_result}")
logger.info(f"MT5APIConnector simulation mode: {mt5_api.simulation_mode}")

# Test getting account info through API
account_info = mt5_api.get_account_info()
logger.info(f"Account info: {account_info}")

# Test getting BTCUSD info through API
btc_api_info = mt5_api.get_symbol_info('BTCUSD')
logger.info(f"BTCUSD info via API: {btc_api_info}")

# Test LiveFeed with the API connector
logger.info("\nTesting LiveFeed with MT5APIConnector...")
api_live_feed = LiveFeed(config.get('stream', {}), mt5_api)
api_feed_start = api_live_feed.start()
logger.info(f"LiveFeed with API start result: {api_feed_start}")
logger.info(f"LiveFeed with API simulation mode: {api_live_feed.simulation_mode}")

# Get some API price updates
import time
logger.info("Waiting for API price updates...")
time.sleep(3)
api_prices = api_live_feed.get_all_prices()
logger.info(f"API Prices: {api_prices}")

# Stop API Live Feed
api_live_feed.stop()
mt5_api.shutdown()
logger.info("API test completed.")

# Now test the regular MT5Connector

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
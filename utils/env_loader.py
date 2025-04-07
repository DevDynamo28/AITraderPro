"""
Environment variable loader utility.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_env_from_file(env_file='.env'):
    """
    Load environment variables from a .env file if it exists.
    Only sets variables that aren't already in the environment.
    
    Args:
        env_file (str): Path to the .env file.
    """
    try:
        # Check if the file exists
        env_path = Path(env_file)
        if not env_path.exists():
            logger.warning(f".env file not found at {env_path.absolute()}. Environment variables must be set manually.")
            return False
            
        # Read and parse the file
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Parse key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
                        logger.debug(f"Loaded environment variable: {key}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error loading .env file: {str(e)}")
        return False

def ensure_api_keys():
    """
    Check if necessary API keys are in the environment.
    """
    missing_keys = []
    
    # Check for OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        logger.warning("OpenAI API key (OPENAI_API_KEY) not found in environment variables.")
        missing_keys.append('OPENAI_API_KEY')
    
    # Check for MT5 credentials
    if not os.environ.get('MT5_SERVER'):
        logger.warning("MT5 server address (MT5_SERVER) not found in environment variables.")
        missing_keys.append('MT5_SERVER')
        
    if not os.environ.get('MT5_LOGIN'):
        logger.warning("MT5 login (MT5_LOGIN) not found in environment variables.")
        missing_keys.append('MT5_LOGIN')
        
    if not os.environ.get('MT5_PASSWORD'):
        logger.warning("MT5 password (MT5_PASSWORD) not found in environment variables.")
        missing_keys.append('MT5_PASSWORD')
    
    return missing_keys

def print_env_setup_instructions():
    """
    Print instructions for setting up environment variables.
    """
    instructions = """
    ============================================================
    MISSING ENVIRONMENT VARIABLES
    ============================================================
    To use all features of this application, please set the 
    following environment variables:
    
    1. Create a file named '.env' in the project root directory
    
    2. Add the following lines to the file, replacing the values 
       with your actual API keys and credentials:
       
       OPENAI_API_KEY=your_openai_api_key_here
       MT5_SERVER=your_mt5_server_address_here
       MT5_LOGIN=your_mt5_login_here
       MT5_PASSWORD=your_mt5_password_here
       
    3. Restart the application after creating the .env file
    
    For local development, you can use a .env file as shown above.
    For deployment, set these as environment variables in your 
    hosting environment.
    ============================================================
    """
    print(instructions)
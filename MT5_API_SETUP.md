# Setting Up Real MT5 API Connection

This document explains how to connect your Agentic AI Trader with a real MetaTrader 5 instance to get live market data and execute trades.

## Current Status

The system has been designed to connect to a real MT5 instance through a REST API. When a real MT5 API server isn't available, the system falls back to simulation mode automatically, providing realistic price simulations for testing purposes.

Currently, you'll see `"simulated": true` in the price data and account information, which indicates that the system is running in simulation mode.

## Setup Options for Real MT5 Connection

To connect to real MT5 data, you have several options:

### Option 1: Use a broker-provided MT5 API

Some brokers provide REST API access to their MT5 servers. If your broker offers this service:

1. Get your API endpoint URL from your broker
2. Update your `.env` file with:
   ```
   MT5_API_URL=https://your-brokers-mt5-api.com
   MT5_LOGIN=your_login_number
   MT5_PASSWORD=your_password
   MT5_SERVER=your_broker_server_name
   ```

### Option 2: Set up a bridge server

You can create a lightweight MT5 API server that acts as a bridge between your MT5 terminal and this application:

1. Install MT5 on a Windows machine or server
2. Install the "MT5 REST API Bridge" (available from several providers)
3. Configure the bridge to expose MT5 data over HTTP/HTTPS
4. Update your `.env` file to point to your bridge server

### Option 3: Use MT5 Web API

MetaQuotes provides an official Web API for MT5. You can set up this connection by:

1. Register for MT5 Web API access through MetaQuotes
2. Generate API keys 
3. Update your configuration to use the official Web API endpoints

## Configuration

Once you have your MT5 API endpoint, update your configuration:

1. In the web interface: Go to Settings and enter your MT5 login, password, and server details
2. Or update your environment variables:
   ```
   MT5_API_URL=https://your-mt5-api-server.com
   MT5_LOGIN=your_login_number
   MT5_PASSWORD=your_password
   MT5_SERVER=your_broker_server_name
   ```

## Verification

To verify your connection:

1. Check the dashboard for "Simulation: false" status
2. Look for real price data with "simulated: false" flag
3. Check the API status at `/api/status` to confirm "mt5_connected": true

## Troubleshooting

If you're having trouble connecting:

1. Check your MT5 credentials (login, password, server)
2. Verify the API URL is correct and accessible
3. Ensure your MT5 server allows API connections
4. Check if your broker restricts API access to specific IPs
5. Review the logs for specific error messages

If you need assistance setting up your MT5 API connection, contact your broker's support team for their specific API documentation and requirements.

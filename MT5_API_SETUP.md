# Setting Up Direct MetaTrader 5 Connection

This document explains how to connect your Agentic AI Trader directly with your local MetaTrader 5 installation to get live market data and execute trades.

## Current Status

The system has been designed to connect directly to a locally installed MetaTrader 5 terminal. When MT5 is not available or cannot be initialized, the system falls back to simulation mode automatically, providing realistic price simulations for testing purposes.

When you see `"simulated": true` in the price data and account information, it indicates that the system is running in simulation mode.

## Setup for Direct MT5 Connection

To connect to your locally installed MT5:

### Prerequisites

1. **Install MetaTrader 5**: Download and install the official MetaTrader 5 terminal from your broker's website or the [official MetaTrader website](https://www.metatrader5.com/en/download).

2. **Create a Trading Account**: Set up a live or demo account with your broker.

3. **Log in to MT5**: Open MetaTrader 5 and log in with your account credentials.

### Configuration

The system will automatically attempt to connect to your running MT5 terminal. If MT5 is not already running, the system will try to start it automatically.

To ensure proper connection:

1. Make sure your MetaTrader 5 terminal is installed in one of these common locations:
   - `C:\Program Files\MetaTrader 5\terminal64.exe`
   - `C:\Program Files (x86)\MetaTrader 5\terminal.exe`
   - `C:\Program Files\MetaTrader 5\metatrader.exe`

2. Optionally, you can provide your credentials in the `.env` file:
   ```
   MT5_LOGIN=your_login_number
   MT5_PASSWORD=your_password
   MT5_SERVER=your_broker_server_name
   ```

3. If your MT5 terminal is already logged in (recommended), the system will use the active session without requiring credentials.

### Auto-Start Feature

The system includes an auto-start feature that will:

1. Check if MT5 is already running
2. If not, attempt to launch the MetaTrader 5 terminal
3. Wait for the terminal to initialize
4. Connect to the running terminal

This feature is enabled by default but can be disabled in the settings.

## Verification

To verify your connection:

1. Check the dashboard for "Simulation: false" status
2. Look for real price data with "simulated: false" flag
3. Check the API status at `/api/status` to confirm "mt5_connected": true
4. Verify that your MT5 terminal is running

## Troubleshooting

If you're having trouble connecting:

1. Make sure MetaTrader 5 is installed on your system
2. Check that your MT5 terminal is running and logged in
3. If using credentials, verify they are correct (login, password, server)
4. Ensure your MT5 terminal allows script/API connections (enabled in Tools > Options > Expert Advisors)
5. Check if MT5 is installed in a non-standard location (if so, update the code or move it to a standard location)
6. Review the application logs for specific error messages

## Advanced: Windows vs. Other Operating Systems

The auto-start feature works best on Windows systems where MetaTrader 5 is natively supported. If you're running on:

- **Windows**: The system should automatically find and start your MT5 terminal
- **macOS**: You'll need to run MT5 through a Windows virtualization solution
- **Linux**: You'll need to run MT5 through Wine or a Windows virtualization solution

If running on a non-Windows system, ensure MT5 is already running before starting the trading system.

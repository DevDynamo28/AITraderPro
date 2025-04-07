# Agentic AI Trader for MetaTrader 5

An intelligent trading bot that leverages GPT-4 and LangChain to analyze BTCUSD on MetaTrader 5, implement trading strategies, manage risk, and execute trades.

## 🚀 Features

- **MetaTrader 5 Integration**: Connect to MT5 for real-time data and trade execution
- **AI-Powered Analysis**: Use GPT-4 to analyze market data and make trading decisions
- **Multi-Agent Architecture**: Specialized agents for planning, strategy, risk, execution, memory, and reflection
- **Technical Indicators**: RSI and MACD strategy implementation with adaptive parameters
- **Risk Management**: Intelligent position sizing and stop-loss/take-profit calculation
- **One-Click Trading**: Execute trades quickly with pre-configured risk profiles
- **Live Price Streaming**: Real-time BTCUSD price updates
- **Learning & Adaptation**: Improve strategies based on past performance
- **Web Dashboard**: Monitor your trading and control the system through a web interface

## 📋 Requirements

- Python 3.9+
- MetaTrader 5 installed
- OpenAI API key
- Required Python packages:
  - openai
  - langchain
  - python-dotenv
  - MetaTrader5
  - pandas
  - pandas-ta
  - chromadb
  - flask
  - schedule
  - pyyaml
  - websockets

## 🔧 Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/agentic-ai-trader-mt5.git
   cd agentic-ai-trader-mt5
   ```

2. Install dependencies:
   ```
   pip install openai langchain python-dotenv MetaTrader5 pandas pandas-ta chromadb flask schedule pyyaml websockets
   ```

3. Set up your environment variables by copying `.env.example` to `.env` and filling in your credentials:
   ```
   # Copy the example file
   cp .env.example .env
   
   # Edit the .env file with your actual API keys and credentials
   nano .env   # or use your preferred text editor
   ```
   
   Required environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key for AI analysis
   - `MT5_SERVER`: Your MetaTrader 5 server address
   - `MT5_LOGIN`: Your MetaTrader 5 account login
   - `MT5_PASSWORD`: Your MetaTrader 5 account password

4. Configure the system by editing `config.yaml` with your MetaTrader 5 credentials and trading preferences.

## 🔌 Connecting to MetaTrader 5

1. Install and set up MetaTrader 5 on your computer.
2. Ensure that you have enabled "Allow automated trading" in MT5 settings:
   - Tools → Options → Expert Advisors → Check "Allow automated trading"
   - Also ensure "Allow WebRequest for listed URL" is checked if needed
3. Configure your MT5 login credentials in `config.yaml`
4. The bot will automatically connect to MT5 when started

## 🏃‍♂️ Running the Bot

### Manual Execution

Start the bot with:

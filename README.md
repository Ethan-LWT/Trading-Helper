# 🚀 Trading AI Dashboard

A comprehensive AI-powered trading platform that combines multiple data sources, advanced technical analysis, and artificial intelligence to generate and optimize trading strategies.

## ✨ Features

### 📊 **Multi-Source Data Integration**
- **Real-time Stock Data**: Support for US, HK, and CN markets
- **Multiple Data Providers**: yfinance, Alpha Vantage, Polygon.io, Twelve Data, Finnhub
- **Intraday Data**: 1m, 5m, 15m, 30m, 1h timeframes with 28+ technical indicators
- **Smart Caching**: Intelligent caching system to reduce API calls and improve performance

### 🤖 **AI-Powered Strategy Generation**
- **Local AI Support**: Integration with Ollama (Llama3) for local AI processing
- **Google AI Integration**: Advanced strategy generation using Google's Gemini AI
- **Multi-Timeframe Strategies**: Coordinated trading across daily, 3-day, weekly, bi-weekly, and monthly timeframes
- **Intelligent Optimization**: AI-driven strategy optimization based on backtest results

### 📈 **Advanced Technical Analysis**
- **28+ Technical Indicators**: RSI, MACD, Bollinger Bands, VWAP, and more
- **Chip Distribution Analysis**: Support and resistance level detection
- **Volume Analysis**: Abnormal volume detection and monitoring
- **Market Sentiment**: Multi-factor sentiment analysis

### 🔄 **Comprehensive Backtesting**
- **Historical Backtesting**: Test strategies against historical data
- **AI Backtesting**: Advanced backtesting with AI-generated strategies
- **Performance Metrics**: Detailed analysis including Sharpe ratio, max drawdown, win rate
- **Multi-Timeframe Backtesting**: Coordinated backtesting across multiple timeframes

### 🌐 **Modern Web Interface**
- **Responsive Design**: Modern, dark-themed UI optimized for trading
- **Real-time Updates**: Live price updates and monitoring
- **Interactive Charts**: Advanced charting with multiple timeframes
- **Multi-language Support**: English and Chinese language switching
- **Mobile Friendly**: Responsive design for all devices

### ⚡ **Real-time Monitoring**
- **Volume Alerts**: Abnormal trading volume detection
- **Price Monitoring**: Real-time price change notifications
- **Strategy Monitoring**: Live strategy performance tracking
- **Push Notifications**: Configurable alert system

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Node.js (for some frontend dependencies)
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ethan-LWT/Trading-Helper.git
   cd Trading_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file and add your API keys:
   ```env
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   GOOGLE_API_KEY=your_google_ai_api_key
   TUSHARE_TOKEN=your_tushare_token
   # Add other API keys as needed
   ```

4. **Run the application**
   ```bash
   cd web
   python app.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to `http://localhost:5000`

## 🔑 API Keys Setup

### Required API Keys

#### Alpha Vantage (Required for basic functionality)
- **Purpose**: Stock market data
- **Get API Key**: [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- **Free Tier**: 5 API requests per minute, 500 requests per day

#### Google AI (Required for AI features)
- **Purpose**: AI strategy generation and optimization
- **Get API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Free Tier**: Generous free quota for testing

### Optional API Keys

#### Polygon.io
- **Purpose**: Enhanced real-time and historical data
- **Get API Key**: [Polygon.io](https://polygon.io/)
- **Free Tier**: Limited requests per minute

#### Finnhub
- **Purpose**: Additional market data and news
- **Get API Key**: [Finnhub](https://finnhub.io/)
- **Free Tier**: 60 API calls/minute

#### Twelve Data
- **Purpose**: Alternative data source
- **Get API Key**: [Twelve Data](https://twelvedata.com/)
- **Free Tier**: 800 requests per day

#### Tushare (For Chinese Markets)
- **Purpose**: Chinese stock market data
- **Get Token**: [Tushare Pro](https://tushare.pro/register)
- **Free Tier**: Limited daily requests

## 🏗️ Project Structure

```
Trading_AI/
├── ai_models/              # AI model implementations
│   ├── google_ai.py       # Google AI integration
│   └── local_ai.py        # Local Ollama integration
├── backtest/               # Backtesting engines
│   ├── backtester.py      # Basic backtesting
│   └── ai_backtester.py   # AI-powered backtesting
├── config/                 # Configuration files
│   └── config.py          # API keys and settings
├── data/                   # Data management
│   ├── data_fetcher.py    # Basic data fetching
│   ├── enhanced_intraday_fetcher.py  # Advanced intraday data
│   ├── multi_source_manager.py       # Multi-source data management
│   └── cache_manager.py   # Caching system
├── strategy/               # Trading strategies
│   ├── ai_strategy_generator.py      # AI strategy generation
│   ├── multi_timeframe_strategy.py   # Multi-timeframe strategies
│   └── strategy_optimizer.py         # Strategy optimization
├── web/                    # Web interface
│   ├── app.py             # Flask application
│   ├── templates/         # HTML templates
│   └── static/            # CSS, JS, and assets
├── utils/                  # Utility functions
├── scrapers/              # Web scraping utilities
└── requirements.txt       # Python dependencies
```

## 🚀 Usage

### Basic Usage

1. **Add Stocks to Watchlist**
   - Click the "+" button in the watchlist
   - Enter stock symbol (e.g., AAPL, TSLA, GOOGL)
   - Select market (US, HK, CN)

2. **View Real-time Data**
   - Select a stock from the watchlist
   - View real-time price, volume, and technical indicators
   - Switch between different timeframes

3. **Create Trading Strategies**
   - Navigate to the Strategy section
   - Choose between traditional indicators or AI-generated strategies
   - Configure strategy parameters

4. **Run Backtests**
   - Select a strategy and stock symbol
   - Choose backtest period
   - Analyze performance metrics and trading signals

5. **AI Strategy Optimization**
   - Use AI to generate optimized strategies
   - Iterative improvement based on backtest results
   - Multi-timeframe coordination

### Advanced Features

#### Multi-Timeframe Trading
The system supports coordinated trading across multiple timeframes:
- **Daily**: Short-term momentum trading
- **3-Day**: Swing trading opportunities
- **Weekly**: Medium-term trend following
- **Bi-weekly**: Position trading
- **Monthly**: Long-term value investing

#### AI Strategy Generation
1. Select AI model (Local Ollama or Google AI)
2. Choose market conditions (Bullish, Bearish, Neutral, Volatile)
3. Let AI generate optimized strategies
4. Backtest and refine automatically

#### Volume Monitoring
- Set volume threshold for alerts
- Monitor abnormal trading activity
- Real-time volume spike detection

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
ALPHA_VANTAGE_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Optional
POLYGON_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
TWELVE_DATA_API_KEY=your_key_here
TUSHARE_TOKEN=your_token_here

# Local AI
OLLAMA_URL=http://localhost:11434
```

### Ollama Setup (Optional)

For local AI processing:

1. **Install Ollama**
   ```bash
   # Visit https://ollama.ai/ for installation instructions
   ```

2. **Pull Llama3 model**
   ```bash
   ollama pull llama3
   ```

3. **Start Ollama service**
   ```bash
   ollama serve
   ```

## 📊 Data Sources

### Supported Markets
- **US Markets**: NYSE, NASDAQ
- **Hong Kong**: HKEX
- **China**: Shanghai Stock Exchange, Shenzhen Stock Exchange

### Data Types
- **OHLCV Data**: Open, High, Low, Close, Volume
- **Technical Indicators**: 28+ indicators including RSI, MACD, Bollinger Bands
- **Fundamental Data**: P/E ratio, market cap, financial metrics
- **News Sentiment**: Market sentiment analysis
- **Volume Analysis**: Institutional flow detection

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Trading stocks involves risk, and you should consult with a qualified financial advisor before making any investment decisions.

## 🆘 Support

- **Documentation**: Check the [Wiki](https://github.com/yourusername/Trading_AI/wiki)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/Trading_AI/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/Trading_AI/discussions)

## 🙏 Acknowledgments

- **yfinance**: For providing free stock market data
- **Alpha Vantage**: For comprehensive financial data API
- **Google AI**: For advanced AI capabilities
- **Ollama**: For local AI processing
- **Chart.js**: For beautiful charting capabilities
- **Flask**: For the web framework

## 📈 Roadmap

- [ ] Real-time trading execution
- [ ] More AI models integration
- [ ] Advanced portfolio management
- [ ] Mobile app development
- [ ] Cryptocurrency support
- [ ] Social trading features

---

**Made with ❤️ for the trading community**
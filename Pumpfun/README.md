# 🚀 Solana New Tokens Dashboard - Version 1.0

A real-time dashboard for tracking newly created and small market cap Solana tokens. Built with HTML, CSS, and JavaScript, this dashboard provides live data from CoinGecko API to help you discover fresh opportunities in the Solana ecosystem.

## ✨ Features

### 🆕 New Token Discovery
- **Real-time Solana token data** from CoinGecko API
- **Small market cap focus** (under $100M) for newer tokens
- **Volume filtering** to ensure active trading
- **Creation date tracking** when available

### 📊 Comprehensive Token Information
- **Market data**: Price, market cap, volume, supply
- **Price performance**: 24h and 7d changes
- **Token details**: Name, symbol, image, rank
- **Creation information**: Genesis date and age

### 🎨 Modern UI/UX
- **Responsive design** that works on all devices
- **Real-time status updates** with color-coded indicators
- **Interactive charts** showing token age distribution
- **Smart filtering** and search capabilities

### 🔍 Advanced Features
- **Multiple data sources** with fallback options
- **Error handling** with user-friendly messages
- **Console logging** for debugging and transparency
- **Sample data fallback** when APIs are unavailable

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Chart.js for data visualization
- **APIs**: CoinGecko API for cryptocurrency data
- **Styling**: Custom CSS with modern gradients and animations

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for API access

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/aghosh121/pumpfun.git
   cd pumpfun
   ```

2. **Choose your dashboard**
   
   **🆕 NEW: Advanced GeckoTerminal Dashboard (Recommended)**
   ```bash
   # Start local server
   python3 -m http.server 8001
   # Open: http://localhost:8001/geckoterminal_scraper_dashboard_working.html
   ```
   
   **Classic CoinGecko Dashboard**
   ```bash
   # Option 1: Double-click the HTML file
   open real_solana_tokens_dashboard.html
   
   # Option 2: Use a local server
   python3 -m http.server 8000
   # Then open http://localhost:8000 in your browser
   ```

3. **Start using the dashboard**
   - Click "Fetch Latest Pools" to get live data
   - Use sorting and filtering options
   - View top scorers and investment opportunities

## 🆕 NEW: Advanced GeckoTerminal Dashboard

### 🏆 Top Features
- **Real-time Solana pool data** from GeckoTerminal API
- **Comprehensive scoring system** (0-100 points) based on:
  - Liquidity (25 points)
  - Volume (25 points) 
  - Price performance (25 points)
  - Image quality (25 points)
- **Top scorers ranking** with beautiful visual display
- **Risk & opportunity assessment** for each token
- **Contract address copying** for easy trading

### 🎯 Investment Intelligence
- **Smart filtering** by score, risk, opportunity, and more
- **Grid & list views** for different analysis perspectives
- **Multi-page fetching** (up to 2,500 tokens)
- **Time-based filtering** (3-24 hours since launch)
- **Professional image analysis** and quality scoring

### 🚀 Advanced Functionality
- **Sorting options**: Score, risk, liquidity, volume, age, image quality
- **Quick filters**: Top scorers (80+), Good (60+), High opportunity, Low risk
- **Copy functionality**: One-click contract address copying
- **Debug logging**: Comprehensive console output for troubleshooting

## 📱 Usage

### GeckoTerminal Dashboard
- **Fetch Latest Pools**: Gets up to 2,500 recent Solana pools
- **Test API Connection**: Verifies GeckoTerminal API connectivity
- **Sort & Filter**: Use dropdown and buttons to find best opportunities
- **Copy Addresses**: Click copy buttons for easy trading

### Classic CoinGecko Dashboard
- **🆕 Get New Solana Tokens**: Fetches small market cap Solana tokens
- **🪙 Recent Tokens**: Gets recently launched tokens
- **🗑️ Clear All**: Clears current data
- **🔍 Test Connection**: Tests API connectivity

### Data Display
- **Token Cards**: Individual token information with metrics
- **Statistics Overview**: Total market cap, volume, token count, average age
- **Age Distribution Chart**: Visual representation of token ages
- **Filtering Options**: Search by name/symbol, filter by market cap and price change

### Real-time Features
- **Live API calls** to GeckoTerminal and CoinGecko
- **Status indicators** showing operation progress
- **Error handling** with helpful messages
- **Console logging** for transparency

## 🔧 Configuration

### API Settings
The dashboard uses CoinGecko API by default. You can modify the API endpoints in the JavaScript code:

```javascript
const COINGECKO_API = 'https://api.coingecko.com/api/v3';
```

### Filtering Options
Adjust the filtering criteria in the `fetchRealNewPools()` function:

```javascript
// Market cap threshold (currently $100M)
return token.market_cap < 100000000;

// Volume threshold (currently $1K)
return token.total_volume > 1000;
```

## 📊 Data Sources

### Primary API
- **CoinGecko API**: Cryptocurrency market data
- **Endpoint**: `/coins/markets` with Solana platform filter
- **Data**: Market cap, price, volume, supply, creation dates

### Data Processing
- **Filtering**: Small market cap, active volume
- **Sorting**: By market cap (ascending for newer tokens)
- **Limiting**: Top 10 results for performance

## 🎯 Use Cases

### For Traders
- **Discover new tokens** before they gain mainstream attention
- **Monitor small cap opportunities** in the Solana ecosystem
- **Track token performance** with real-time data

### For Researchers
- **Analyze token creation patterns** and trends
- **Study market dynamics** of new launches
- **Research Solana ecosystem growth**

### For Developers
- **Learn from existing implementations** of token dashboards
- **Understand API integration** with cryptocurrency data
- **Build upon the foundation** for custom solutions

## 🚨 Limitations

### API Constraints
- **Rate limiting**: CoinGecko has API call limits
- **Data availability**: Not all tokens have complete information
- **Update frequency**: Data may not be real-time

### Token Coverage
- **Solana platform only**: Focuses on Solana ecosystem
- **Market cap filtering**: May miss some legitimate new tokens
- **Volume requirements**: Excludes inactive tokens

## 🔮 Future Enhancements

### Planned Features
- **Multiple blockchain support** (Ethereum, Polygon, etc.)
- **Advanced filtering options** (launch date, token type)
- **Portfolio tracking** for discovered tokens
- **Price alerts** and notifications

### Technical Improvements
- **WebSocket integration** for real-time updates
- **Caching system** to reduce API calls
- **Progressive Web App** capabilities
- **Mobile app** versions

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Bug Reports
- **Use GitHub Issues** to report bugs
- **Include detailed descriptions** of the problem
- **Provide steps to reproduce** the issue
- **Share console logs** if available

### Feature Requests
- **Describe the feature** you'd like to see
- **Explain the use case** and benefits
- **Consider implementation complexity**
- **Be specific** about requirements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CoinGecko** for providing the cryptocurrency API
- **Chart.js** for the excellent charting library
- **Solana community** for building an amazing ecosystem
- **Open source contributors** who made this possible

## 📞 Support

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community support
- **Wiki**: For detailed documentation and guides

### Community
- **Discord**: Join our community server
- **Twitter**: Follow for updates and announcements
- **Telegram**: Get instant notifications

---

**Made with ❤️ for the Solana community**

*Version 1.0 - Stable Release*

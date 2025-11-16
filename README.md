#  Rootstock Token Metrics Visualization Tool

A comprehensive Python-based data visualization tool for analyzing Rootstock (RSK) token metrics with **real-time price data** from CoinGecko and Sovryn DEX. This tool helps developers and analysts visualize transaction volumes, holder distributions, live price movements, and on-chain activity with interactive charts and real-time data.

##  Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Web Dashboard](#web-dashboard)
  - [Python API](#python-api)
- [Configuration](#configuration)
- [Examples](#examples)
- [API Reference](#api-reference)

##  Features

### Data Collection
- **Real-time blockchain data** fetching from Rootstock RPC nodes
- **Live price data** from CoinGecko API and Sovryn DEX pools
- **Token metrics extraction** including transfers, balances, and holder information
- **Intelligent fallback mechanisms** ensuring all features work reliably
- **Configurable caching** for improved performance
- **Rate limiting** to respect API limits
- Support for both **mainnet and testnet**

### Data Sources & Reliability
- **Price Data**: CoinGecko API → Sovryn DEX → BTC conversion (always available)
- **Volume Data**: API historical volume → On-chain transfers → Simulated patterns
- **Holder Distribution**: On-chain balances → Pareto distribution simulation
- **Network Graph**: Transfer events → Network visualization
- **Automatic Fallback**: Every chart type has multiple data sources ensuring 100% uptime

### Visualization Capabilities
- **Interactive Charts**: All chart types work reliably with intelligent data fallbacks
  - Price History: Real-time prices from APIs
  - Volume Analysis: Historical volume with realistic patterns
  - Holder Distribution: Token distribution with Pareto modeling
  - Network Graph: Transfer network visualization
  - Multi-Metric Dashboard: Combined analytics view
- **Multiple Chart Types**: Line charts, bar charts, histograms, pie charts, network graphs
- **Customizable Themes**: Dark and light themes with configurable color schemes
- **Real-time Updates**: Auto-refresh capabilities for live monitoring
- **Comparison Tools**: Compare metrics across multiple tokens

### User Interfaces
- **Command-Line Interface (CLI)**: Quick access to all features from terminal
- **Web Dashboard**: Interactive browser-based interface with Dash
- **Python API**: Programmatic access for integration into other projects

### Export Options
- Charts as **PNG, SVG, or HTML**
- Data as **CSV** for further analysis
- **Configurable export settings** for resolution and format

##  Architecture

```
rskPython/
├── src/
│   ├── data/
│   │   ├── fetcher.py      # Blockchain data fetching
│   │   └── processor.py    # Data processing and analysis
│   ├── visualization/
│   │   ├── engine.py        # Core visualization engine
│   │   └── charts.py        # Chart factory and types
│   ├── interfaces/
│   │   ├── cli.py          # Command-line interface
│   │   └── web.py          # Web dashboard
│   └── utils/              # Utility functions
├── examples/               # Example scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher (tested up to 3.14)
- pip package manager
- Access to Rootstock RPC endpoint (public nodes available)
- Internet connection for price data APIs

### Step 1: Clone the Repository

```bash
git clone https://github.com/luisawatkins/Rootstock-Token-Metrics.git
cd Rootstock-Token-Metrics
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .  # Development installation
# or
pip install .     # Regular installation
```

### Step 4: Configure Settings (Optional)

The default `config.yaml` is pre-configured with public RPC endpoints and popular tokens. You can customize it if needed:

```yaml
rootstock:
  mainnet:
    rpc_url: "https://public-node.rsk.co"
  testnet:
    rpc_url: "https://public-node.testnet.rsk.co"
```

##  Quick Start

### Using the CLI

```bash
# Get token information
python -m src.interfaces.cli info 0x2acc95758f8b5F583470ba265eb685a8f45fc9d5

# Analyze transaction volume
python -m src.interfaces.cli volume 0x2acc95758f8b5F583470ba265eb685a8f45fc9d5 --days 30

# View holder distribution
python -m src.interfaces.cli holders 0x2acc95758f8b5F583470ba265eb685a8f45fc9d5 --top 20

# Create visualizations
python -m src.interfaces.cli visualize 0x2acc95758f8b5F583470ba265eb685a8f45fc9d5 --chart-type price --show
```

### Starting the Web Dashboard

```bash
python -m src.interfaces.web --network mainnet --port 8050
```

Then open your browser and navigate to `http://localhost:8050`

### Using as Python Library

```python
from src.data.fetcher import TokenDataFetcher
from src.visualization.engine import VisualizationEngine

# Initialize components
fetcher = TokenDataFetcher(network="mainnet")
viz_engine = VisualizationEngine()

# Fetch token data
token_address = "0x2acc95758f8b5F583470ba265eb685a8f45fc9d5"  # RIF Token
token_info = fetcher.get_token_info(token_address)

# Get REAL price data (from CoinGecko/Sovryn DEX)
price_data = fetcher.get_price_data(token_address, days=30)
print(f"Current price: ${price_data['price'].iloc[-1]:.2f}")  # Real market price!

# Create visualization
fig = viz_engine.create_price_chart(price_data)
fig.show()
```

##  Usage

### Command Line Interface

The CLI provides access to all tool features through simple commands:

#### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `info` | Get token information | `python -m src.interfaces.cli info TOKEN_ADDRESS` |
| `balance` | Check token balance | `python -m src.interfaces.cli balance TOKEN_ADDRESS WALLET_ADDRESS` |
| `volume` | Analyze transaction volume | `python -m src.interfaces.cli volume TOKEN_ADDRESS --days 7` |
| `holders` | View holder distribution | `python -m src.interfaces.cli holders TOKEN_ADDRESS --top 50` |
| `visualize` | Create charts | `python -m src.interfaces.cli visualize TOKEN_ADDRESS --chart-type price` |
| `compare` | Compare multiple tokens | `python -m src.interfaces.cli compare TOKEN1 TOKEN2 --metric price` |
| `network` | Display network status | `python -m src.interfaces.cli network` |
| `alert` | Check for anomalies | `python -m src.interfaces.cli alert TOKEN_ADDRESS --threshold 1000` |

#### CLI Options

- `--config, -c`: Specify configuration file
- `--network, -n`: Choose network (mainnet/testnet)
- `--verbose, -v`: Enable verbose logging
- `--export, -e`: Export data to file
- `--show, -s`: Display chart in browser

### Web Dashboard

The web dashboard provides an interactive interface for exploring token metrics:

#### Features
- **Token Selection**: Dropdown menu or custom address input
- **Real-time Updates**: Auto-refresh every minute
- **Interactive Charts**: All chart types work reliably
  - ✅ Price History - Live market prices
  - ✅ Volume Analysis - Historical & simulated volume
  - ✅ Holder Distribution - On-chain or modeled distribution
  - ✅ Network Graph - Transfer visualizations
  - ✅ Multi-Metric - Combined dashboard view
- **Export Options**: Download charts and data
- **Intelligent Fallbacks**: Charts always display useful data

#### Starting Options

```bash
python -m src.interfaces.web [OPTIONS]

Options:
  --config PATH     Configuration file path
  --network TEXT    Network to use (mainnet/testnet)
  --port INTEGER    Port to run server on
  --debug          Enable debug mode
```

### Python API

#### Data Fetching

```python
from src.data.fetcher import TokenDataFetcher

fetcher = TokenDataFetcher(network="mainnet")

# Get token information
token_info = fetcher.get_token_info("0x2acc95758f8b5F583470ba265eb685a8f45fc9d5")

# Get transfer events
transfers = fetcher.get_transfer_events(
    token_address="0x2acc95758f8b5F583470ba265eb685a8f45fc9d5",
    from_block=1000000,
    to_block=1001000
)

# Get transaction volume
volume_df = fetcher.get_transaction_volume(
    token_address="0x2acc95758f8b5F583470ba265eb685a8f45fc9d5",
    days=30,
    interval="daily"
)
```

#### Data Processing

```python
from src.data.processor import DataProcessor

processor = DataProcessor()

# Calculate moving averages
df_with_ma = processor.calculate_moving_average(price_df, "price", window=7)

# Detect anomalies
df_with_anomalies = processor.detect_anomalies(volume_df, "volume", method="zscore")

# Calculate holder metrics
metrics = processor.calculate_holder_metrics(holders_df)
```

#### Visualization

```python
from src.visualization.engine import VisualizationEngine
from src.visualization.charts import ChartFactory

# Using the engine directly
engine = VisualizationEngine()
fig = engine.create_price_chart(
    price_df,
    show_ma=True,
    ma_periods=[7, 30]
)

# Using the chart factory
factory = ChartFactory()
fig = factory.create_chart("holders", holders_df, top_n=20)

# Export chart
engine.export_figure(fig, "chart.png", format="png")
```

##  Configuration

The `config.yaml` file controls various aspects of the tool:

### Network Configuration

```yaml
rootstock:
  mainnet:
    rpc_url: "https://public-node.rsk.co"
    chain_id: 30
  testnet:
    rpc_url: "https://public-node.testnet.rsk.co"
    chain_id: 31
  default_network: "mainnet"
```

### Data Collection Settings

```yaml
data_collection:
  max_blocks_per_request: 1000
  default_time_range: 30
  enable_cache: true
  cache_ttl_seconds: 300
  requests_per_second: 10
```

### Visualization Settings

```yaml
visualization:
  theme: "plotly_dark"
  color_schemes:
    default: ["#FF6B6B", "#4ECDC4", "#45B7D1"]
  default_width: 1200
  default_height: 600
  export:
    dpi: 300
    format: "png"
```

### Token Presets

```yaml
tokens:
  RIF:
    address: "0x2acc95758f8b5F583470ba265eb685a8f45fc9d5"
    decimals: 18
    symbol: "RIF"
```

##  Supported Tokens with Live Prices

The tool fetches **real-time price data** for these Rootstock tokens:

| Token | Symbol | Data Source | Price Feed |
|-------|--------|-------------|------------|
| RIF Token | RIF | CoinGecko API | ✅ Live prices |
| Sovryn | SOV | CoinGecko API | ✅ Live prices |
| Dollar on Chain | DOC | CoinGecko API | ✅ Live prices |
| Bitcoin Pro | BPRO | CoinGecko API | ✅ Live prices |
| Wrapped BTC | WRBTC | BTC Price Feeds | ✅ Live prices |
| USDT on RSK | rUSDT | Sovryn DEX | ✅ On-chain prices |
| Other tokens | - | DEX Pools | ✅ Calculated prices |

##  Examples

The `examples/` directory contains complete working scripts:

- **`basic_usage.py`** - Simple token analysis with real price data
- **`advanced_analysis.py`** - Multi-token correlation analysis  
- **`automated_reporting.py`** - Set up scheduled reports with live prices

Run any example to see the tool in action:
```bash
python examples/basic_usage.py
```

##  API Reference

### TokenDataFetcher

Main class for fetching blockchain data:

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `get_token_info()` | Get token details | `token_address` | `Dict` |
| `get_token_balance()` | Get wallet balance | `token_address, wallet_address` | `Dict` |
| `get_transfer_events()` | Get transfers | `token_address, from_block, to_block` | `List[Dict]` |
| `get_transaction_volume()` | Get volume data | `token_address, days, interval` | `DataFrame` |
| `get_holder_distribution()` | Get holders | `token_address, top_n` | `DataFrame` |

### VisualizationEngine

Core visualization functionality:

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `create_price_chart()` | Price history chart | `df, show_ma, ma_periods` | `Figure` |
| `create_volume_chart()` | Volume chart | `df, transaction_count` | `Figure` |
| `create_holder_distribution_chart()` | Holder chart | `df, top_n` | `Figure` |
| `create_network_graph()` | Transfer network | `transfers, top_n` | `Figure` |
| `export_figure()` | Export chart | `fig, filepath, format` | `bool` |

### DataProcessor

Data processing utilities:

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `calculate_moving_average()` | Add MA to data | `df, column, window` | `DataFrame` |
| `calculate_volatility()` | Calculate volatility | `df, price_column, window` | `DataFrame` |
| `detect_anomalies()` | Find anomalies | `df, column, method` | `DataFrame` |
| `calculate_holder_metrics()` | Holder statistics | `holder_df` | `Dict` |

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


**Built with ❤️ for the Rootstock ecosystem**
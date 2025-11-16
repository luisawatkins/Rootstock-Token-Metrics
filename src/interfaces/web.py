"""
Web interface for the Rootstock Token Metrics Visualization Tool using Dash.

This module provides an interactive web dashboard for visualizing token metrics
with real-time updates and user-friendly controls.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fetcher import TokenDataFetcher
from src.data.processor import DataProcessor
from src.visualization.engine import VisualizationEngine
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)


class TokenMetricsDashboard:
    """Interactive web dashboard for token metrics visualization."""
    
    def __init__(self, config_path: str = "config.yaml", network: str = "mainnet"):
        """
        Initialize the dashboard.
        
        Args:
            config_path: Path to configuration file
            network: Network to connect to
        """
        self.config = self._load_config(config_path)
        self.network = network
        
        # Initialize components
        self.fetcher = TokenDataFetcher(config_path, network)
        self.processor = DataProcessor()
        self.chart_factory = ChartFactory(config_path)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"Dashboard initialized for {network}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}")
            return {}
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 Rootstock Token Metrics Dashboard", 
                           className="text-center mb-4 mt-3"),
                    html.P(f"Network: {self.network.upper()}", 
                          className="text-center text-muted")
                ])
            ]),
            
            html.Hr(),
            
            # Controls Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Token Selection", className="card-title"),
                            dcc.Dropdown(
                                id="token-dropdown",
                                options=[
                                    {"label": f"{token['symbol']} ({name})", 
                                     "value": token["address"]}
                                    for name, token in self.config.get("tokens", {}).items()
                                ],
                                placeholder="Select a token or enter address",
                                className="mb-3"
                            ),
                            dbc.Input(
                                id="custom-token-input",
                                placeholder="Or enter custom token address",
                                className="mb-3"
                            ),
                            dbc.Button(
                                "Load Token Data",
                                id="load-button",
                                color="primary",
                                className="w-100"
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Analysis Options", className="card-title"),
                            dbc.Label("Time Range (days)"),
                            dbc.Input(
                                id="days-input",
                                type="number",
                                value=30,
                                min=1,
                                max=365,
                                className="mb-3"
                            ),
                            dbc.Label("Chart Type"),
                            dcc.Dropdown(
                                id="chart-type-dropdown",
                                options=[
                                    {"label": "Price History", "value": "price"},
                                    {"label": "Volume Analysis", "value": "volume"},
                                    {"label": "Holder Distribution", "value": "holders"},
                                    {"label": "Network Graph", "value": "network"},
                                    {"label": "Multi-Metric", "value": "multi"}
                                ],
                                value="price",
                                className="mb-3"
                            ),
                            dbc.Checklist(
                                id="options-checklist",
                                options=[
                                    {"label": "Show Moving Averages", "value": "ma"},
                                    {"label": "Log Scale", "value": "log"},
                                    {"label": "Show Anomalies", "value": "anomalies"}
                                ],
                                value=["ma"],
                                inline=True
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Token Info", className="card-title"),
                            html.Div(id="token-info-display", children=[
                                html.P("No token selected", className="text-muted")
                            ])
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Network Status", className="card-title"),
                            html.Div(id="network-status", children=[
                                html.P("Checking connection...", className="text-muted")
                            ])
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main Chart Area
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-chart",
                                type="graph",
                                children=[
                                    dcc.Graph(
                                        id="main-chart",
                                        config={"displayModeBar": True},
                                        style={"height": "500px"}
                                    )
                                ]
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Key Metrics", className="card-title"),
                            html.Div(id="metrics-display")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Additional Analysis", className="card-title"),
                            dcc.Graph(
                                id="secondary-chart",
                                config={"displayModeBar": False},
                                style={"height": "300px"}
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Export Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Export Options", className="card-title"),
                            dbc.ButtonGroup([
                                dbc.Button("Export Chart (PNG)", id="export-png-button", color="secondary"),
                                dbc.Button("Export Chart (HTML)", id="export-html-button", color="secondary"),
                                dbc.Button("Export Data (CSV)", id="export-csv-button", color="secondary")
                            ])
                        ])
                    ])
                ], width=12)
            ]),
            
            # Hidden divs for storing data
            dcc.Store(id="token-data-store"),
            dcc.Store(id="chart-data-store"),
            
            # Download component
            dcc.Download(id="download-component"),
            
            # Auto-refresh interval
            dcc.Interval(
                id="interval-component",
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output("network-status", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_network_status(n):
            """Update network status display."""
            try:
                metrics = self.fetcher.get_network_metrics()
                if metrics and metrics.get("connected"):
                    return [
                        html.P([
                            html.Strong("Status: "),
                            html.Span("Connected ✅", className="text-success")
                        ]),
                        html.P([
                            html.Strong("Block: "),
                            f"{metrics.get('block_number', 'N/A')}"
                        ]),
                        html.P([
                            html.Strong("Gas Price: "),
                            f"{metrics.get('gas_price', 0) / 1e9:.2f} Gwei"
                        ])
                    ]
                else:
                    return [html.P("Disconnected ❌", className="text-danger")]
            except Exception as e:
                logger.error(f"Error updating network status: {e}")
                return [html.P("Error ⚠️", className="text-warning")]
        
        @self.app.callback(
            [Output("token-info-display", "children"),
             Output("token-data-store", "data")],
            [Input("load-button", "n_clicks")],
            [State("token-dropdown", "value"),
             State("custom-token-input", "value")]
        )
        def load_token_info(n_clicks, dropdown_value, custom_value):
            """Load token information."""
            if not n_clicks:
                return [html.P("No token selected", className="text-muted")], None
            
            token_address = custom_value if custom_value else dropdown_value
            if not token_address:
                return [html.P("Please select or enter a token", className="text-warning")], None
            
            try:
                token_info = self.fetcher.get_token_info(token_address)
                if token_info:
                    display = [
                        html.P([html.Strong("Name: "), token_info.get("name", "Unknown")]),
                        html.P([html.Strong("Symbol: "), token_info.get("symbol", "Unknown")]),
                        html.P([html.Strong("Decimals: "), str(token_info.get("decimals", 0))]),
                        html.P([html.Strong("Total Supply: "), 
                               f"{token_info.get('total_supply_formatted', 0):,.2f}"])
                    ]
                    return display, token_info
                else:
                    return [html.P("Failed to load token", className="text-danger")], None
            except Exception as e:
                logger.error(f"Error loading token info: {e}")
                return [html.P(f"Error: {str(e)}", className="text-danger")], None
        
        @self.app.callback(
            [Output("main-chart", "figure"),
             Output("chart-data-store", "data")],
            [Input("token-data-store", "data"),
             Input("chart-type-dropdown", "value"),
             Input("days-input", "value"),
             Input("options-checklist", "value")]
        )
        def update_main_chart(token_data, chart_type, days, options):
            """Update the main chart based on selections."""
            if not token_data:
                return go.Figure().add_annotation(
                    text="Please load token data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                ), None
            
            try:
                token_address = token_data["address"]
                
                if chart_type == "price":
                    df = self.fetcher.get_price_data(token_address, days)
                    if not df.empty:
                        fig = self.chart_factory.create_chart(
                            "price", df,
                            show_ma="ma" in options,
                            title=f"{token_data.get('symbol', 'Token')} Price History"
                        )
                        return fig, df.to_dict()
                
                elif chart_type == "volume":
                    df = self.fetcher.get_transaction_volume(token_address, days)
                    if not df.empty:
                        fig = self.chart_factory.create_chart(
                            "volume", df,
                            title=f"{token_data.get('symbol', 'Token')} Transaction Volume"
                        )
                        return fig, df.to_dict()
                
                elif chart_type == "holders":
                    df = self.fetcher.get_holder_distribution(token_address)
                    if not df.empty:
                        fig = self.chart_factory.create_chart(
                            "holders", df,
                            title=f"{token_data.get('symbol', 'Token')} Holder Distribution"
                        )
                        return fig, df.to_dict()
                
                elif chart_type == "network":
                    transfers = self.fetcher.get_transfer_events(token_address, limit=500)
                    if transfers:
                        fig = self.chart_factory.create_chart(
                            "network", transfers,
                            title=f"{token_data.get('symbol', 'Token')} Transfer Network"
                        )
                        return fig, {"transfers": transfers}
                
                elif chart_type == "multi":
                    # Create multi-metric dashboard
                    data = {
                        "price": self.fetcher.get_price_data(token_address, days),
                        "volume": self.fetcher.get_transaction_volume(token_address, days),
                        "holders": self.fetcher.get_holder_distribution(token_address, top_n=10)
                    }
                    fig = self.chart_factory.create_chart(
                        "dashboard", data,
                        title=f"{token_data.get('symbol', 'Token')} Dashboard"
                    )
                    return fig, {"multi": True}
                
                # If no data available
                return go.Figure().add_annotation(
                    text="No data available for this chart type",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                ), None
                
            except Exception as e:
                logger.error(f"Error updating chart: {e}")
                return go.Figure().add_annotation(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                ), None
        
        @self.app.callback(
            Output("metrics-display", "children"),
            [Input("chart-data-store", "data"),
             Input("chart-type-dropdown", "value")]
        )
        def update_metrics(chart_data, chart_type):
            """Update metrics display based on chart data."""
            if not chart_data:
                return html.P("No data available", className="text-muted")
            
            try:
                metrics_cards = []
                
                if chart_type == "price" and "price" in chart_data:
                    df = pd.DataFrame(chart_data)
                    growth_metrics = self.processor.calculate_growth_metrics(df, "price")
                    
                    metrics_cards.append(
                        dbc.Row([
                            dbc.Col([
                                html.H6("Current Price"),
                                html.H4(f"${df['price'].iloc[-1]:,.2f}")
                            ], width=4),
                            dbc.Col([
                                html.H6("24h Change"),
                                html.H4(f"{growth_metrics.get('growth_1d', 0):+.2f}%",
                                       className="text-success" if growth_metrics.get('growth_1d', 0) > 0 else "text-danger")
                            ], width=4),
                            dbc.Col([
                                html.H6("7d Change"),
                                html.H4(f"{growth_metrics.get('growth_7d', 0):+.2f}%",
                                       className="text-success" if growth_metrics.get('growth_7d', 0) > 0 else "text-danger")
                            ], width=4)
                        ])
                    )
                
                elif chart_type == "volume" and "volume" in chart_data:
                    df = pd.DataFrame(chart_data)
                    metrics_cards.append(
                        dbc.Row([
                            dbc.Col([
                                html.H6("Total Volume"),
                                html.H4(f"{df['volume'].sum():,.0f}")
                            ], width=4),
                            dbc.Col([
                                html.H6("Avg Daily Volume"),
                                html.H4(f"{df['volume'].mean():,.0f}")
                            ], width=4),
                            dbc.Col([
                                html.H6("Total Transactions"),
                                html.H4(f"{df.get('transaction_count', pd.Series([0])).sum():,.0f}")
                            ], width=4)
                        ])
                    )
                
                elif chart_type == "holders" and "address" in chart_data:
                    df = pd.DataFrame(chart_data)
                    holder_metrics = self.processor.calculate_holder_metrics(df)
                    
                    metrics_cards.append(
                        dbc.Row([
                            dbc.Col([
                                html.H6("Total Holders"),
                                html.H4(str(holder_metrics.get("total_holders", 0)))
                            ], width=4),
                            dbc.Col([
                                html.H6("Top 10 Concentration"),
                                html.H4(f"{holder_metrics.get('top_10_concentration', 0):.1f}%")
                            ], width=4),
                            dbc.Col([
                                html.H6("Gini Coefficient"),
                                html.H4(f"{holder_metrics.get('gini_coefficient', 0):.3f}")
                            ], width=4)
                        ])
                    )
                
                return metrics_cards if metrics_cards else html.P("Metrics not available for this chart", className="text-muted")
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                return html.P(f"Error calculating metrics", className="text-danger")
        
        @self.app.callback(
            Output("secondary-chart", "figure"),
            [Input("chart-data-store", "data"),
             Input("chart-type-dropdown", "value")]
        )
        def update_secondary_chart(chart_data, chart_type):
            """Update secondary analysis chart."""
            if not chart_data:
                return go.Figure()
            
            try:
                if chart_type == "price" and "price" in chart_data:
                    df = pd.DataFrame(chart_data)
                    # Create volatility chart
                    df = self.processor.calculate_volatility(df)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["date"],
                        y=df["volatility"],
                        mode="lines",
                        name="Volatility",
                        line=dict(color="orange")
                    ))
                    fig.update_layout(
                        title="Price Volatility",
                        height=300,
                        showlegend=False,
                        margin=dict(t=30, b=30, l=30, r=30)
                    )
                    return fig
                
                elif chart_type == "volume" and "volume" in chart_data:
                    df = pd.DataFrame(chart_data)
                    # Create volume distribution histogram
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df["volume"],
                        nbinsx=20,
                        marker_color="lightblue"
                    ))
                    fig.update_layout(
                        title="Volume Distribution",
                        height=300,
                        showlegend=False,
                        margin=dict(t=30, b=30, l=30, r=30)
                    )
                    return fig
                
                elif chart_type == "holders" and "balance" in chart_data:
                    df = pd.DataFrame(chart_data)
                    # Create holder tier pie chart
                    holder_metrics = self.processor.calculate_holder_metrics(df)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=["Whales", "Large", "Medium", "Small"],
                        values=[
                            holder_metrics.get("whales", 0),
                            holder_metrics.get("large_holders", 0),
                            holder_metrics.get("medium_holders", 0),
                            holder_metrics.get("small_holders", 0)
                        ],
                        hole=0.3
                    ))
                    fig.update_layout(
                        title="Holder Tiers",
                        height=300,
                        showlegend=True,
                        margin=dict(t=30, b=30, l=30, r=30)
                    )
                    return fig
                
                return go.Figure()
                
            except Exception as e:
                logger.error(f"Error updating secondary chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("download-component", "data"),
            [Input("export-png-button", "n_clicks"),
             Input("export-html-button", "n_clicks"),
             Input("export-csv-button", "n_clicks")],
            [State("main-chart", "figure"),
             State("chart-data-store", "data")]
        )
        def handle_export(png_clicks, html_clicks, csv_clicks, figure, chart_data):
            """Handle export button clicks."""
            ctx = callback_context
            if not ctx.triggered:
                return None
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            try:
                if button_id == "export-csv-button" and chart_data:
                    # Export data to CSV
                    df = pd.DataFrame(chart_data)
                    return dcc.send_data_frame(df.to_csv, "token_data.csv", index=False)
                
                # For chart exports, would need additional libraries
                # This is a placeholder for the export functionality
                return None
                
            except Exception as e:
                logger.error(f"Error exporting: {e}")
                return None
    
    def run(self, debug: bool = False, port: int = 8050):
        """
        Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        host = self.config.get("web_interface", {}).get("host", "127.0.0.1")
        self.app.run(debug=debug, host=host, port=port)


def main():
    """Main entry point for the web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rootstock Token Metrics Web Dashboard")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--network", "-n", default="mainnet", choices=["mainnet", "testnet"])
    parser.add_argument("--port", "-p", type=int, default=8050, help="Port to run on")
    parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and run dashboard
    dashboard = TokenMetricsDashboard(args.config, args.network)
    
    print(f"Starting Rootstock Token Metrics Dashboard...")
    print(f"Network: {args.network}")
    print(f"Access the dashboard at: http://127.0.0.1:{args.port}")
    
    dashboard.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()

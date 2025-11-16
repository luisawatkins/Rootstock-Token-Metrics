"""
Visualization engine for creating interactive charts and graphs.

This module provides the core visualization functionality for token metrics,
including time series charts, distribution plots, and network graphs.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import io
import base64

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Main visualization engine for creating charts and graphs."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the VisualizationEngine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_theme()
        logger.info("VisualizationEngine initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "visualization": {
                "theme": "plotly_dark",
                "color_schemes": {
                    "default": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
                    "price": ["#00D632", "#FF3838"],
                    "volume": ["#3498DB", "#2980B9"]
                },
                "default_width": 1200,
                "default_height": 600,
                "export": {
                    "dpi": 300,
                    "format": "png"
                }
            }
        }
    
    def _setup_theme(self):
        """Setup visualization theme and styles."""
        # Set Plotly theme
        import plotly.io as pio
        theme = self.config["visualization"]["theme"]
        if theme in pio.templates:
            pio.templates.default = theme
        
        # Set matplotlib/seaborn style
        plt.style.use("dark_background" if "dark" in theme else "default")
        sns.set_theme(style="darkgrid" if "dark" in theme else "whitegrid")
    
    def create_price_chart(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        price_column: str = "price",
        volume_column: Optional[str] = "volume",
        title: str = "Token Price History",
        show_ma: bool = True,
        ma_periods: List[int] = [7, 30]
    ) -> go.Figure:
        """
        Create an interactive price chart with optional volume.
        
        Args:
            df: DataFrame with price data
            date_column: Column containing dates
            price_column: Column containing prices
            volume_column: Optional column containing volume
            title: Chart title
            show_ma: Whether to show moving averages
            ma_periods: Periods for moving averages
            
        Returns:
            Plotly figure object
        """
        # Create subplots if volume is provided
        if volume_column and volume_column in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=("Price", "Volume")
            )
            
            # Add volume bar chart
            fig.add_trace(
                go.Bar(
                    x=df[date_column],
                    y=df[volume_column],
                    name="Volume",
                    marker_color=self.config["visualization"]["color_schemes"]["volume"][0],
                    opacity=0.7
                ),
                row=2, col=1
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # Add main price line
        fig.add_trace(
            go.Scatter(
                x=df[date_column],
                y=df[price_column],
                mode="lines",
                name="Price",
                line=dict(
                    color=self.config["visualization"]["color_schemes"]["price"][0],
                    width=2
                ),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add moving averages if requested
        if show_ma:
            colors = self.config["visualization"]["color_schemes"]["default"]
            for i, period in enumerate(ma_periods):
                if len(df) >= period:
                    ma_values = df[price_column].rolling(window=period).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df[date_column],
                            y=ma_values,
                            mode="lines",
                            name=f"MA{period}",
                            line=dict(
                                color=colors[i % len(colors)],
                                width=1,
                                dash="dash"
                            ),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified",
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    def create_volume_chart(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        volume_column: str = "volume",
        transaction_count_column: Optional[str] = "transaction_count",
        title: str = "Transaction Volume Over Time"
    ) -> go.Figure:
        """
        Create a volume chart with optional transaction count.
        
        Args:
            df: DataFrame with volume data
            date_column: Column containing dates
            volume_column: Column containing volume
            transaction_count_column: Optional column with transaction counts
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df[date_column],
                y=df[volume_column],
                name="Volume",
                marker_color=self.config["visualization"]["color_schemes"]["volume"][0],
                hovertemplate="Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add transaction count line if available
        if transaction_count_column and transaction_count_column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[date_column],
                    y=df[transaction_count_column],
                    mode="lines+markers",
                    name="Transaction Count",
                    line=dict(
                        color=self.config["visualization"]["color_schemes"]["default"][1],
                        width=2
                    ),
                    marker=dict(size=4),
                    hovertemplate="Date: %{x}<br>Transactions: %{y}<extra></extra>"
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            hovermode="x unified",
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"],
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Volume", secondary_y=False)
        if transaction_count_column:
            fig.update_yaxes(title_text="Transaction Count", secondary_y=True)
        
        return fig
    
    def create_holder_distribution_chart(
        self,
        df: pd.DataFrame,
        address_column: str = "address",
        balance_column: str = "balance",
        percentage_column: str = "percentage",
        top_n: int = 20,
        title: str = "Top Token Holders Distribution"
    ) -> go.Figure:
        """
        Create a holder distribution chart.
        
        Args:
            df: DataFrame with holder data
            address_column: Column containing addresses
            balance_column: Column containing balances
            percentage_column: Column containing percentages
            top_n: Number of top holders to show
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Get top N holders
        df_top = df.head(top_n).copy()
        
        # Shorten addresses for display
        df_top["address_short"] = df_top[address_column].apply(
            lambda x: f"{x[:6]}...{x[-4:]}" if len(x) > 10 else x
        )
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                y=df_top["address_short"],
                x=df_top[percentage_column],
                orientation="h",
                text=df_top[percentage_column].apply(lambda x: f"{x:.2f}%"),
                textposition="auto",
                marker_color=self.config["visualization"]["color_schemes"]["default"][0],
                hovertemplate="Address: %{y}<br>Balance: %{customdata[0]:,.2f}<br>Percentage: %{x:.2f}%<extra></extra>",
                customdata=df_top[[balance_column]]
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Percentage of Total Supply (%)",
            yaxis_title="Holder Address",
            width=self.config["visualization"]["default_width"],
            height=max(600, top_n * 30),  # Adjust height based on number of holders
            showlegend=False
        )
        
        # Reverse y-axis to show largest holders at top
        fig.update_yaxes(autorange="reversed")
        
        return fig
    
    def create_distribution_histogram(
        self,
        df: pd.DataFrame,
        value_column: str,
        bins: int = 50,
        title: str = "Distribution Histogram",
        log_scale: bool = False
    ) -> go.Figure:
        """
        Create a distribution histogram.
        
        Args:
            df: DataFrame with data
            value_column: Column to create histogram for
            bins: Number of bins
            title: Chart title
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=df[value_column],
                nbinsx=bins,
                marker_color=self.config["visualization"]["color_schemes"]["default"][2],
                opacity=0.7,
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
            )
        )
        
        # Add mean and median lines
        mean_val = df[value_column].mean()
        median_val = df[value_column].median()
        
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}"
        )
        
        fig.add_vline(
            x=median_val,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {median_val:.2f}"
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=value_column.replace("_", " ").title(),
            yaxis_title="Count",
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"],
            showlegend=False
        )
        
        if log_scale:
            fig.update_yaxes(type="log")
        
        return fig
    
    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: List[str],
        title: str = "Correlation Heatmap"
    ) -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            df: DataFrame with data
            columns: Columns to include in correlation
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"]
        )
        
        return fig
    
    def create_pie_chart(
        self,
        df: pd.DataFrame,
        labels_column: str,
        values_column: str,
        title: str = "Distribution",
        hole: float = 0.3
    ) -> go.Figure:
        """
        Create a pie or donut chart.
        
        Args:
            df: DataFrame with data
            labels_column: Column containing labels
            values_column: Column containing values
            title: Chart title
            hole: Size of hole for donut chart (0 for pie chart)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=df[labels_column],
                    values=df[values_column],
                    hole=hole,
                    marker=dict(
                        colors=self.config["visualization"]["color_schemes"]["default"]
                    ),
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>"
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"]
        )
        
        return fig
    
    def create_network_graph(
        self,
        transfers: List[Dict[str, Any]],
        title: str = "Token Transfer Network",
        top_n: int = 50
    ) -> go.Figure:
        """
        Create a network graph of token transfers.
        
        Args:
            transfers: List of transfer events
            title: Chart title
            top_n: Number of top addresses to include
            
        Returns:
            Plotly figure object
        """
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges from transfers
        for transfer in transfers[:top_n * 10]:  # Limit transfers for performance
            G.add_edge(
                transfer["from"][:8] + "...",
                transfer["to"][:8] + "...",
                weight=transfer.get("value_formatted", 1)
            )
        
        # Get top nodes by degree
        node_degrees = dict(G.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_names = [node for node, _ in top_nodes]
        
        # Create subgraph with top nodes
        G_sub = G.subgraph(top_node_names)
        
        # Calculate layout
        pos = nx.spring_layout(G_sub, k=1, iterations=50)
        
        # Create edge traces
        edge_trace = []
        for edge in G_sub.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=0.5, color="#888"),
                    hoverinfo="none"
                )
            )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G_sub.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Degree: {G_sub.degree(node)}")
            node_size.append(10 + G_sub.degree(node) * 2)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[node for node in G_sub.nodes()],
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_size,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Node Degree",
                    thickness=15,
                    xanchor="left"
                )
            ),
            hovertext=node_text,
            hoverinfo="text"
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_candlestick_chart(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        open_column: str = "open",
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close",
        title: str = "Price Candlestick Chart"
    ) -> go.Figure:
        """
        Create a candlestick chart for OHLC data.
        
        Args:
            df: DataFrame with OHLC data
            date_column: Column containing dates
            open_column: Column containing open prices
            high_column: Column containing high prices
            low_column: Column containing low prices
            close_column: Column containing close prices
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df[date_column],
                    open=df[open_column],
                    high=df[high_column],
                    low=df[low_column],
                    close=df[close_column],
                    increasing_line_color=self.config["visualization"]["color_schemes"]["price"][0],
                    decreasing_line_color=self.config["visualization"]["color_schemes"]["price"][1]
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"],
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def export_figure(
        self,
        fig: go.Figure,
        filepath: str,
        format: Optional[str] = None
    ) -> bool:
        """
        Export a Plotly figure to file.
        
        Args:
            fig: Plotly figure object
            filepath: Path to save the file
            format: Export format (png, svg, pdf, html)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format is None:
                format = self.config["visualization"]["export"]["format"]
            
            if format == "html":
                fig.write_html(filepath)
            elif format in ["png", "svg", "pdf"]:
                fig.write_image(filepath, format=format)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Figure exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting figure: {e}")
            return False
    
    def create_dashboard(
        self,
        data: Dict[str, pd.DataFrame],
        title: str = "Token Metrics Dashboard"
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            data: Dictionary of DataFrames with different metrics
            title: Dashboard title
            
        Returns:
            Plotly figure object with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Price History", "Volume", "Holder Distribution", "Metrics"),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add price chart if available
        if "price" in data and not data["price"].empty:
            fig.add_trace(
                go.Scatter(
                    x=data["price"]["date"],
                    y=data["price"]["price"],
                    mode="lines",
                    name="Price",
                    line=dict(color=self.config["visualization"]["color_schemes"]["price"][0])
                ),
                row=1, col=1
            )
        
        # Add volume chart if available
        if "volume" in data and not data["volume"].empty:
            fig.add_trace(
                go.Bar(
                    x=data["volume"]["date"],
                    y=data["volume"]["volume"],
                    name="Volume",
                    marker_color=self.config["visualization"]["color_schemes"]["volume"][0]
                ),
                row=1, col=2
            )
        
        # Add holder distribution if available
        if "holders" in data and not data["holders"].empty:
            top_holders = data["holders"].head(10)
            fig.add_trace(
                go.Bar(
                    x=top_holders["address"].apply(lambda x: x[:8] + "..."),
                    y=top_holders["percentage"],
                    name="Top Holders",
                    marker_color=self.config["visualization"]["color_schemes"]["default"][1]
                ),
                row=2, col=1
            )
        
        # Add key metrics indicator if available
        if "metrics" in data:
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=data["metrics"].get("current_price", 0),
                    delta={"reference": data["metrics"].get("previous_price", 0)},
                    title={"text": "Current Price"},
                    domain={"x": [0.5, 1], "y": [0, 0.5]}
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            width=self.config["visualization"]["default_width"],
            height=self.config["visualization"]["default_height"] * 1.5
        )
        
        return fig

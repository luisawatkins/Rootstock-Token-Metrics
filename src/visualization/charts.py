"""
Chart factory for creating specific chart types.

This module provides a factory pattern for creating different chart types
based on the data and requirements.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.graph_objects as go

from .engine import VisualizationEngine

logger = logging.getLogger(__name__)


class ChartFactory:
    """Factory class for creating various chart types."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ChartFactory.
        
        Args:
            config_path: Path to configuration file
        """
        self.engine = VisualizationEngine(config_path)
        self.chart_types = {
            "price": self.engine.create_price_chart,
            "volume": self.engine.create_volume_chart,
            "holders": self.engine.create_holder_distribution_chart,
            "histogram": self.engine.create_distribution_histogram,
            "correlation": self.engine.create_correlation_heatmap,
            "pie": self.engine.create_pie_chart,
            "network": self.engine.create_network_graph,
            "candlestick": self.engine.create_candlestick_chart,
            "dashboard": self.engine.create_dashboard
        }
        logger.info("ChartFactory initialized")
    
    def create_chart(
        self,
        chart_type: str,
        data: Any,
        **kwargs
    ) -> Optional[go.Figure]:
        """
        Create a chart of the specified type.
        
        Args:
            chart_type: Type of chart to create
            data: Data for the chart
            **kwargs: Additional arguments for the chart
            
        Returns:
            Plotly figure object or None if chart type not found
        """
        if chart_type not in self.chart_types:
            logger.error(f"Unknown chart type: {chart_type}")
            logger.info(f"Available chart types: {list(self.chart_types.keys())}")
            return None
        
        try:
            return self.chart_types[chart_type](data, **kwargs)
        except Exception as e:
            logger.error(f"Error creating {chart_type} chart: {e}")
            return None
    
    def get_available_charts(self) -> List[str]:
        """
        Get list of available chart types.
        
        Returns:
            List of chart type names
        """
        return list(self.chart_types.keys())
    
    def create_comparison_chart(
        self,
        dfs: Dict[str, pd.DataFrame],
        chart_type: str = "price",
        **kwargs
    ) -> go.Figure:
        """
        Create a comparison chart for multiple tokens.
        
        Args:
            dfs: Dictionary of DataFrames (token_name -> DataFrame)
            chart_type: Type of comparison chart
            **kwargs: Additional arguments
            
        Returns:
            Plotly figure with comparison
        """
        fig = go.Figure()
        colors = self.engine.config["visualization"]["color_schemes"]["default"]
        
        for i, (token_name, df) in enumerate(dfs.items()):
            color = colors[i % len(colors)]
            
            if chart_type == "price":
                fig.add_trace(
                    go.Scatter(
                        x=df.get("date", df.index),
                        y=df.get("price", df.iloc[:, 0]),
                        mode="lines",
                        name=token_name,
                        line=dict(color=color, width=2)
                    )
                )
            elif chart_type == "volume":
                fig.add_trace(
                    go.Bar(
                        x=df.get("date", df.index),
                        y=df.get("volume", df.iloc[:, 0]),
                        name=token_name,
                        marker_color=color,
                        opacity=0.7
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=kwargs.get("title", f"Token {chart_type.title()} Comparison"),
            xaxis_title="Date",
            yaxis_title=chart_type.title(),
            hovermode="x unified",
            width=self.engine.config["visualization"]["default_width"],
            height=self.engine.config["visualization"]["default_height"],
            showlegend=True
        )
        
        return fig
    
    def create_multi_metric_chart(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        date_column: str = "date",
        title: str = "Multi-Metric Analysis"
    ) -> go.Figure:
        """
        Create a chart with multiple metrics on different y-axes.
        
        Args:
            df: DataFrame with metrics
            metrics: List of metric column names
            date_column: Date column name
            title: Chart title
            
        Returns:
            Plotly figure with multiple metrics
        """
        from plotly.subplots import make_subplots
        
        # Create subplots with secondary y-axis
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=metrics
        )
        
        colors = self.engine.config["visualization"]["color_schemes"]["default"]
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df[date_column],
                        y=df[metric],
                        mode="lines",
                        name=metric,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ),
                    row=i+1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=300 * len(metrics),
            width=self.engine.config["visualization"]["default_width"],
            showlegend=True
        )
        
        # Update x-axis label for the bottom subplot only
        fig.update_xaxes(title_text="Date", row=len(metrics), col=1)
        
        return fig
    
    def create_alert_chart(
        self,
        df: pd.DataFrame,
        metric_column: str,
        threshold_upper: Optional[float] = None,
        threshold_lower: Optional[float] = None,
        date_column: str = "date",
        title: str = "Metric with Alert Thresholds"
    ) -> go.Figure:
        """
        Create a chart with alert thresholds highlighted.
        
        Args:
            df: DataFrame with metric data
            metric_column: Column containing the metric
            threshold_upper: Upper threshold value
            threshold_lower: Lower threshold value
            date_column: Date column name
            title: Chart title
            
        Returns:
            Plotly figure with thresholds
        """
        fig = go.Figure()
        
        # Add main metric line
        fig.add_trace(
            go.Scatter(
                x=df[date_column],
                y=df[metric_column],
                mode="lines",
                name=metric_column,
                line=dict(color="blue", width=2)
            )
        )
        
        # Add upper threshold if provided
        if threshold_upper is not None:
            fig.add_hline(
                y=threshold_upper,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Upper Threshold: {threshold_upper}"
            )
            
            # Highlight values above threshold
            above_threshold = df[df[metric_column] > threshold_upper]
            if not above_threshold.empty:
                fig.add_trace(
                    go.Scatter(
                        x=above_threshold[date_column],
                        y=above_threshold[metric_column],
                        mode="markers",
                        name="Above Threshold",
                        marker=dict(color="red", size=8)
                    )
                )
        
        # Add lower threshold if provided
        if threshold_lower is not None:
            fig.add_hline(
                y=threshold_lower,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Lower Threshold: {threshold_lower}"
            )
            
            # Highlight values below threshold
            below_threshold = df[df[metric_column] < threshold_lower]
            if not below_threshold.empty:
                fig.add_trace(
                    go.Scatter(
                        x=below_threshold[date_column],
                        y=below_threshold[metric_column],
                        mode="markers",
                        name="Below Threshold",
                        marker=dict(color="orange", size=8)
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=metric_column.replace("_", " ").title(),
            hovermode="x unified",
            width=self.engine.config["visualization"]["default_width"],
            height=self.engine.config["visualization"]["default_height"],
            showlegend=True
        )
        
        return fig

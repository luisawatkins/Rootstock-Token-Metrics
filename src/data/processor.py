"""
Data processing module for transforming and analyzing token metrics.

This module provides utilities for processing raw blockchain data into
formats suitable for visualization and analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and transform token data for visualization."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        logger.info("DataProcessor initialized")
    
    def calculate_moving_average(
        self,
        df: pd.DataFrame,
        column: str,
        window: int = 7
    ) -> pd.DataFrame:
        """
        Calculate moving average for a column.
        
        Args:
            df: Input DataFrame
            column: Column name to calculate MA for
            window: Window size for moving average
            
        Returns:
            DataFrame with additional MA column
        """
        df[f"{column}_ma{window}"] = df[column].rolling(window=window, min_periods=1).mean()
        return df
    
    def calculate_volatility(
        self,
        df: pd.DataFrame,
        price_column: str = "price",
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate price volatility.
        
        Args:
            df: DataFrame with price data
            price_column: Name of the price column
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility column added
        """
        # Calculate daily returns
        df["returns"] = df[price_column].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        df["volatility"] = df["returns"].rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        return df
    
    def calculate_rsi(
        self,
        df: pd.DataFrame,
        price_column: str = "price",
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with price data
            price_column: Name of the price column
            period: Period for RSI calculation
            
        Returns:
            DataFrame with RSI column added
        """
        delta = df[price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        return df
    
    def aggregate_by_time(
        self,
        df: pd.DataFrame,
        time_column: str,
        value_column: str,
        aggregation: str = "sum",
        freq: str = "D"
    ) -> pd.DataFrame:
        """
        Aggregate data by time periods.
        
        Args:
            df: Input DataFrame
            time_column: Column containing timestamps
            value_column: Column to aggregate
            aggregation: Aggregation method (sum, mean, count, etc.)
            freq: Frequency for resampling (D=daily, H=hourly, W=weekly)
            
        Returns:
            Aggregated DataFrame
        """
        df[time_column] = pd.to_datetime(df[time_column])
        df_resampled = df.set_index(time_column).resample(freq)[value_column].agg(aggregation)
        return df_resampled.reset_index()
    
    def calculate_holder_metrics(
        self,
        holder_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate holder distribution metrics.
        
        Args:
            holder_df: DataFrame with holder balances
            
        Returns:
            Dictionary with distribution metrics
        """
        if holder_df.empty:
            return {}
        
        metrics = {
            "total_holders": len(holder_df),
            "top_10_concentration": holder_df.head(10)["percentage"].sum() if "percentage" in holder_df else 0,
            "top_50_concentration": holder_df.head(50)["percentage"].sum() if "percentage" in holder_df else 0,
            "gini_coefficient": self._calculate_gini(holder_df["balance"].values) if "balance" in holder_df else 0,
            "median_balance": holder_df["balance"].median() if "balance" in holder_df else 0,
            "mean_balance": holder_df["balance"].mean() if "balance" in holder_df else 0,
            "std_balance": holder_df["balance"].std() if "balance" in holder_df else 0
        }
        
        # Calculate holder tiers
        if "balance" in holder_df:
            balance_values = holder_df["balance"].values
            metrics["whales"] = len(balance_values[balance_values > np.percentile(balance_values, 99)])
            metrics["large_holders"] = len(balance_values[(balance_values > np.percentile(balance_values, 95)) & 
                                                          (balance_values <= np.percentile(balance_values, 99))])
            metrics["medium_holders"] = len(balance_values[(balance_values > np.percentile(balance_values, 50)) & 
                                                           (balance_values <= np.percentile(balance_values, 95))])
            metrics["small_holders"] = len(balance_values[balance_values <= np.percentile(balance_values, 50)])
        
        return metrics
    
    def _calculate_gini(self, x: np.ndarray) -> float:
        """
        Calculate Gini coefficient for wealth distribution.
        
        Args:
            x: Array of wealth/balance values
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if len(x) == 0:
            return 0
        
        # Sort the array
        sorted_x = np.sort(x)
        n = len(x)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_x)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n
        
        return min(max(gini, 0), 1)  # Ensure between 0 and 1
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect anomalies in data.
        
        Args:
            df: Input DataFrame
            column: Column to analyze for anomalies
            method: Detection method (zscore, iqr)
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly column added
        """
        if method == "zscore":
            series = pd.to_numeric(df[column], errors="coerce")
            mean_val = series.mean()
            if pd.isna(mean_val):
                df["is_anomaly"] = False
                return df

            filled = series.fillna(mean_val)
            std_val = filled.std(ddof=0)
            if std_val == 0 or pd.isna(std_val):
                df["is_anomaly"] = False
                return df

            z_scores = np.abs((filled - mean_val) / std_val)
            df["is_anomaly"] = z_scores > threshold
        elif method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df["is_anomaly"] = (df[column] < lower_bound) | (df[column] > upper_bound)
        else:
            df["is_anomaly"] = False
        
        return df
    
    def calculate_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to include
            
        Returns:
            Correlation matrix as DataFrame
        """
        return df[columns].corr()
    
    def prepare_time_series(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        fill_method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Prepare time series data for visualization.
        
        Args:
            df: Input DataFrame
            date_column: Column containing dates
            value_columns: Columns with values to include
            fill_method: Method to fill missing values
            
        Returns:
            Cleaned and prepared DataFrame
        """
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Set date as index
        df = df.set_index(date_column)
        
        # Resample to ensure continuous time series
        df = df[value_columns].resample("D").mean()
        
        # Fill missing values
        if fill_method == "ffill":
            df = df.fillna(method="ffill")
        elif fill_method == "interpolate":
            df = df.interpolate(method="linear")
        elif fill_method == "zero":
            df = df.fillna(0)
        
        return df.reset_index()
    
    def calculate_growth_metrics(
        self,
        df: pd.DataFrame,
        value_column: str,
        periods: List[int] = [1, 7, 30]
    ) -> Dict[str, float]:
        """
        Calculate growth metrics over different periods.
        
        Args:
            df: DataFrame with time series data
            value_column: Column to calculate growth for
            periods: List of periods (in days) to calculate growth
            
        Returns:
            Dictionary with growth metrics
        """
        metrics = {}
        
        if len(df) < 2:
            return metrics
        
        current_value = df[value_column].iloc[-1]
        
        for period in periods:
            if len(df) > period:
                past_value = df[value_column].iloc[-period-1]
                if past_value != 0:
                    growth = ((current_value - past_value) / past_value) * 100
                    metrics[f"growth_{period}d"] = growth
                    metrics[f"growth_{period}d_abs"] = current_value - past_value
        
        # Calculate compound annual growth rate (CAGR) if we have enough data
        if len(df) > 365:
            initial_value = df[value_column].iloc[0]
            years = len(df) / 365
            if initial_value != 0:
                cagr = (pow(current_value / initial_value, 1/years) - 1) * 100
                metrics["cagr"] = cagr
        
        return metrics
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "minmax"
    ) -> pd.DataFrame:
        """
        Normalize data columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method (minmax, zscore)
            
        Returns:
            DataFrame with normalized columns
        """
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df[f"{col}_normalized"] = (df[col] - mean_val) / std_val
        
        return df
    
    def export_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        index: bool = False
    ) -> bool:
        """
        Export DataFrame to CSV file.
        
        Args:
            df: DataFrame to export
            filepath: Path to save the CSV file
            index: Whether to include index in export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df.to_csv(filepath, index=index)
            logger.info(f"Data exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

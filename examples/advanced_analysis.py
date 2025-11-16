#!/usr/bin/env python3
"""
Advanced analysis example for the Rootstock Token Metrics Visualization Tool.

This script demonstrates complex multi-token analysis, comparisons, and
advanced visualization techniques.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta

from src.data.fetcher import TokenDataFetcher
from src.data.processor import DataProcessor
from src.visualization.engine import VisualizationEngine
from src.visualization.charts import ChartFactory


def analyze_token_correlations(tokens_data):
    """Analyze correlations between multiple tokens."""
    processor = DataProcessor()
    
    # Prepare data for correlation analysis
    price_data = {}
    for token_name, data in tokens_data.items():
        if "price_df" in data and not data["price_df"].empty:
            price_data[token_name] = data["price_df"]["price"]
    
    if len(price_data) > 1:
        # Create combined DataFrame
        combined_df = pd.DataFrame(price_data)
        
        # Calculate correlation matrix
        correlation_matrix = processor.calculate_correlation_matrix(
            combined_df,
            list(combined_df.columns)
        )
        
        return correlation_matrix
    return None


def detect_market_anomalies(token_data):
    """Detect anomalies in token metrics."""
    processor = DataProcessor()
    anomalies = {}
    
    # Check for price anomalies
    if "price_df" in token_data and not token_data["price_df"].empty:
        price_df = processor.detect_anomalies(
            token_data["price_df"],
            "price",
            method="zscore",
            threshold=2.5
        )
        anomalies["price"] = price_df[price_df["is_anomaly"]]
    
    # Check for volume anomalies
    if "volume_df" in token_data and not token_data["volume_df"].empty:
        volume_df = processor.detect_anomalies(
            token_data["volume_df"],
            "volume",
            method="iqr",
            threshold=1.5
        )
        anomalies["volume"] = volume_df[volume_df["is_anomaly"]]
    
    return anomalies


def calculate_advanced_metrics(token_data):
    """Calculate advanced metrics for a token."""
    processor = DataProcessor()
    metrics = {}
    
    # Price metrics
    if "price_df" in token_data and not token_data["price_df"].empty:
        df = token_data["price_df"]
        
        # Add RSI
        df = processor.calculate_rsi(df, "price")
        metrics["current_rsi"] = df["rsi"].iloc[-1] if "rsi" in df else None
        
        # Add volatility
        df = processor.calculate_volatility(df, "price")
        metrics["current_volatility"] = df["volatility"].iloc[-1] if "volatility" in df else None
        
        # Growth metrics
        growth = processor.calculate_growth_metrics(df, "price", periods=[1, 7, 30])
        metrics.update(growth)
    
    # Holder metrics
    if "holders_df" in token_data and not token_data["holders_df"].empty:
        holder_metrics = processor.calculate_holder_metrics(token_data["holders_df"])
        metrics["holder_concentration"] = holder_metrics.get("top_10_concentration")
        metrics["gini_coefficient"] = holder_metrics.get("gini_coefficient")
    
    return metrics


def main():
    """Run advanced token analysis example."""
    
    print("🚀 Rootstock Token Metrics - Advanced Analysis Example")
    print("=" * 60)
    
    # Token addresses (using example addresses)
    TOKENS = {
        "RIF": "0x2acc95758f8b5F583470ba265eb685a8f45fc9d5",
        "WRBTC": "0x542fDA317318eBF1d3DEAf76E0b632741A7e677d",
        "DOC": "0xe700691da7b9851f2f35f8b8182c69c53ccad9db"
    }
    
    # Initialize components
    print("\n📊 Initializing components...")
    fetcher = TokenDataFetcher(network="mainnet")
    processor = DataProcessor()
    viz_engine = VisualizationEngine()
    chart_factory = ChartFactory()
    
    # Collect data for all tokens
    tokens_data = {}
    
    for token_name, token_address in TOKENS.items():
        print(f"\n📝 Analyzing {token_name} ({token_address[:10]}...):")
        
        token_data = {}
        
        # Get token info
        token_info = fetcher.get_token_info(token_address)
        token_data["info"] = token_info
        
        if token_info:
            print(f"  ✓ Token: {token_info.get('symbol')} - {token_info.get('name')}")
        
        # Get price data
        price_df = fetcher.get_price_data(token_address, days=30)
        token_data["price_df"] = price_df
        
        # Get volume data
        volume_df = fetcher.get_transaction_volume(token_address, days=30)
        token_data["volume_df"] = volume_df
        
        # Get holder distribution
        holders_df = fetcher.get_holder_distribution(token_address, top_n=100)
        token_data["holders_df"] = holders_df
        
        # Calculate advanced metrics
        metrics = calculate_advanced_metrics(token_data)
        token_data["metrics"] = metrics
        
        # Display key metrics
        if metrics:
            print(f"  📈 Current RSI: {metrics.get('current_rsi', 'N/A'):.2f}" if metrics.get('current_rsi') else "  📈 RSI: N/A")
            print(f"  📊 Volatility: {metrics.get('current_volatility', 0):.2%}")
            print(f"  📉 7d Growth: {metrics.get('growth_7d', 0):+.2f}%")
            print(f"  👥 Top 10 Concentration: {metrics.get('holder_concentration', 0):.2f}%")
            print(f"  📏 Gini Coefficient: {metrics.get('gini_coefficient', 0):.4f}")
        
        # Detect anomalies
        anomalies = detect_market_anomalies(token_data)
        token_data["anomalies"] = anomalies
        
        if anomalies.get("price") is not None and not anomalies["price"].empty:
            print(f"  ⚠️  Price anomalies detected: {len(anomalies['price'])} events")
        if anomalies.get("volume") is not None and not anomalies["volume"].empty:
            print(f"  ⚠️  Volume anomalies detected: {len(anomalies['volume'])} events")
        
        tokens_data[token_name] = token_data
    
    # Correlation Analysis
    print("\n🔗 Analyzing token correlations...")
    correlation_matrix = analyze_token_correlations(tokens_data)
    
    if correlation_matrix is not None:
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(3))
        
        # Create correlation heatmap
        corr_fig = viz_engine.create_correlation_heatmap(
            pd.DataFrame({name: data["price_df"]["price"] 
                         for name, data in tokens_data.items() 
                         if "price_df" in data and not data["price_df"].empty}),
            list(tokens_data.keys()),
            title="Token Price Correlations"
        )
        
        viz_engine.export_figure(corr_fig, "token_correlations.html", format="html")
        print("  ✅ Correlation heatmap saved to token_correlations.html")
    
    # Create Comparison Charts
    print("\n🎨 Creating comparison visualizations...")
    
    # Price comparison
    price_dfs = {}
    for name, data in tokens_data.items():
        if "price_df" in data and not data["price_df"].empty:
            # Normalize prices for comparison
            df = data["price_df"].copy()
            df = processor.normalize_data(df, ["price"], method="minmax")
            price_dfs[name] = df[["date", "price_normalized"]].rename(
                columns={"price_normalized": "price"}
            )
    
    if price_dfs:
        comparison_fig = chart_factory.create_comparison_chart(
            price_dfs,
            chart_type="price",
            title="Normalized Price Comparison (30 days)"
        )
        
        viz_engine.export_figure(comparison_fig, "price_comparison.html", format="html")
        print("  ✅ Price comparison chart saved to price_comparison.html")
    
    # Volume comparison
    volume_dfs = {}
    for name, data in tokens_data.items():
        if "volume_df" in data and not data["volume_df"].empty:
            volume_dfs[name] = data["volume_df"]
    
    if volume_dfs:
        volume_comparison_fig = chart_factory.create_comparison_chart(
            volume_dfs,
            chart_type="volume",
            title="Transaction Volume Comparison"
        )
        
        viz_engine.export_figure(volume_comparison_fig, "volume_comparison.html", format="html")
        print("  ✅ Volume comparison chart saved to volume_comparison.html")
    
    # Create Multi-Metric Dashboard for each token
    print("\n📊 Creating individual token dashboards...")
    
    for token_name, token_data in tokens_data.items():
        dashboard_data = {
            "price": token_data.get("price_df", pd.DataFrame()),
            "volume": token_data.get("volume_df", pd.DataFrame()),
            "holders": token_data.get("holders_df", pd.DataFrame()),
            "metrics": token_data.get("metrics", {})
        }
        
        dashboard_fig = viz_engine.create_dashboard(
            dashboard_data,
            title=f"{token_name} Token Metrics Dashboard"
        )
        
        filename = f"{token_name.lower()}_dashboard.html"
        viz_engine.export_figure(dashboard_fig, filename, format="html")
        print(f"  ✅ {token_name} dashboard saved to {filename}")
    
    # Create Alert Chart for Anomalies
    print("\n🚨 Creating anomaly detection charts...")
    
    for token_name, token_data in tokens_data.items():
        if token_data.get("anomalies"):
            # Price anomaly chart
            if "price" in token_data["anomalies"] and "price_df" in token_data:
                alert_fig = chart_factory.create_alert_chart(
                    token_data["price_df"],
                    "price",
                    threshold_upper=token_data["price_df"]["price"].mean() + 2 * token_data["price_df"]["price"].std(),
                    threshold_lower=token_data["price_df"]["price"].mean() - 2 * token_data["price_df"]["price"].std(),
                    title=f"{token_name} Price Anomalies"
                )
                
                filename = f"{token_name.lower()}_price_anomalies.html"
                viz_engine.export_figure(alert_fig, filename, format="html")
                print(f"  ✅ {token_name} anomaly chart saved to {filename}")
    
    # Generate Summary Report
    print("\n📄 Generating summary report...")
    
    report_data = []
    for token_name, token_data in tokens_data.items():
        metrics = token_data.get("metrics", {})
        report_data.append({
            "Token": token_name,
            "Symbol": token_data.get("info", {}).get("symbol", "N/A"),
            "RSI": f"{metrics.get('current_rsi', 0):.2f}" if metrics.get('current_rsi') else "N/A",
            "Volatility": f"{metrics.get('current_volatility', 0):.2%}",
            "1d Growth": f"{metrics.get('growth_1d', 0):+.2f}%",
            "7d Growth": f"{metrics.get('growth_7d', 0):+.2f}%",
            "30d Growth": f"{metrics.get('growth_30d', 0):+.2f}%",
            "Top 10 Conc.": f"{metrics.get('holder_concentration', 0):.2f}%",
            "Gini Coeff.": f"{metrics.get('gini_coefficient', 0):.4f}"
        })
    
    report_df = pd.DataFrame(report_data)
    processor.export_to_csv(report_df, "token_analysis_report.csv")
    print("  ✅ Summary report saved to token_analysis_report.csv")
    
    print("\n" + "=" * 60)
    print("✨ Advanced analysis complete!")
    print("\nGenerated files:")
    print("  - token_correlations.html")
    print("  - price_comparison.html")
    print("  - volume_comparison.html")
    print("  - [token]_dashboard.html (for each token)")
    print("  - [token]_price_anomalies.html (where anomalies detected)")
    print("  - token_analysis_report.csv")
    print("\nOpen the HTML files in your browser to view interactive charts.")


if __name__ == "__main__":
    main()

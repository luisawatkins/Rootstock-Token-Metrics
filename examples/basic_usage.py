#!/usr/bin/env python3
"""
Basic usage example for the Rootstock Token Metrics Visualization Tool.

This script demonstrates simple token analysis and visualization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.fetcher import TokenDataFetcher
from src.data.processor import DataProcessor
from src.visualization.engine import VisualizationEngine


def main():
    """Run basic token analysis example."""
    
    # RIF Token address on Rootstock mainnet
    RIF_TOKEN = "0x2acc95758f8b5F583470ba265eb685a8f45fc9d5"
    
    print("🚀 Rootstock Token Metrics - Basic Usage Example")
    print("=" * 50)
    
    # Initialize components
    print("\n📊 Initializing components...")
    fetcher = TokenDataFetcher(network="mainnet")
    processor = DataProcessor()
    viz_engine = VisualizationEngine()
    
    # 1. Get Token Information
    print(f"\n📝 Fetching token information for {RIF_TOKEN[:10]}...")
    token_info = fetcher.get_token_info(RIF_TOKEN)
    
    if token_info:
        print(f"  Name: {token_info.get('name')}")
        print(f"  Symbol: {token_info.get('symbol')}")
        print(f"  Decimals: {token_info.get('decimals')}")
        print(f"  Total Supply: {token_info.get('total_supply_formatted'):,.2f}")
    
    # 2. Get Price Data (mock data for demonstration)
    print("\n📈 Fetching price data (last 30 days)...")
    price_df = fetcher.get_price_data(RIF_TOKEN, days=30)
    
    if not price_df.empty:
        print(f"  Data points: {len(price_df)}")
        print(f"  Current price: ${price_df['price'].iloc[-1]:.2f}")
        print(f"  Average price: ${price_df['price'].mean():.2f}")
        print(f"  Max price: ${price_df['price'].max():.2f}")
        print(f"  Min price: ${price_df['price'].min():.2f}")
        
        # Add technical indicators
        price_df = processor.calculate_moving_average(price_df, "price", window=7)
        price_df = processor.calculate_volatility(price_df, "price")
        
        print(f"  Current volatility: {price_df['volatility'].iloc[-1]:.2%}")
    
    # 3. Get Transaction Volume
    print("\n📊 Analyzing transaction volume...")
    volume_df = fetcher.get_transaction_volume(RIF_TOKEN, days=7, interval="daily")
    
    if not volume_df.empty:
        print(f"  Total volume (7d): {volume_df['volume'].sum():,.2f}")
        print(f"  Average daily volume: {volume_df['volume'].mean():,.2f}")
        print(f"  Total transactions: {volume_df['transaction_count'].sum()}")
    
    # 4. Get Holder Distribution
    print("\n👥 Analyzing holder distribution...")
    holders_df = fetcher.get_holder_distribution(RIF_TOKEN, top_n=100)
    
    if not holders_df.empty:
        metrics = processor.calculate_holder_metrics(holders_df)
        print(f"  Total holders analyzed: {metrics.get('total_holders')}")
        print(f"  Top 10 concentration: {metrics.get('top_10_concentration'):.2f}%")
        print(f"  Gini coefficient: {metrics.get('gini_coefficient'):.4f}")
        print(f"  Whale accounts: {metrics.get('whales')}")
    
    # 5. Create Visualizations
    print("\n🎨 Creating visualizations...")
    
    # Price chart with moving averages
    if not price_df.empty:
        price_fig = viz_engine.create_price_chart(
            price_df,
            title="RIF Token Price History",
            show_ma=True,
            ma_periods=[7, 30]
        )
        
        # Save chart
        if viz_engine.export_figure(price_fig, "rif_price_chart.html", format="html"):
            print("  ✅ Price chart saved to rif_price_chart.html")
    
    # Volume chart
    if not volume_df.empty:
        volume_fig = viz_engine.create_volume_chart(
            volume_df,
            title="RIF Transaction Volume"
        )
        
        if viz_engine.export_figure(volume_fig, "rif_volume_chart.html", format="html"):
            print("  ✅ Volume chart saved to rif_volume_chart.html")
    
    # Holder distribution chart
    if not holders_df.empty:
        holder_fig = viz_engine.create_holder_distribution_chart(
            holders_df,
            top_n=20,
            title="Top 20 RIF Token Holders"
        )
        
        if viz_engine.export_figure(holder_fig, "rif_holders_chart.html", format="html"):
            print("  ✅ Holder chart saved to rif_holders_chart.html")
    
    # 6. Export Data
    print("\n💾 Exporting data to CSV...")
    if not price_df.empty:
        processor.export_to_csv(price_df, "rif_price_data.csv")
        print("  ✅ Price data exported to rif_price_data.csv")
    
    if not holders_df.empty:
        processor.export_to_csv(holders_df, "rif_holders_data.csv")
        print("  ✅ Holder data exported to rif_holders_data.csv")
    
    print("\n✨ Analysis complete!")
    print("Check the generated files for charts and data exports.")


if __name__ == "__main__":
    main()

"""
Command-line interface for the Rootstock Token Metrics Visualization Tool.

This module provides a CLI for interacting with the tool, allowing users to
fetch data, create visualizations, and export results from the command line.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fetcher import TokenDataFetcher
from src.data.processor import DataProcessor
from src.visualization.engine import VisualizationEngine
from src.visualization.charts import ChartFactory

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to configuration file")
@click.option("--network", "-n", default="mainnet", type=click.Choice(["mainnet", "testnet"]))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, network, verbose):
    """Rootstock Token Metrics Visualization Tool CLI"""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Store config in context
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["network"] = network
    
    console.print(f"[bold green]Rootstock Token Metrics Tool[/bold green]")
    console.print(f"Network: [cyan]{network}[/cyan]")
    console.print(f"Config: [cyan]{config}[/cyan]\n")


@cli.command()
@click.argument("token_address")
@click.pass_context
def info(ctx, token_address):
    """Get basic information about a token"""
    with console.status("[bold green]Fetching token information..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        token_info = fetcher.get_token_info(token_address)
    
    if token_info:
        table = Table(title="Token Information", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in token_info.items():
            if key != "total_supply":  # Skip raw total supply
                table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
    else:
        console.print("[red]Failed to fetch token information[/red]")


@cli.command()
@click.argument("token_address")
@click.argument("wallet_address")
@click.pass_context
def balance(ctx, token_address, wallet_address):
    """Get token balance for a wallet"""
    with console.status("[bold green]Fetching balance..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        balance_info = fetcher.get_token_balance(token_address, wallet_address)
    
    if balance_info:
        console.print(f"[green]Token:[/green] {token_address}")
        console.print(f"[green]Wallet:[/green] {wallet_address}")
        console.print(f"[bold cyan]Balance:[/bold cyan] {balance_info['balance']:.6f}")
    else:
        console.print("[red]Failed to fetch balance[/red]")


@cli.command()
@click.argument("token_address")
@click.option("--days", "-d", default=30, help="Number of days to analyze")
@click.option("--interval", "-i", default="daily", type=click.Choice(["daily", "hourly"]))
@click.option("--export", "-e", help="Export data to CSV file")
@click.pass_context
def volume(ctx, token_address, days, interval, export):
    """Analyze transaction volume for a token"""
    with console.status(f"[bold green]Analyzing {days} days of volume data..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        volume_df = fetcher.get_transaction_volume(token_address, days, interval)
    
    if not volume_df.empty:
        # Display summary
        table = Table(title="Volume Analysis", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Volume", f"{volume_df['volume'].sum():,.2f}")
        table.add_row("Average Daily Volume", f"{volume_df['volume'].mean():,.2f}")
        table.add_row("Max Volume", f"{volume_df['volume'].max():,.2f}")
        table.add_row("Min Volume", f"{volume_df['volume'].min():,.2f}")
        table.add_row("Total Transactions", str(volume_df['transaction_count'].sum()))
        
        console.print(table)
        
        # Export if requested
        if export:
            processor = DataProcessor()
            if processor.export_to_csv(volume_df, export):
                console.print(f"[green]Data exported to {export}[/green]")
    else:
        console.print("[red]No volume data found[/red]")


@cli.command()
@click.argument("token_address")
@click.option("--top", "-t", default=20, help="Number of top holders to show")
@click.option("--export", "-e", help="Export data to CSV file")
@click.pass_context
def holders(ctx, token_address, top, export):
    """Analyze token holder distribution"""
    with console.status(f"[bold green]Fetching top {top} holders..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        holders_df = fetcher.get_holder_distribution(token_address, top)
    
    if not holders_df.empty:
        # Calculate metrics
        processor = DataProcessor()
        metrics = processor.calculate_holder_metrics(holders_df)
        
        # Display metrics
        metrics_table = Table(title="Holder Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Total Holders", str(metrics.get("total_holders", 0)))
        metrics_table.add_row("Top 10 Concentration", f"{metrics.get('top_10_concentration', 0):.2f}%")
        metrics_table.add_row("Gini Coefficient", f"{metrics.get('gini_coefficient', 0):.4f}")
        metrics_table.add_row("Median Balance", f"{metrics.get('median_balance', 0):.6f}")
        
        console.print(metrics_table)
        
        # Display top holders
        holders_table = Table(title=f"Top {min(10, len(holders_df))} Holders", show_header=True)
        holders_table.add_column("#", style="cyan")
        holders_table.add_column("Address", style="yellow")
        holders_table.add_column("Balance", style="green")
        holders_table.add_column("Percentage", style="magenta")
        
        for i, row in holders_df.head(10).iterrows():
            address_short = f"{row['address'][:6]}...{row['address'][-4:]}"
            holders_table.add_row(
                str(i + 1),
                address_short,
                f"{row['balance']:,.2f}",
                f"{row['percentage']:.2f}%"
            )
        
        console.print(holders_table)
        
        # Export if requested
        if export:
            if processor.export_to_csv(holders_df, export):
                console.print(f"[green]Data exported to {export}[/green]")
    else:
        console.print("[red]No holder data found[/red]")


@cli.command()
@click.argument("token_address")
@click.option("--chart-type", "-t", default="price", 
              type=click.Choice(["price", "volume", "holders", "histogram"]))
@click.option("--days", "-d", default=30, help="Number of days for time series charts")
@click.option("--output", "-o", help="Output file for chart (HTML or image)")
@click.option("--show", "-s", is_flag=True, help="Open chart in browser")
@click.pass_context
def visualize(ctx, token_address, chart_type, days, output, show):
    """Create and display visualizations for token metrics"""
    with console.status(f"[bold green]Creating {chart_type} visualization..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        chart_factory = ChartFactory(ctx.obj["config"])
        
        # Fetch appropriate data based on chart type
        if chart_type == "price":
            df = fetcher.get_price_data(token_address, days)
            if not df.empty:
                fig = chart_factory.create_chart("price", df)
            else:
                console.print("[red]No price data available[/red]")
                return
        
        elif chart_type == "volume":
            df = fetcher.get_transaction_volume(token_address, days)
            if not df.empty:
                fig = chart_factory.create_chart("volume", df)
            else:
                console.print("[red]No volume data available[/red]")
                return
        
        elif chart_type == "holders":
            df = fetcher.get_holder_distribution(token_address)
            if not df.empty:
                fig = chart_factory.create_chart("holders", df)
            else:
                console.print("[red]No holder data available[/red]")
                return
        
        elif chart_type == "histogram":
            df = fetcher.get_holder_distribution(token_address, top_n=1000)
            if not df.empty:
                fig = chart_factory.create_chart("histogram", df, value_column="balance")
            else:
                console.print("[red]No data available for histogram[/red]")
                return
        
        # Save or show the chart
        if fig:
            if output:
                format = output.split(".")[-1] if "." in output else "html"
                engine = VisualizationEngine(ctx.obj["config"])
                if engine.export_figure(fig, output, format):
                    console.print(f"[green]Chart saved to {output}[/green]")
                else:
                    console.print(f"[red]Failed to save chart[/red]")
            
            if show or not output:
                fig.show()
                console.print("[green]Chart opened in browser[/green]")


@cli.command()
@click.argument("token_addresses", nargs=-1, required=True)
@click.option("--metric", "-m", default="price", 
              type=click.Choice(["price", "volume"]))
@click.option("--days", "-d", default=30, help="Number of days to compare")
@click.option("--output", "-o", help="Output file for comparison chart")
@click.pass_context
def compare(ctx, token_addresses, metric, days, output):
    """Compare metrics across multiple tokens"""
    with console.status(f"[bold green]Comparing {metric} for {len(token_addresses)} tokens..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        chart_factory = ChartFactory(ctx.obj["config"])
        
        # Fetch data for each token
        token_data = {}
        for token_address in track(token_addresses, description="Fetching data..."):
            if metric == "price":
                df = fetcher.get_price_data(token_address, days)
            else:
                df = fetcher.get_transaction_volume(token_address, days)
            
            if not df.empty:
                # Get token symbol for labeling
                token_info = fetcher.get_token_info(token_address)
                token_name = token_info.get("symbol", token_address[:8])
                token_data[token_name] = df
        
        if token_data:
            # Create comparison chart
            fig = chart_factory.create_comparison_chart(
                token_data,
                chart_type=metric,
                title=f"{metric.title()} Comparison"
            )
            
            if output:
                format = output.split(".")[-1] if "." in output else "html"
                engine = VisualizationEngine(ctx.obj["config"])
                if engine.export_figure(fig, output, format):
                    console.print(f"[green]Comparison chart saved to {output}[/green]")
            else:
                fig.show()
                console.print("[green]Comparison chart opened in browser[/green]")
        else:
            console.print("[red]No data available for comparison[/red]")


@cli.command()
@click.pass_context
def network(ctx):
    """Display current network metrics"""
    with console.status("[bold green]Fetching network metrics..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        metrics = fetcher.get_network_metrics()
    
    if metrics:
        table = Table(title="Network Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Network", metrics.get("network", "Unknown"))
        table.add_row("Connected", str(metrics.get("connected", False)))
        table.add_row("Block Number", str(metrics.get("block_number", 0)))
        table.add_row("Block Time", str(metrics.get("block_timestamp", "")))
        table.add_row("Gas Price", f"{metrics.get('gas_price', 0)} wei")
        table.add_row("Chain ID", str(metrics.get("chain_id", 0)))
        
        console.print(table)
    else:
        console.print("[red]Failed to fetch network metrics[/red]")


@cli.command()
@click.argument("token_address")
@click.option("--metric", "-m", default="volume", help="Metric to analyze")
@click.option("--threshold", "-t", type=float, help="Alert threshold value")
@click.option("--days", "-d", default=30, help="Number of days to analyze")
@click.pass_context
def alert(ctx, token_address, metric, threshold, days):
    """Check for anomalies and threshold breaches"""
    with console.status(f"[bold green]Analyzing {metric} for alerts..."):
        fetcher = TokenDataFetcher(ctx.obj["config"], ctx.obj["network"])
        processor = DataProcessor()
        
        # Fetch data
        if metric == "volume":
            df = fetcher.get_transaction_volume(token_address, days)
            value_column = "volume"
        elif metric == "price":
            df = fetcher.get_price_data(token_address, days)
            value_column = "price"
        else:
            console.print(f"[red]Unknown metric: {metric}[/red]")
            return
        
        if not df.empty:
            # Detect anomalies
            df = processor.detect_anomalies(df, value_column)
            anomalies = df[df["is_anomaly"]]
            
            # Check threshold breaches if provided
            if threshold:
                breaches = df[df[value_column] > threshold]
                
                if not breaches.empty:
                    console.print(f"[bold red]⚠️  Alert: {len(breaches)} threshold breaches detected![/bold red]")
                    
                    table = Table(title="Threshold Breaches", show_header=True)
                    table.add_column("Date", style="cyan")
                    table.add_column(metric.title(), style="red")
                    
                    for _, row in breaches.head(10).iterrows():
                        date_str = str(row.get("date", row.get("timestamp", "")))
                        table.add_row(date_str, f"{row[value_column]:,.2f}")
                    
                    console.print(table)
            
            # Display anomalies
            if not anomalies.empty:
                console.print(f"[bold yellow]📊 {len(anomalies)} anomalies detected[/bold yellow]")
                
                anomaly_table = Table(title="Anomalies", show_header=True)
                anomaly_table.add_column("Date", style="cyan")
                anomaly_table.add_column(metric.title(), style="yellow")
                
                for _, row in anomalies.head(10).iterrows():
                    date_str = str(row.get("date", row.get("timestamp", "")))
                    anomaly_table.add_row(date_str, f"{row[value_column]:,.2f}")
                
                console.print(anomaly_table)
            else:
                console.print("[green]✓ No anomalies detected[/green]")
        else:
            console.print("[red]No data available for analysis[/red]")


def main():
    """Main entry point for the CLI"""
    try:
        cli(obj={})
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.error(f"CLI error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

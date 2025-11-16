#!/usr/bin/env python3
"""
Automated reporting example for the Rootstock Token Metrics Visualization Tool.

This script demonstrates how to set up automated reports that can be scheduled
to run periodically (e.g., daily, weekly) to monitor token metrics.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
import schedule
import time

from src.data.fetcher import TokenDataFetcher
from src.data.processor import DataProcessor
from src.visualization.engine import VisualizationEngine
from src.visualization.charts import ChartFactory


class TokenMetricsReporter:
    """Automated reporter for token metrics."""
    
    def __init__(self, tokens, network="mainnet"):
        """
        Initialize the reporter.
        
        Args:
            tokens: Dictionary of token names and addresses
            network: Network to use
        """
        self.tokens = tokens
        self.network = network
        
        # Initialize components
        self.fetcher = TokenDataFetcher(network=network)
        self.processor = DataProcessor()
        self.viz_engine = VisualizationEngine()
        self.chart_factory = ChartFactory()
        
        # Report settings
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_daily_report(self):
        """Generate daily token metrics report."""
        print(f"\n{'='*60}")
        print(f"📊 Generating Daily Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        report_date = datetime.now().strftime("%Y%m%d")
        report_data = {}
        
        for token_name, token_address in self.tokens.items():
            print(f"\n📝 Analyzing {token_name}...")
            
            try:
                # Fetch data
                token_info = self.fetcher.get_token_info(token_address)
                price_df = self.fetcher.get_price_data(token_address, days=1)
                volume_df = self.fetcher.get_transaction_volume(token_address, days=1, interval="hourly")
                
                # Calculate metrics
                metrics = {}
                
                if not price_df.empty:
                    current_price = price_df["price"].iloc[-1]
                    price_change_24h = ((price_df["price"].iloc[-1] - price_df["price"].iloc[0]) / 
                                       price_df["price"].iloc[0] * 100)
                    
                    metrics["current_price"] = current_price
                    metrics["price_change_24h"] = price_change_24h
                    
                    # Add volatility
                    price_df = self.processor.calculate_volatility(price_df, "price", window=24)
                    metrics["volatility_24h"] = price_df["volatility"].iloc[-1] if "volatility" in price_df else 0
                
                if not volume_df.empty:
                    metrics["volume_24h"] = volume_df["volume"].sum()
                    metrics["transactions_24h"] = volume_df["transaction_count"].sum()
                    metrics["avg_hourly_volume"] = volume_df["volume"].mean()
                
                report_data[token_name] = {
                    "info": token_info,
                    "metrics": metrics,
                    "price_data": price_df.to_dict() if not price_df.empty else None,
                    "volume_data": volume_df.to_dict() if not volume_df.empty else None
                }
                
                print(f"  ✅ {token_name} analysis complete")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {token_name}: {e}")
                report_data[token_name] = {"error": str(e)}
        
        # Generate report files
        self._save_report(report_data, report_date)
        self._create_visualizations(report_data, report_date)
        self._generate_summary_csv(report_data, report_date)
        
        print(f"\n✅ Daily report generated successfully!")
        return report_data
    
    def generate_weekly_report(self):
        """Generate weekly token metrics report."""
        print(f"\n{'='*60}")
        print(f"📊 Generating Weekly Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        report_date = datetime.now().strftime("%Y%m%d")
        report_data = {}
        
        for token_name, token_address in self.tokens.items():
            print(f"\n📝 Analyzing {token_name} (7-day period)...")
            
            try:
                # Fetch weekly data
                token_info = self.fetcher.get_token_info(token_address)
                price_df = self.fetcher.get_price_data(token_address, days=7)
                volume_df = self.fetcher.get_transaction_volume(token_address, days=7)
                holders_df = self.fetcher.get_holder_distribution(token_address, top_n=50)
                
                # Calculate weekly metrics
                metrics = {}
                
                if not price_df.empty:
                    # Price metrics
                    metrics["price_high_7d"] = price_df["price"].max()
                    metrics["price_low_7d"] = price_df["price"].min()
                    metrics["price_avg_7d"] = price_df["price"].mean()
                    
                    # Calculate growth
                    growth = self.processor.calculate_growth_metrics(price_df, "price", periods=[7])
                    metrics.update(growth)
                    
                    # Add technical indicators
                    price_df = self.processor.calculate_rsi(price_df, "price")
                    metrics["rsi_current"] = price_df["rsi"].iloc[-1] if "rsi" in price_df else None
                
                if not volume_df.empty:
                    metrics["volume_total_7d"] = volume_df["volume"].sum()
                    metrics["volume_avg_daily"] = volume_df["volume"].mean()
                    metrics["transactions_total_7d"] = volume_df["transaction_count"].sum()
                    
                    # Detect volume anomalies
                    volume_df = self.processor.detect_anomalies(volume_df, "volume")
                    metrics["volume_anomalies"] = volume_df["is_anomaly"].sum()
                
                if not holders_df.empty:
                    holder_metrics = self.processor.calculate_holder_metrics(holders_df)
                    metrics.update({
                        "holder_concentration": holder_metrics.get("top_10_concentration"),
                        "gini_coefficient": holder_metrics.get("gini_coefficient"),
                        "total_holders": holder_metrics.get("total_holders")
                    })
                
                report_data[token_name] = {
                    "info": token_info,
                    "metrics": metrics,
                    "price_data": price_df,
                    "volume_data": volume_df,
                    "holders_data": holders_df
                }
                
                print(f"  ✅ {token_name} weekly analysis complete")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {token_name}: {e}")
                report_data[token_name] = {"error": str(e)}
        
        # Generate comprehensive weekly report
        self._save_weekly_report(report_data, report_date)
        self._create_weekly_visualizations(report_data, report_date)
        
        print(f"\n✅ Weekly report generated successfully!")
        return report_data
    
    def _save_report(self, report_data, report_date):
        """Save report data to JSON file."""
        report_file = self.report_dir / f"daily_report_{report_date}.json"
        
        # Convert DataFrames to dictionaries for JSON serialization
        json_data = {}
        for token, data in report_data.items():
            json_data[token] = {
                "info": data.get("info", {}),
                "metrics": data.get("metrics", {}),
                "timestamp": datetime.now().isoformat()
            }
        
        with open(report_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"  📄 Report saved to {report_file}")
    
    def _save_weekly_report(self, report_data, report_date):
        """Save weekly report with additional analysis."""
        report_file = self.report_dir / f"weekly_report_{report_date}.json"
        
        json_data = {}
        for token, data in report_data.items():
            if "error" not in data:
                json_data[token] = {
                    "info": data.get("info", {}),
                    "metrics": data.get("metrics", {}),
                    "period": "7_days",
                    "timestamp": datetime.now().isoformat()
                }
        
        with open(report_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"  📄 Weekly report saved to {report_file}")
    
    def _create_visualizations(self, report_data, report_date):
        """Create visualization charts for the report."""
        charts_dir = self.report_dir / f"charts_{report_date}"
        charts_dir.mkdir(exist_ok=True)
        
        for token_name, data in report_data.items():
            if "error" in data:
                continue
            
            # Create price chart if data available
            if data.get("price_data"):
                price_df = pd.DataFrame(data["price_data"])
                if not price_df.empty:
                    fig = self.viz_engine.create_price_chart(
                        price_df,
                        title=f"{token_name} - 24h Price Movement"
                    )
                    chart_file = charts_dir / f"{token_name.lower()}_price_24h.html"
                    self.viz_engine.export_figure(fig, str(chart_file), format="html")
            
            # Create volume chart if data available
            if data.get("volume_data"):
                volume_df = pd.DataFrame(data["volume_data"])
                if not volume_df.empty:
                    fig = self.viz_engine.create_volume_chart(
                        volume_df,
                        title=f"{token_name} - 24h Volume"
                    )
                    chart_file = charts_dir / f"{token_name.lower()}_volume_24h.html"
                    self.viz_engine.export_figure(fig, str(chart_file), format="html")
        
        print(f"  📊 Charts saved to {charts_dir}")
    
    def _create_weekly_visualizations(self, report_data, report_date):
        """Create comprehensive weekly visualizations."""
        charts_dir = self.report_dir / f"weekly_charts_{report_date}"
        charts_dir.mkdir(exist_ok=True)
        
        # Create comparison charts
        price_dfs = {}
        volume_dfs = {}
        
        for token_name, data in report_data.items():
            if "error" not in data:
                if isinstance(data.get("price_data"), pd.DataFrame):
                    price_dfs[token_name] = data["price_data"]
                if isinstance(data.get("volume_data"), pd.DataFrame):
                    volume_dfs[token_name] = data["volume_data"]
        
        # Price comparison
        if price_dfs:
            fig = self.chart_factory.create_comparison_chart(
                price_dfs,
                chart_type="price",
                title="Weekly Price Comparison"
            )
            self.viz_engine.export_figure(fig, str(charts_dir / "price_comparison.html"), format="html")
        
        # Volume comparison
        if volume_dfs:
            fig = self.chart_factory.create_comparison_chart(
                volume_dfs,
                chart_type="volume",
                title="Weekly Volume Comparison"
            )
            self.viz_engine.export_figure(fig, str(charts_dir / "volume_comparison.html"), format="html")
        
        # Individual token dashboards
        for token_name, data in report_data.items():
            if "error" not in data:
                dashboard_data = {
                    "price": data.get("price_data", pd.DataFrame()),
                    "volume": data.get("volume_data", pd.DataFrame()),
                    "holders": data.get("holders_data", pd.DataFrame()),
                    "metrics": data.get("metrics", {})
                }
                
                fig = self.viz_engine.create_dashboard(
                    dashboard_data,
                    title=f"{token_name} - Weekly Dashboard"
                )
                
                chart_file = charts_dir / f"{token_name.lower()}_dashboard.html"
                self.viz_engine.export_figure(fig, str(chart_file), format="html")
        
        print(f"  📊 Weekly charts saved to {charts_dir}")
    
    def _generate_summary_csv(self, report_data, report_date):
        """Generate summary CSV file."""
        summary_data = []
        
        for token_name, data in report_data.items():
            if "error" not in data and "metrics" in data:
                metrics = data["metrics"]
                summary_data.append({
                    "Token": token_name,
                    "Symbol": data.get("info", {}).get("symbol", "N/A"),
                    "Current Price": metrics.get("current_price", "N/A"),
                    "24h Change (%)": f"{metrics.get('price_change_24h', 0):.2f}",
                    "24h Volume": metrics.get("volume_24h", "N/A"),
                    "24h Transactions": metrics.get("transactions_24h", "N/A"),
                    "Volatility": f"{metrics.get('volatility_24h', 0):.2%}" if metrics.get('volatility_24h') else "N/A",
                    "Report Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_file = self.report_dir / f"summary_{report_date}.csv"
            summary_df.to_csv(csv_file, index=False)
            print(f"  📋 Summary CSV saved to {csv_file}")
    
    def send_email_report(self, report_data, recipient_email, sender_email=None, sender_password=None):
        """
        Send report via email (requires email configuration).
        
        Args:
            report_data: Report data to send
            recipient_email: Email address to send to
            sender_email: Sender email address
            sender_password: Sender email password
        """
        if not sender_email or not sender_password:
            print("  ⚠️  Email credentials not configured")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"Token Metrics Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Create email body
        body = "Token Metrics Daily Report\n" + "="*40 + "\n\n"
        
        for token_name, data in report_data.items():
            if "metrics" in data:
                metrics = data["metrics"]
                body += f"\n{token_name}:\n"
                body += f"  Current Price: ${metrics.get('current_price', 'N/A'):.2f}\n"
                body += f"  24h Change: {metrics.get('price_change_24h', 0):+.2f}%\n"
                body += f"  24h Volume: {metrics.get('volume_24h', 'N/A'):,.2f}\n"
                body += f"  Transactions: {metrics.get('transactions_24h', 'N/A')}\n"
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            print(f"  ✉️  Report sent to {recipient_email}")
        except Exception as e:
            print(f"  ❌ Failed to send email: {e}")


def main():
    """Run automated reporting example."""
    
    print("🤖 Rootstock Token Metrics - Automated Reporting")
    print("=" * 60)
    
    # Configure tokens to monitor
    TOKENS = {
        "RIF": "0x2acc95758f8b5F583470ba265eb685a8f45fc9d5",
        "WRBTC": "0x542fDA317318eBF1d3DEAf76E0b632741A7e677d",
        "DOC": "0xe700691da7b9851f2f35f8b8182c69c53ccad9db"
    }
    
    # Initialize reporter
    reporter = TokenMetricsReporter(TOKENS, network="mainnet")
    
    # Generate immediate reports for demonstration
    print("\n📊 Generating immediate reports for demonstration...")
    
    # Daily report
    daily_data = reporter.generate_daily_report()
    
    # Weekly report
    weekly_data = reporter.generate_weekly_report()
    
    # Schedule automated reports (commented out for example)
    print("\n⏰ Scheduling automated reports...")
    print("  Daily report: Every day at 09:00")
    print("  Weekly report: Every Monday at 10:00")
    
    # Uncomment to enable scheduling
    # schedule.every().day.at("09:00").do(reporter.generate_daily_report)
    # schedule.every().monday.at("10:00").do(reporter.generate_weekly_report)
    
    # Example of how to run scheduled tasks
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)  # Check every minute
    
    print("\n✨ Automated reporting setup complete!")
    print(f"\nReports saved to: {reporter.report_dir}")
    print("\nTo enable continuous reporting, uncomment the scheduling code.")


if __name__ == "__main__":
    main()

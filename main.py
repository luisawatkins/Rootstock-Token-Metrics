#!/usr/bin/env python3
"""
Main entry point for the Rootstock Token Metrics Visualization Tool.

This script provides a unified interface to access all features of the tool.
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Rootstock Token Metrics Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLI interface
  python main.py cli info 0x2acc95758f8b5F583470ba265eb685a8f45fc9d5
  
  # Start web dashboard
  python main.py web --port 8050
  
  # Run example scripts
  python main.py example basic
  python main.py example advanced
  
For more information, see README.md
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # CLI mode
    cli_parser = subparsers.add_parser("cli", help="Run command-line interface")
    cli_parser.add_argument("args", nargs="*", help="CLI arguments")
    
    # Web mode
    web_parser = subparsers.add_parser("web", help="Start web dashboard")
    web_parser.add_argument("--port", type=int, default=8050, help="Port to run on")
    web_parser.add_argument("--network", default="mainnet", choices=["mainnet", "testnet"])
    web_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    # Example mode
    example_parser = subparsers.add_parser("example", help="Run example scripts")
    example_parser.add_argument(
        "script",
        choices=["basic", "advanced", "reporting"],
        help="Example script to run"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Execute based on mode
    if args.mode == "cli":
        # Run CLI with provided arguments
        from src.interfaces.cli import main as cli_main
        
        # Modify sys.argv for Click
        sys.argv = ["cli"] + args.args
        cli_main()
    
    elif args.mode == "web":
        # Start web dashboard
        from src.interfaces.web import TokenMetricsDashboard
        
        print("🚀 Starting Rootstock Token Metrics Dashboard...")
        print(f"Network: {args.network}")
        print(f"Access the dashboard at: http://localhost:{args.port}")
        print("\nPress Ctrl+C to stop the server")
        
        dashboard = TokenMetricsDashboard(network=args.network)
        dashboard.run(debug=args.debug, port=args.port)
    
    elif args.mode == "example":
        # Run example scripts
        if args.script == "basic":
            from examples.basic_usage import main as basic_main
            basic_main()
        elif args.script == "advanced":
            from examples.advanced_analysis import main as advanced_main
            advanced_main()
        elif args.script == "reporting":
            from examples.automated_reporting import main as reporting_main
            reporting_main()


if __name__ == "__main__":
    main()

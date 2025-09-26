#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring utilities for Trading AI system
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from data.data_fetcher import get_daily_adjusted

# Try to import scrapers, create fallback if not available
try:
    from scrapers.stock_scraper import StockDataScraper
except ImportError:
    # Create a simple fallback scraper
    class StockDataScraper:
        def get_real_time_price(self, symbol: str):
            try:
                import yfinance as yf
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                if not hist.empty:
                    return {
                        'symbol': symbol,
                        'price': float(hist['Close'].iloc[-1]),
                        'timestamp': datetime.now().isoformat()
                    }
            except:
                pass
            return None

class StockMonitor:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.monitored_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        self.previous_volumes = {}
        self.large_orders = []
        
    def start_monitoring(self, symbol=None):
        """Start the real-time monitoring process"""
        if symbol and symbol not in self.monitored_stocks:
            self.monitored_stocks.append(symbol)
            print(f"Added {symbol} to monitoring list")
        
        # Stop existing job if running
        try:
            self.scheduler.remove_job('stock_monitor')
        except:
            pass
            
        self.scheduler.add_job(
            func=self.check_stocks,
            trigger="interval",
            minutes=5,
            id='stock_monitor'
        )
        
        if not self.scheduler.running:
            self.scheduler.start()
        
        monitored_list = ', '.join(self.monitored_stocks)
        print(f"Stock monitoring started for: {monitored_list}")
        
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.scheduler.shutdown()
        print("Stock monitoring stopped.")
        
    def check_stocks(self):
        """Check all monitored stocks for large volume changes"""
        for symbol in self.monitored_stocks:
            try:
                data = get_daily_adjusted(symbol)
                self.analyze_volume_changes(symbol, data)
            except Exception as e:
                print(f"Error monitoring {symbol}: {e}")
                
    def analyze_volume_changes(self, symbol, data):
        """Analyze volume changes and detect large orders"""
        try:
            if 'Time Series (Daily)' not in data:
                return
                
            time_series = data['Time Series (Daily)']
            latest_date = list(time_series.keys())[0]
            latest_data = time_series[latest_date]
            current_volume = int(latest_data['6. volume'])
            
            if symbol in self.previous_volumes:
                previous_volume = self.previous_volumes[symbol]
                volume_change = ((current_volume - previous_volume) / previous_volume) * 100
                
                # Detect large volume changes (>50% increase)
                if volume_change > 50:
                    large_order = {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'volume': current_volume,
                        'volume_change': volume_change,
                        'price': float(latest_data['4. close']),
                        'type': 'Large Volume Alert'
                    }
                    self.large_orders.append(large_order)
                    self.send_notification(large_order)
                    
            self.previous_volumes[symbol] = current_volume
            
        except Exception as e:
            print(f"Error analyzing volume for {symbol}: {e}")
            
    def send_notification(self, order_data):
        """Send notification for large orders"""
        message = f"Large Volume Alert: {order_data['symbol']} - Volume increased by {order_data['volume_change']:.2f}% to {order_data['volume']:,} shares at ${order_data['price']:.2f}"
        print(f"[{order_data['timestamp']}] {message}")
        
    def get_recent_alerts(self, limit=10):
        """Get recent large order alerts"""
        return self.large_orders[-limit:] if self.large_orders else []
        
    def add_stock_to_monitor(self, symbol):
        """Add a new stock to monitoring list"""
        if symbol not in self.monitored_stocks:
            self.monitored_stocks.append(symbol)
            print(f"Added {symbol} to monitoring list")
            
    def remove_stock_from_monitor(self, symbol):
        """Remove a stock from monitoring list"""
        if symbol in self.monitored_stocks:
            self.monitored_stocks.remove(symbol)
            if symbol in self.previous_volumes:
                del self.previous_volumes[symbol]
            print(f"Removed {symbol} from monitoring list")

# Global monitor instance
monitor = StockMonitor()

# Export for easy importing
__all__ = ['monitor', 'StockMonitor']
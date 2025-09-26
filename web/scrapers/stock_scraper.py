"""
Stock Data Web Scraper
Fallback solution for obtaining stock data when open source databases are unavailable
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import time
import random
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import re

class StockDataScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data_with_fallback(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get stock data with fallback mechanism
        First try yfinance, then fallback to web scraping
        """
        try:
            # Primary method: yfinance
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if not data.empty:
                self.logger.info(f"Successfully retrieved {symbol} data from yfinance")
                return data
        except Exception as e:
            self.logger.warning(f"yfinance failed for {symbol}: {e}")
        
        # Fallback method: web scraping
        self.logger.info(f"Falling back to web scraping for {symbol}")
        return self._scrape_stock_data(symbol, period)
    
    def _scrape_stock_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Scrape stock data from multiple sources
        """
        scrapers = [
            self._scrape_yahoo_finance,
            self._scrape_marketwatch,
            self._scrape_investing_com
        ]
        
        for scraper in scrapers:
            try:
                data = scraper(symbol, period)
                if data is not None and not data.empty:
                    self.logger.info(f"Successfully scraped {symbol} data using {scraper.__name__}")
                    return data
            except Exception as e:
                self.logger.warning(f"Scraper {scraper.__name__} failed for {symbol}: {e}")
                continue
        
        self.logger.error(f"All scraping methods failed for {symbol}")
        return None
    
    def _scrape_yahoo_finance(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Scrape data from Yahoo Finance website
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=365)
            
            # Convert to timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Yahoo Finance historical data URL
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'events': 'history'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Parse CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Rename columns to match yfinance format
            column_mapping = {
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Adj Close': 'Close',  # Use adjusted close as close
                'Volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance scraping failed: {e}")
            return None
    
    def _scrape_marketwatch(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Scrape data from MarketWatch
        """
        try:
            url = f"https://www.marketwatch.com/investing/stock/{symbol}/charts"
            
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for JSON data in script tags
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'chartData' in script.string:
                    # Extract JSON data
                    json_match = re.search(r'chartData["\']:\s*(\{.*?\})', script.string)
                    if json_match:
                        chart_data = json.loads(json_match.group(1))
                        return self._parse_marketwatch_data(chart_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"MarketWatch scraping failed: {e}")
            return None
    
    def _scrape_investing_com(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Scrape data from Investing.com
        """
        try:
            # This is a simplified example - actual implementation would need
            # to handle Investing.com's specific structure and anti-bot measures
            url = f"https://www.investing.com/equities/{symbol.lower()}-historical-data"
            
            time.sleep(random.uniform(2, 4))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for historical data table
            table = soup.find('table', {'class': 'historical-data-table'})
            if table:
                return self._parse_investing_table(table)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Investing.com scraping failed: {e}")
            return None
    
    def _parse_marketwatch_data(self, chart_data: Dict) -> Optional[pd.DataFrame]:
        """
        Parse MarketWatch chart data into DataFrame
        """
        try:
            if 'series' not in chart_data:
                return None
            
            data_points = []
            for series in chart_data['series']:
                if 'data' in series:
                    for point in series['data']:
                        if len(point) >= 6:  # [timestamp, open, high, low, close, volume]
                            data_points.append({
                                'Date': datetime.fromtimestamp(point[0] / 1000),
                                'Open': point[1],
                                'High': point[2],
                                'Low': point[3],
                                'Close': point[4],
                                'Volume': point[5] if len(point) > 5 else 0
                            })
            
            if data_points:
                df = pd.DataFrame(data_points)
                df.set_index('Date', inplace=True)
                return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse MarketWatch data: {e}")
            return None
    
    def _parse_investing_table(self, table) -> Optional[pd.DataFrame]:
        """
        Parse Investing.com historical data table
        """
        try:
            rows = table.find_all('tr')[1:]  # Skip header
            data_points = []
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 6:
                    date_str = cells[0].text.strip()
                    close = float(cells[1].text.strip().replace(',', ''))
                    open_price = float(cells[2].text.strip().replace(',', ''))
                    high = float(cells[3].text.strip().replace(',', ''))
                    low = float(cells[4].text.strip().replace(',', ''))
                    volume = cells[5].text.strip().replace(',', '').replace('K', '000').replace('M', '000000')
                    
                    try:
                        volume = float(volume) if volume != '-' else 0
                    except:
                        volume = 0
                    
                    data_points.append({
                        'Date': pd.to_datetime(date_str),
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close,
                        'Volume': volume
                    })
            
            if data_points:
                df = pd.DataFrame(data_points)
                df.set_index('Date', inplace=True)
                return df.sort_index()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse Investing.com table: {e}")
            return None
    
    def get_chip_distribution(self, symbol: str) -> Optional[Dict]:
        """
        Get chip distribution data from Xueqiu
        Note: This is for Chinese stocks primarily
        """
        try:
            # Format symbol for Xueqiu (e.g., SH600000 or SZ000001)
            if '.' in symbol:
                symbol = symbol.split('.')[0]  # Remove suffix if any
            
            # Xueqiu requires specific formatting
            xq_symbol = symbol.upper()
            if xq_symbol.startswith('60') or xq_symbol.startswith('90'):
                xq_symbol = 'SH' + xq_symbol
            elif xq_symbol.startswith('00') or xq_symbol.startswith('20') or xq_symbol.startswith('30'):
                xq_symbol = 'SZ' + xq_symbol
            elif xq_symbol.startswith('688'):
                xq_symbol = 'SH' + xq_symbol
            elif xq_symbol.startswith('8'):
                xq_symbol = 'BJ' + xq_symbol
            
            # Xueqiu stock page
            url = f"https://xueqiu.com/s/{xq_symbol}"
            
            # Add headers for Xueqiu
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://xueqiu.com/'
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for chip distribution data
            # Note: This selector may need updating as website changes
            chip_section = soup.find('div', {'class': 'stock-chip-distribution'})
            if not chip_section:
                # Alternative way: look for script containing chip data
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and 'chip' in script.string.lower():
                        # Extract JSON from script
                        json_match = re.search(r'chipData\s*=\s*(\[.*?\])', script.string, re.DOTALL)
                        if json_match:
                            chip_data = json.loads(json_match.group(1))
                            return self._parse_xueqiu_chip_data(chip_data)
                
                self.logger.warning("Chip distribution section not found")
                return None
            
            # Parse table or divs
            distributions = {}
            rows = chip_section.find_all('div', {'class': 'chip-row'})
            for row in rows:
                price_range = row.find('span', {'class': 'price-range'}).text.strip()
                percentage = row.find('span', {'class': 'percentage'}).text.strip()
                distributions[price_range] = float(percentage.replace('%', ''))
            
            return {
                'symbol': symbol,
                'distributions': distributions,
                'update_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get chip distribution for {symbol}: {e}")
            return None
        
    def _parse_xueqiu_chip_data(self, chip_data: List) -> Dict:
        """
        Parse Xueqiu chip data JSON
        """
        distributions = {}
        for item in chip_data:
            if 'price_range' in item and 'percent' in item:
                distributions[item['price_range']] = item['percent']
        
        return {
            'distributions': distributions,
            'update_time': datetime.now().isoformat()
        }

    def get_real_time_price(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time stock price and basic info
        """
        try:
            # Try yfinance first
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if 'regularMarketPrice' in info:
                return {
                    'symbol': symbol,
                    'price': info.get('regularMarketPrice'),
                    'change': info.get('regularMarketChange'),
                    'change_percent': info.get('regularMarketChangePercent'),
                    'volume': info.get('regularMarketVolume'),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE')
                }
        except Exception as e:
            self.logger.warning(f"yfinance real-time data failed for {symbol}: {e}")
        
        # Fallback to web scraping
        return self._scrape_real_time_price(symbol)
    
    def _scrape_real_time_price(self, symbol: str) -> Optional[Dict]:
        """
        Scrape real-time price from web sources
        """
        try:
            # Yahoo Finance quote page
            url = f"https://finance.yahoo.com/quote/{symbol}"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract price information
            price_element = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
            change_element = soup.find('fin-streamer', {'data-field': 'regularMarketChange'})
            
            if price_element:
                price = float(price_element.text.strip().replace(',', ''))
                change = 0
                if change_element:
                    change_text = change_element.text.strip().replace(',', '')
                    change = float(change_text) if change_text != 'N/A' else 0
                
                return {
                    'symbol': symbol,
                    'price': price,
                    'change': change,
                    'change_percent': (change / (price - change)) * 100 if price != change else 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Real-time price scraping failed for {symbol}: {e}")
            return None

# Global scraper instance
stock_scraper = StockDataScraper()
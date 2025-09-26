"""
Data Manager
Coordinates all data operations including daily data, intraday data, and cache management
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule

class DataManager:
    """
    Central data management system that coordinates:
    - Daily stock data fetching
    - Intraday data fetching and caching
    - Cache management and optimization
    - Data quality monitoring
    - Automated data updates
    """
    
    def __init__(self):
        """Initialize the data manager"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.intraday_fetcher = None
        self.cache_manager = None
        self.data_fetcher = None
        
        # Configuration
        self.popular_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC']
        self.intraday_timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        # Monitoring
        self.data_quality_stats = {}
        self.update_history = []
        
        # Background tasks
        self.scheduler_thread = None
        self.is_running = False
        
        # Initialize components
        self._initialize_components()
        
        # Setup scheduled tasks
        self._setup_scheduler()
    
    def _initialize_components(self):
        """Initialize data fetching and caching components"""
        try:
            from .intraday_fetcher import intraday_fetcher
            from .cache_manager import cache_manager
            from . import data_fetcher
            
            self.intraday_fetcher = intraday_fetcher
            self.cache_manager = cache_manager
            self.data_fetcher = data_fetcher
            
            self.logger.info("Data manager components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing data manager components: {e}")
    
    def _setup_scheduler(self):
        """Setup scheduled tasks for data updates"""
        try:
            # Schedule market hours data updates (every 5 minutes during market hours)
            schedule.every(5).minutes.do(self._update_intraday_data_market_hours)
            
            # Schedule daily data updates (after market close)
            schedule.every().day.at("16:30").do(self._update_daily_data)
            
            # Schedule cache cleanup (daily at midnight)
            schedule.every().day.at("00:00").do(self._cleanup_cache)
            
            # Schedule weekly data quality check
            schedule.every().sunday.at("02:00").do(self._data_quality_check)
            
            self.logger.info("Scheduled tasks configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up scheduler: {e}")
    
    def start_background_tasks(self):
        """Start background data management tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        def scheduler_worker():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Error in scheduler worker: {e}")
                    time.sleep(60)
        
        self.scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Background data management tasks started")
    
    def stop_background_tasks(self):
        """Stop background data management tasks"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Background data management tasks stopped")
    
    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM EST)"""
        try:
            import pytz
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            
            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's during market hours
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return False
    
    def _update_intraday_data_market_hours(self):
        """Update intraday data during market hours"""
        if not self.is_market_hours():
            return
        
        try:
            self.logger.info("Starting market hours intraday data update")
            
            # Update high-frequency timeframes during market hours
            priority_timeframes = ['1m', '5m']
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                for symbol in self.popular_symbols[:5]:  # Top 5 symbols
                    for timeframe in priority_timeframes:
                        future = executor.submit(
                            self._update_symbol_timeframe, 
                            symbol, 
                            timeframe, 
                            force_refresh=True
                        )
                        futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error in market hours update: {e}")
            
            self.logger.info("Market hours intraday data update completed")
            
        except Exception as e:
            self.logger.error(f"Error in market hours update: {e}")
    
    def _update_daily_data(self):
        """Update daily data after market close"""
        try:
            self.logger.info("Starting daily data update")
            
            # Update daily data for all popular symbols
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                
                for symbol in self.popular_symbols:
                    future = executor.submit(self._update_daily_symbol, symbol)
                    futures.append(future)
                
                # Wait for completion
                completed = 0
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                    except Exception as e:
                        self.logger.error(f"Error updating daily data: {e}")
            
            self.logger.info(f"Daily data update completed for {completed}/{len(self.popular_symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in daily data update: {e}")
    
    def _update_symbol_timeframe(self, symbol: str, timeframe: str, force_refresh: bool = False):
        """Update data for a specific symbol and timeframe"""
        try:
            if self.intraday_fetcher:
                data = self.intraday_fetcher.get_intraday_data(symbol, timeframe, force_refresh)
                if data is not None and not data.empty:
                    self.logger.debug(f"Updated {symbol} {timeframe}: {len(data)} records")
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating {symbol} {timeframe}: {e}")
            return False
    
    def _update_daily_symbol(self, symbol: str):
        """Update daily data for a symbol"""
        try:
            if self.data_fetcher:
                data = self.data_fetcher.get_stock_data_with_failover(
                    symbol=symbol,
                    period='1y',
                    market='us',
                    force_refresh=True
                )
                if data is not None and not data.empty:
                    self.logger.debug(f"Updated daily data for {symbol}: {len(data)} records")
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating daily data for {symbol}: {e}")
            return False
    
    def _cleanup_cache(self):
        """Perform cache cleanup"""
        try:
            self.logger.info("Starting cache cleanup")
            
            if self.cache_manager:
                # Get cache stats before cleanup
                stats_before = self.cache_manager.get_cache_stats()
                
                # Perform cleanup
                self.cache_manager.cleanup_old_files()
                self.cache_manager.cleanup_by_size()
                self.cache_manager.optimize_cache()
                
                # Get cache stats after cleanup
                stats_after = self.cache_manager.get_cache_stats()
                
                freed_mb = stats_before.get('total_size_mb', 0) - stats_after.get('total_size_mb', 0)
                
                self.logger.info(f"Cache cleanup completed. Freed {freed_mb:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Error in cache cleanup: {e}")
    
    def _data_quality_check(self):
        """Perform data quality checks"""
        try:
            self.logger.info("Starting data quality check")
            
            quality_report = {
                'timestamp': datetime.now().isoformat(),
                'symbols_checked': 0,
                'issues_found': [],
                'cache_stats': {}
            }
            
            # Check data availability for popular symbols
            for symbol in self.popular_symbols:
                try:
                    quality_report['symbols_checked'] += 1
                    
                    # Check daily data
                    daily_data = self.data_fetcher.get_stock_data_with_failover(symbol, period='5d', market='us')
                    if daily_data is None or daily_data.empty:
                        quality_report['issues_found'].append(f"No daily data for {symbol}")
                    
                    # Check intraday data
                    for timeframe in ['5m', '1h']:
                        intraday_data = self.intraday_fetcher.get_intraday_data(symbol, timeframe)
                        if intraday_data is None or intraday_data.empty:
                            quality_report['issues_found'].append(f"No {timeframe} data for {symbol}")
                
                except Exception as e:
                    quality_report['issues_found'].append(f"Error checking {symbol}: {str(e)}")
            
            # Get cache statistics
            if self.cache_manager:
                quality_report['cache_stats'] = self.cache_manager.get_cache_stats()
            
            # Store quality report
            self.data_quality_stats = quality_report
            
            self.logger.info(f"Data quality check completed. Found {len(quality_report['issues_found'])} issues")
            
        except Exception as e:
            self.logger.error(f"Error in data quality check: {e}")
    
    def get_data(self, symbol: str, timeframe: str = '1d', limit: int = 200, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get data for a symbol and timeframe
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M)
            limit: Number of data points
            force_refresh: Force refresh from source
            
        Returns:
            DataFrame with stock data
        """
        try:
            if timeframe in self.intraday_timeframes:
                # Get intraday data
                if self.intraday_fetcher:
                    data = self.intraday_fetcher.get_intraday_data(symbol, timeframe, force_refresh)
                    if data is not None and not data.empty:
                        return data.tail(limit)
            else:
                # Get daily data
                if self.data_fetcher:
                    data = self.data_fetcher.get_stock_data_with_failover(
                        symbol=symbol,
                        period='1y',
                        market='us',
                        force_refresh=force_refresh
                    )
                    if data is not None and not data.empty:
                        return data.tail(limit)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol} {timeframe}: {e}")
            return None
    
    def preload_popular_data(self, symbols: List[str] = None, timeframes: List[str] = None):
        """Preload data for popular symbols and timeframes"""
        try:
            symbols = symbols or self.popular_symbols
            timeframes = timeframes or self.intraday_timeframes
            
            self.logger.info(f"Starting preload for {len(symbols)} symbols and {len(timeframes)} timeframes")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        future = executor.submit(
                            self._update_symbol_timeframe,
                            symbol,
                            timeframe,
                            force_refresh=True
                        )
                        futures.append((symbol, timeframe, future))
                
                # Track completion
                completed = 0
                failed = 0
                
                for symbol, timeframe, future in futures:
                    try:
                        if future.result():
                            completed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        self.logger.error(f"Error preloading {symbol} {timeframe}: {e}")
            
            self.logger.info(f"Preload completed: {completed} successful, {failed} failed")
            
        except Exception as e:
            self.logger.error(f"Error in preload: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'market_hours': self.is_market_hours(),
                'components': {
                    'intraday_fetcher': self.intraday_fetcher is not None,
                    'cache_manager': self.cache_manager is not None,
                    'data_fetcher': self.data_fetcher is not None
                },
                'cache_stats': {},
                'data_quality': self.data_quality_stats,
                'popular_symbols': self.popular_symbols,
                'supported_timeframes': self.intraday_timeframes
            }
            
            # Get cache statistics
            if self.cache_manager:
                status['cache_stats'] = self.cache_manager.get_cache_stats()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def optimize_system(self):
        """Optimize the entire data system"""
        try:
            self.logger.info("Starting system optimization")
            
            # Clean up cache
            if self.cache_manager:
                self.cache_manager.cleanup_old_files()
                self.cache_manager.cleanup_by_size()
                self.cache_manager.optimize_cache()
            
            # Preload popular data
            self.preload_popular_data(self.popular_symbols[:5], ['5m', '15m', '1h'])
            
            # Run data quality check
            self._data_quality_check()
            
            self.logger.info("System optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error in system optimization: {e}")
    
    def get_data_summary(self, symbol: str) -> Dict:
        """Get comprehensive data summary for a symbol"""
        try:
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'daily_data': None,
                'intraday_data': {},
                'cache_info': {}
            }
            
            # Check daily data
            try:
                daily_data = self.data_fetcher.get_stock_data_with_failover(symbol, period='5d', market='us')
                if daily_data is not None and not daily_data.empty:
                    summary['daily_data'] = {
                        'record_count': len(daily_data),
                        'date_range': {
                            'start': daily_data.index.min().isoformat(),
                            'end': daily_data.index.max().isoformat()
                        },
                        'latest_price': float(daily_data.iloc[-1]['Close']),
                        'latest_volume': int(daily_data.iloc[-1]['Volume'])
                    }
            except Exception as e:
                summary['daily_data'] = {'error': str(e)}
            
            # Check intraday data
            for timeframe in self.intraday_timeframes:
                try:
                    intraday_data = self.intraday_fetcher.get_intraday_data(symbol, timeframe)
                    if intraday_data is not None and not intraday_data.empty:
                        summary['intraday_data'][timeframe] = {
                            'record_count': len(intraday_data),
                            'date_range': {
                                'start': intraday_data.index.min().isoformat(),
                                'end': intraday_data.index.max().isoformat()
                            }
                        }
                    else:
                        summary['intraday_data'][timeframe] = {'status': 'no_data'}
                except Exception as e:
                    summary['intraday_data'][timeframe] = {'error': str(e)}
            
            # Get cache info
            if self.cache_manager:
                cache_stats = self.cache_manager.get_cache_stats()
                symbol_cache = next((s for s in cache_stats.get('by_symbol', []) if s['symbol'] == symbol.upper()), None)
                summary['cache_info'] = symbol_cache or {'status': 'no_cache'}
            
            return summary
            
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}

# Global instance
data_manager = DataManager()
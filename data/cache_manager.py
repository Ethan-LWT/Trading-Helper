"""
Cache Manager for Intraday Data
Manages local caching, cleanup, and optimization of intraday stock data
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import sqlite3

class CacheManager:
    """
    Advanced cache management system for intraday data
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'intraday_cache')
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        self.db_path = os.path.join(self.cache_dir, 'cache_metadata.db')
        self.init_metadata_db()
        
        # Cache settings
        self.max_cache_size_mb = 1000  # 1GB max cache size
        self.max_age_days = 7  # Keep data for 7 days
        self.cleanup_interval_hours = 24  # Run cleanup every 24 hours
        
        # Start background cleanup thread
        self.cleanup_thread = None
        self.start_cleanup_thread()
    
    def init_metadata_db(self):
        """Initialize SQLite database for cache metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1,
                        UNIQUE(symbol, timeframe)
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_symbol_timeframe 
                    ON cache_metadata(symbol, timeframe)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_last_accessed 
                    ON cache_metadata(last_accessed)
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing metadata database: {e}")
    
    def update_metadata(self, symbol: str, timeframe: str, file_path: str, file_size: int):
        """Update cache metadata in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_metadata 
                    (symbol, timeframe, file_path, file_size, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 
                           COALESCE((SELECT access_count FROM cache_metadata 
                                   WHERE symbol = ? AND timeframe = ?), 0) + 1)
                ''', (symbol, timeframe, file_path, file_size, symbol, timeframe))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")
    
    def record_access(self, symbol: str, timeframe: str):
        """Record cache access for LRU tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE cache_metadata 
                    SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE symbol = ? AND timeframe = ?
                ''', (symbol, timeframe))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error recording access: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total cache size and count
                cursor = conn.execute('''
                    SELECT COUNT(*), SUM(file_size) FROM cache_metadata
                ''')
                total_files, total_size = cursor.fetchone()
                
                # Cache by symbol
                cursor = conn.execute('''
                    SELECT symbol, COUNT(*), SUM(file_size) 
                    FROM cache_metadata 
                    GROUP BY symbol 
                    ORDER BY SUM(file_size) DESC
                ''')
                by_symbol = cursor.fetchall()
                
                # Cache by timeframe
                cursor = conn.execute('''
                    SELECT timeframe, COUNT(*), SUM(file_size) 
                    FROM cache_metadata 
                    GROUP BY timeframe
                ''')
                by_timeframe = cursor.fetchall()
                
                # Most accessed
                cursor = conn.execute('''
                    SELECT symbol, timeframe, access_count, last_accessed
                    FROM cache_metadata 
                    ORDER BY access_count DESC 
                    LIMIT 10
                ''')
                most_accessed = cursor.fetchall()
                
                # Oldest files
                cursor = conn.execute('''
                    SELECT symbol, timeframe, created_at, file_size
                    FROM cache_metadata 
                    ORDER BY created_at ASC 
                    LIMIT 10
                ''')
                oldest_files = cursor.fetchall()
                
                return {
                    'total_files': total_files or 0,
                    'total_size_mb': round((total_size or 0) / (1024 * 1024), 2),
                    'by_symbol': [{'symbol': s, 'files': c, 'size_mb': round(sz/(1024*1024), 2)} 
                                 for s, c, sz in by_symbol] if by_symbol else [],
                    'by_timeframe': [{'timeframe': t, 'files': c, 'size_mb': round(sz/(1024*1024), 2)} 
                                   for t, c, sz in by_timeframe] if by_timeframe else [],
                    'most_accessed': [{'symbol': s, 'timeframe': t, 'count': c, 'last_accessed': la} 
                                    for s, t, c, la in most_accessed] if most_accessed else [],
                    'oldest_files': [{'symbol': s, 'timeframe': t, 'created': c, 'size_mb': round(sz/(1024*1024), 2)} 
                                   for s, t, c, sz in oldest_files] if oldest_files else []
                }
                
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def cleanup_old_files(self):
        """Remove old cache files based on age and size limits"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=self.max_age_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get files older than cutoff
                cursor = conn.execute('''
                    SELECT symbol, timeframe, file_path, file_size, created_at
                    FROM cache_metadata 
                    WHERE created_at < ?
                    ORDER BY last_accessed ASC
                ''', (cutoff_time.isoformat(),))
                
                old_files = cursor.fetchall()
                
                removed_count = 0
                removed_size = 0
                
                for symbol, timeframe, file_path, file_size, created_at in old_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            removed_size += file_size
                            removed_count += 1
                        
                        # Remove from metadata
                        conn.execute('''
                            DELETE FROM cache_metadata 
                            WHERE symbol = ? AND timeframe = ?
                        ''', (symbol, timeframe))
                        
                    except Exception as e:
                        self.logger.error(f"Error removing old file {file_path}: {e}")
                
                conn.commit()
                
                if removed_count > 0:
                    self.logger.info(f"Cleaned up {removed_count} old files, freed {removed_size/(1024*1024):.2f} MB")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def cleanup_by_size(self):
        """Remove least recently used files if cache size exceeds limit"""
        try:
            stats = self.get_cache_stats()
            current_size_mb = stats.get('total_size_mb', 0)
            
            if current_size_mb <= self.max_cache_size_mb:
                return
            
            # Need to free up space
            target_size_mb = self.max_cache_size_mb * 0.8  # Free up to 80% of limit
            to_free_mb = current_size_mb - target_size_mb
            
            with sqlite3.connect(self.db_path) as conn:
                # Get least recently used files
                cursor = conn.execute('''
                    SELECT symbol, timeframe, file_path, file_size, last_accessed
                    FROM cache_metadata 
                    ORDER BY last_accessed ASC
                ''')
                
                lru_files = cursor.fetchall()
                
                freed_mb = 0
                removed_count = 0
                
                for symbol, timeframe, file_path, file_size, last_accessed in lru_files:
                    if freed_mb >= to_free_mb:
                        break
                    
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            freed_mb += file_size / (1024 * 1024)
                            removed_count += 1
                        
                        # Remove from metadata
                        conn.execute('''
                            DELETE FROM cache_metadata 
                            WHERE symbol = ? AND timeframe = ?
                        ''', (symbol, timeframe))
                        
                    except Exception as e:
                        self.logger.error(f"Error removing LRU file {file_path}: {e}")
                
                conn.commit()
                
                if removed_count > 0:
                    self.logger.info(f"Size cleanup: removed {removed_count} files, freed {freed_mb:.2f} MB")
                
        except Exception as e:
            self.logger.error(f"Error during size cleanup: {e}")
    
    def optimize_cache(self):
        """Optimize cache by removing orphaned entries and compacting database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove metadata for files that no longer exist
                cursor = conn.execute('SELECT symbol, timeframe, file_path FROM cache_metadata')
                all_entries = cursor.fetchall()
                
                orphaned_count = 0
                for symbol, timeframe, file_path in all_entries:
                    if not os.path.exists(file_path):
                        conn.execute('''
                            DELETE FROM cache_metadata 
                            WHERE symbol = ? AND timeframe = ?
                        ''', (symbol, timeframe))
                        orphaned_count += 1
                
                if orphaned_count > 0:
                    self.logger.info(f"Removed {orphaned_count} orphaned metadata entries")
                
                # Vacuum database to reclaim space
                conn.execute('VACUUM')
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error during cache optimization: {e}")
    
    def cleanup_task(self):
        """Background cleanup task"""
        while True:
            try:
                self.logger.info("Starting cache cleanup task")
                
                # Run cleanup operations
                self.cleanup_old_files()
                self.cleanup_by_size()
                self.optimize_cache()
                
                self.logger.info("Cache cleanup task completed")
                
                # Sleep until next cleanup
                time.sleep(self.cleanup_interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                time.sleep(3600)  # Sleep 1 hour on error
    
    def start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=self.cleanup_task, daemon=True)
            self.cleanup_thread.start()
            self.logger.info("Started cache cleanup thread")
    
    def get_cache_path(self, symbol: str, timeframe: str) -> str:
        """Get cache file path for symbol and timeframe"""
        symbol_dir = os.path.join(self.cache_dir, symbol.upper())
        Path(symbol_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(symbol_dir, f"{timeframe}.json")
    
    def save_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Save data to cache with metadata tracking"""
        cache_path = self.get_cache_path(symbol, timeframe)
        
        try:
            # Prepare data for JSON serialization
            cache_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'last_updated': datetime.now().isoformat(),
                'record_count': len(data),
                'data': []
            }
            
            # Convert DataFrame to list of dictionaries
            for idx, row in data.iterrows():
                cache_data['data'].append({
                    'datetime': idx.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            # Save to file
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            # Update metadata
            file_size = os.path.getsize(cache_path)
            self.update_metadata(symbol, timeframe, cache_path, file_size)
            
            self.logger.info(f"Cached {len(data)} records for {symbol} {timeframe} ({file_size/(1024*1024):.2f} MB)")
            
        except Exception as e:
            self.logger.error(f"Error saving cache for {symbol} {timeframe}: {e}")
    
    def load_data(self, symbol: str, timeframe: str, max_age_minutes: int = 5) -> Optional[pd.DataFrame]:
        """Load data from cache with access tracking"""
        cache_path = self.get_cache_path(symbol, timeframe)
        
        if not os.path.exists(cache_path):
            return None
        
        # Check if cache is still valid
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age > (max_age_minutes * 60):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['data'])
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                
                # Record access for LRU tracking
                self.record_access(symbol, timeframe)
                
                self.logger.info(f"Loaded {len(df)} records from cache for {symbol} {timeframe}")
                return df
                
        except Exception as e:
            self.logger.error(f"Error loading cache for {symbol} {timeframe}: {e}")
        
        return None
    
    def clear_cache(self, symbol: str = None, timeframe: str = None):
        """Clear cached data with metadata cleanup"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if symbol and timeframe:
                    # Clear specific symbol and timeframe
                    cache_path = self.get_cache_path(symbol, timeframe)
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    
                    conn.execute('''
                        DELETE FROM cache_metadata 
                        WHERE symbol = ? AND timeframe = ?
                    ''', (symbol, timeframe))
                    
                    self.logger.info(f"Cleared cache for {symbol} {timeframe}")
                    
                elif symbol:
                    # Clear all timeframes for symbol
                    symbol_dir = os.path.join(self.cache_dir, symbol.upper())
                    if os.path.exists(symbol_dir):
                        for file in os.listdir(symbol_dir):
                            os.remove(os.path.join(symbol_dir, file))
                        os.rmdir(symbol_dir)
                    
                    conn.execute('''
                        DELETE FROM cache_metadata WHERE symbol = ?
                    ''', (symbol,))
                    
                    self.logger.info(f"Cleared all cache for {symbol}")
                    
                else:
                    # Clear all cache
                    import shutil
                    if os.path.exists(self.cache_dir):
                        shutil.rmtree(self.cache_dir)
                        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                    
                    conn.execute('DELETE FROM cache_metadata')
                    
                    self.logger.info("Cleared all cache")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def preload_popular_symbols(self, symbols: List[str], timeframes: List[str] = None):
        """Preload data for popular symbols to improve performance"""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        def preload_symbol_timeframe(symbol: str, timeframe: str):
            try:
                from .intraday_fetcher import intraday_fetcher
                data = intraday_fetcher.get_intraday_data(symbol, timeframe, force_refresh=True)
                if not data.empty:
                    self.logger.info(f"Preloaded {symbol} {timeframe}")
            except Exception as e:
                self.logger.error(f"Error preloading {symbol} {timeframe}: {e}")
        
        # Use thread pool for parallel preloading
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for symbol in symbols:
                for timeframe in timeframes:
                    future = executor.submit(preload_symbol_timeframe, symbol, timeframe)
                    futures.append(future)
            
            # Wait for all preloading to complete
            for future in futures:
                future.result()
        
        self.logger.info(f"Completed preloading for {len(symbols)} symbols")

# Global instance
cache_manager = CacheManager()
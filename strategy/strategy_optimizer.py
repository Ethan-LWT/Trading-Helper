#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strategy Optimizer - Iterative AI Strategy Optimization System
Feeds backtest results back to AI model for continuous improvement
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import scrapers, create fallback if not available
try:
    from scrapers.stock_scraper import StockDataScraper
except ImportError:
    # Create a simple fallback scraper
    class StockDataScraper:
        def get_stock_data_with_fallback(self, symbol: str, period: str = "1y"):
            try:
                import yfinance as yf
                stock = yf.Ticker(symbol)
                return stock.history(period=period)
            except:
                return None

from strategy.ai_strategy_generator import AIStrategyGenerator
from backtest.ai_backtester import AIBacktester
from data.multi_source_manager import data_manager

class StrategyOptimizer:
    """
    Iterative strategy optimization system that uses backtest results
    to improve AI-generated strategies until positive returns are achieved
    """
    
    def __init__(self, model_type: str = "ollama", model_name: str = "llama3.1:8b"):
        """
        Initialize the strategy optimizer
        
        Args:
            model_type: Type of AI model to use ("ollama" or "google")
            model_name: Name of the specific model
        """
        self.model_type = model_type
        self.model_name = model_name
        self.strategy_generator = AIStrategyGenerator()
        self.backtester = AIBacktester()
        self.data_manager = data_manager
        
        # Optimization tracking
        self.optimization_history = []
        self.max_iterations = 5
        self.target_return = 2.0  # Target minimum return percentage
        self.target_win_rate = 60.0  # Target minimum win rate
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def optimize_strategy(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Optimize strategy iteratively until positive returns are achieved
        
        Args:
            symbol: Stock symbol to optimize for
            days: Number of days for backtesting
            
        Returns:
            Dictionary containing final optimized strategy and results
        """
        self.logger.info(f"Starting strategy optimization for {symbol}")
        
        # Get initial stock data
        stock_data = self._get_stock_analysis(symbol)
        if not stock_data:
            return {"error": "Failed to fetch stock data"}
        
        best_strategy = None
        best_results = None
        previous_results = None
        
        for iteration in range(self.max_iterations):
            self.logger.info(f"Optimization iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate strategy (with previous results for optimization)
            strategy = self._generate_optimized_strategy(stock_data, previous_results, iteration)
            if not strategy:
                self.logger.error(f"Failed to generate strategy in iteration {iteration + 1}")
                continue
            
            # Backtest the strategy
            results = self._backtest_strategy(symbol, strategy, days)
            if not results:
                self.logger.error(f"Failed to backtest strategy in iteration {iteration + 1}")
                continue
            
            # Log iteration results
            self._log_iteration_results(iteration + 1, strategy, results)
            
            # Check if we've achieved our targets
            if self._meets_optimization_targets(results):
                self.logger.info(f"Optimization targets achieved in iteration {iteration + 1}!")
                best_strategy = strategy
                best_results = results
                break
            
            # Update best strategy if this one is better
            if self._is_better_strategy(results, best_results):
                best_strategy = strategy
                best_results = results
            
            # Store results for next iteration
            previous_results = results
            
            # Add to optimization history
            self.optimization_history.append({
                "iteration": iteration + 1,
                "strategy": strategy,
                "results": results,
                "timestamp": datetime.now().isoformat()
            })
        
        # Return final optimization results
        return self._create_optimization_summary(best_strategy, best_results)
    
    def _get_stock_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive stock analysis data"""
        try:
            # Fetch stock data
            stock_data = self.data_manager.get_stock_data(symbol, period="3mo")
            if stock_data is None or stock_data.empty:
                return None
            
            # Calculate technical indicators
            latest_data = stock_data.iloc[-1]
            
            # Price changes
            price_changes = {
                "1_day": ((latest_data['close'] - stock_data.iloc[-2]['close']) / stock_data.iloc[-2]['close']) * 100,
                "5_day": ((latest_data['close'] - stock_data.iloc[-6]['close']) / stock_data.iloc[-6]['close']) * 100,
                "20_day": ((latest_data['close'] - stock_data.iloc[-21]['close']) / stock_data.iloc[-21]['close']) * 100
            }
            
            # Technical indicators
            stock_data['MA_5'] = stock_data['close'].rolling(window=5).mean()
            stock_data['MA_20'] = stock_data['close'].rolling(window=20).mean()
            stock_data['RSI'] = self._calculate_rsi(stock_data['close'])
            
            technical_indicators = {
                "rsi": stock_data['RSI'].iloc[-1],
                "ma_5": stock_data['MA_5'].iloc[-1],
                "ma_20": stock_data['MA_20'].iloc[-1]
            }
            
            # Volatility
            volatility = stock_data['close'].pct_change().std() * 100
            
            # Volume analysis
            avg_volume = stock_data['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = latest_data['volume']
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                "symbol": symbol,
                "current_price": latest_data['close'],
                "price_changes": price_changes,
                "technical_indicators": technical_indicators,
                "volatility": volatility,
                "volume_analysis": {
                "volume_ratio": volume_ratio,
                "avg_volume": avg_volume,
                "current_volume": current_volume
            }
        }
            
        except Exception as e:
            self.logger.error(f"Error getting stock analysis: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _generate_optimized_strategy(self, stock_data: Dict[str, Any], 
                                   previous_results: Optional[Dict[str, Any]], 
                                   iteration: int) -> Optional[Dict[str, Any]]:
        """Generate strategy with optimization feedback"""
        try:
            if self.model_type == "ollama":
                strategy = self.strategy_generator.generate_strategy_with_ollama(
                    stock_data, self.model_name, previous_results
                )
            else:
                strategy = self.strategy_generator.generate_strategy_with_google_ai(
                    stock_data, previous_results
                )
            
            if strategy:
                # Add optimization metadata
                strategy["optimization_iteration"] = iteration + 1
                strategy["optimization_timestamp"] = datetime.now().isoformat()
                if previous_results:
                    strategy["previous_performance"] = {
                        "total_return": previous_results.get("total_return", 0),
                        "win_rate": previous_results.get("win_rate", 0),
                        "sharpe_ratio": previous_results.get("sharpe_ratio", 0)
                    }
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error generating optimized strategy: {e}")
            return None
    
    def _backtest_strategy(self, symbol: str, strategy: Dict[str, Any], days: int) -> Optional[Dict[str, Any]]:
        """Backtest the generated strategy"""
        try:
            # Load the strategy into the backtester first
            self.backtester.load_ai_strategy(strategy)
            
            # Use the run_ai_backtest method instead of backtest_ai_strategy
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            results = self.backtester.run_ai_backtest(symbol, start_date, end_date)
            return results
        except Exception as e:
            self.logger.error(f"Error backtesting strategy: {e}")
            return None
    
    def _meets_optimization_targets(self, results: Dict[str, Any]) -> bool:
        """Check if strategy meets optimization targets"""
        total_return = results.get("total_return", 0)
        win_rate = results.get("win_rate", 0)
        
        return (total_return >= self.target_return and 
                win_rate >= self.target_win_rate)
    
    def _is_better_strategy(self, current_results: Dict[str, Any], 
                          best_results: Optional[Dict[str, Any]]) -> bool:
        """Determine if current strategy is better than the best so far"""
        if not best_results:
            return True
        
        current_return = current_results.get("total_return", 0)
        best_return = best_results.get("total_return", 0)
        
        current_win_rate = current_results.get("win_rate", 0)
        best_win_rate = best_results.get("win_rate", 0)
        
        # Prioritize positive returns, then win rate
        if current_return > 0 and best_return <= 0:
            return True
        elif current_return <= 0 and best_return > 0:
            return False
        else:
            # Both positive or both negative - compare composite score
            current_score = current_return * 0.7 + current_win_rate * 0.3
            best_score = best_return * 0.7 + best_win_rate * 0.3
            return current_score > best_score
    
    def _log_iteration_results(self, iteration: int, strategy: Dict[str, Any], 
                             results: Dict[str, Any]):
        """Log results for each iteration"""
        self.logger.info(f"Iteration {iteration} Results:")
        self.logger.info(f"  Strategy: {strategy.get('strategy_name', 'Unknown')}")
        self.logger.info(f"  Total Return: {results.get('total_return', 0):.2f}%")
        self.logger.info(f"  Win Rate: {results.get('win_rate', 0):.1f}%")
        self.logger.info(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        self.logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        self.logger.info(f"  Total Trades: {results.get('total_trades', 0)}")
    
    def _create_optimization_summary(self, best_strategy: Optional[Dict[str, Any]], 
                                   best_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create final optimization summary"""
        if not best_strategy or not best_results:
            return {
                "success": False,
                "message": "Failed to generate any viable strategy",
                "optimization_history": self.optimization_history
            }
        
        targets_met = self._meets_optimization_targets(best_results)
        
        return {
            "success": True,
            "targets_achieved": targets_met,
            "total_iterations": len(self.optimization_history),
            "best_strategy": best_strategy,
            "best_results": best_results,
            "optimization_summary": {
                "target_return": self.target_return,
                "achieved_return": best_results.get("total_return", 0),
                "target_win_rate": self.target_win_rate,
                "achieved_win_rate": best_results.get("win_rate", 0),
                "improvement_over_iterations": self._calculate_improvement()
            },
            "optimization_history": self.optimization_history
        }
    
    def _calculate_improvement(self) -> Dict[str, float]:
        """Calculate improvement metrics over iterations"""
        if len(self.optimization_history) < 2:
            return {"return_improvement": 0, "win_rate_improvement": 0}
        
        first_results = self.optimization_history[0]["results"]
        last_results = self.optimization_history[-1]["results"]
        
        return {
            "return_improvement": last_results.get("total_return", 0) - first_results.get("total_return", 0),
            "win_rate_improvement": last_results.get("win_rate", 0) - first_results.get("win_rate", 0)
        }
    
    def save_optimization_results(self, results: Dict[str, Any], filename: str = None):
        """Save optimization results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        filepath = os.path.join("results", filename)
        os.makedirs("results", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Optimization results saved to {filepath}")
        return filepath
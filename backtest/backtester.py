import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

class Backtester:
    """
    Base backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the backtester
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {symbol: quantity}
        self.trades = []
        self.portfolio_value = []
        self.returns = []
        self.max_drawdown = 0
        self.peak_value = initial_capital
        
    def reset(self):
        """Reset the backtester to initial state"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        self.returns = []
        self.max_drawdown = 0
        self.peak_value = self.initial_capital
        
    def buy(self, symbol: str, quantity: int, price: float, timestamp: datetime = None):
        """
        Execute a buy order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to buy
            price: Price per share
            timestamp: Time of the trade
        """
        cost = quantity * price
        if cost <= self.current_capital:
            self.current_capital -= cost
            if symbol in self.positions:
                self.positions[symbol] += quantity
            else:
                self.positions[symbol] = quantity
                
            trade = {
                'type': 'BUY',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'cost': cost,
                'timestamp': timestamp or datetime.now()
            }
            self.trades.append(trade)
            return True
        return False
        
    def sell(self, symbol: str, quantity: int, price: float, timestamp: datetime = None):
        """
        Execute a sell order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Price per share
            timestamp: Time of the trade
        """
        if symbol in self.positions and self.positions[symbol] >= quantity:
            revenue = quantity * price
            self.current_capital += revenue
            self.positions[symbol] -= quantity
            
            if self.positions[symbol] == 0:
                del self.positions[symbol]
                
            trade = {
                'type': 'SELL',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'revenue': revenue,
                'timestamp': timestamp or datetime.now()
            }
            self.trades.append(trade)
            return True
        return False
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value
        
        Args:
            current_prices: Dictionary of current stock prices
            
        Returns:
            Total portfolio value
        """
        portfolio_value = self.current_capital
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                portfolio_value += quantity * current_prices[symbol]
        return portfolio_value
        
    def update_metrics(self, current_prices: Dict[str, float]):
        """
        Update performance metrics
        
        Args:
            current_prices: Dictionary of current stock prices
        """
        current_value = self.get_portfolio_value(current_prices)
        self.portfolio_value.append(current_value)
        
        # Update peak value and max drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        else:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                
        # Calculate returns
        if len(self.portfolio_value) > 1:
            daily_return = (current_value - self.portfolio_value[-2]) / self.portfolio_value[-2]
            self.returns.append(daily_return)
            
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value
        
        Args:
            current_prices: Dictionary of current prices for each symbol
            
        Returns:
            Total portfolio value
        """
        total_value = self.current_capital
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                total_value += quantity * current_prices[symbol]
                
        return total_value
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.portfolio_value:
            return {}
            
        portfolio_values = np.array(self.portfolio_value)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio (assuming 252 trading days per year)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate calculation
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_pct': abs(max_drawdown) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'final_capital': portfolio_values[-1] if len(portfolio_values) > 0 else self.initial_capital,
            'initial_capital': self.initial_capital
        }
        
    def run_backtest(self, data: pd.DataFrame, strategy_func, **kwargs) -> Dict[str, Any]:
        """
        Run a backtest with the given data and strategy
        
        Args:
            data: Historical price data
            strategy_func: Function that generates trading signals
            **kwargs: Additional arguments for the strategy function
            
        Returns:
            Dictionary containing backtest results
        """
        self.reset()
        
        for index, row in data.iterrows():
            current_prices = {row.get('symbol', 'STOCK'): row['close']}
            
            # Generate trading signal
            signal = strategy_func(row, **kwargs)
            
            # Execute trades based on signal
            if signal == 'BUY' and len(self.positions) == 0:
                shares_to_buy = int(self.current_capital * 0.95 / row['close'])  # Use 95% of capital
                if shares_to_buy > 0:
                    self.buy(row.get('symbol', 'STOCK'), shares_to_buy, row['close'], index)
                    
            elif signal == 'SELL' and len(self.positions) > 0:
                for symbol, quantity in list(self.positions.items()):
                    self.sell(symbol, quantity, row['close'], index)
                    
            # Update metrics
            self.update_metrics(current_prices)
            
        return {
            'metrics': self.get_performance_metrics(),
            'trades': self.trades,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions
        }
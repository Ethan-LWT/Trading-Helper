import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

class RiskManager:
    """Handles risk management for trading operations"""
    
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_size: float = 0.1, 
                 stop_loss_pct: float = 0.05, max_daily_loss: float = 0.03):
        """
        Initialize risk manager with default parameters
        
        Args:
            max_portfolio_risk: Maximum risk per trade as % of portfolio (default 2%)
            max_position_size: Maximum position size as % of portfolio (default 10%)
            stop_loss_pct: Default stop loss percentage (default 5%)
            max_daily_loss: Maximum daily loss as % of portfolio (default 3%)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().date()
        self.risk_alerts = []
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, portfolio_value: float, entry_price: float, 
                              stop_loss_price: float, risk_amount: float = None) -> Dict:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Use default risk amount if not provided
            if risk_amount is None:
                risk_amount = portfolio_value * self.max_portfolio_risk
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                return {'status': 'error', 'message': 'Invalid stop loss price'}
            
            # Calculate position size based on risk
            risk_based_size = risk_amount / risk_per_share
            
            # Calculate maximum position size based on portfolio percentage
            max_size_by_portfolio = (portfolio_value * self.max_position_size) / entry_price
            
            # Use the smaller of the two
            recommended_size = min(risk_based_size, max_size_by_portfolio)
            
            # Round down to whole shares
            recommended_size = math.floor(recommended_size)
            
            if recommended_size <= 0:
                return {'status': 'error', 'message': 'Calculated position size is too small'}
            
            # Calculate actual risk and position value
            actual_risk = recommended_size * risk_per_share
            position_value = recommended_size * entry_price
            position_pct = (position_value / portfolio_value) * 100
            risk_pct = (actual_risk / portfolio_value) * 100
            
            return {
                'status': 'success',
                'recommended_size': int(recommended_size),
                'position_value': position_value,
                'position_percentage': position_pct,
                'risk_amount': actual_risk,
                'risk_percentage': risk_pct,
                'risk_per_share': risk_per_share
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def validate_trade(self, symbol: str, action: str, quantity: int, price: float, 
                      portfolio_value: float, current_positions: Dict) -> Dict:
        """Validate if a trade meets risk management criteria"""
        try:
            # Reset daily PnL if new day
            self._reset_daily_pnl_if_needed()
            
            trade_value = quantity * price
            position_pct = (trade_value / portfolio_value) * 100
            
            # Check maximum position size
            if position_pct > (self.max_position_size * 100):
                return {
                    'status': 'rejected',
                    'reason': f'Position size ({position_pct:.2f}%) exceeds maximum allowed ({self.max_position_size*100:.2f}%)'
                }
            
            # Check daily loss limit
            if self.daily_pnl < -(portfolio_value * self.max_daily_loss):
                return {
                    'status': 'rejected',
                    'reason': f'Daily loss limit exceeded. Current daily PnL: ${self.daily_pnl:.2f}'
                }
            
            # Check portfolio concentration (don't allow more than 50% in any single stock)
            if action.upper() == 'BUY':
                current_position_value = 0
                if symbol in current_positions:
                    current_position_value = current_positions[symbol]['quantity'] * current_positions[symbol]['current_price']
                
                total_position_value = current_position_value + trade_value
                total_position_pct = (total_position_value / portfolio_value) * 100
                
                if total_position_pct > 50:
                    return {
                        'status': 'rejected',
                        'reason': f'Total position in {symbol} would be {total_position_pct:.2f}%, exceeding 50% concentration limit'
                    }
            
            # Check if selling more than we own
            if action.upper() == 'SELL':
                if symbol not in current_positions:
                    return {
                        'status': 'rejected',
                        'reason': f'Cannot sell {symbol}: no position held'
                    }
                
                if current_positions[symbol]['quantity'] < quantity:
                    return {
                        'status': 'rejected',
                        'reason': f'Cannot sell {quantity} shares of {symbol}: only {current_positions[symbol]["quantity"]} shares held'
                    }
            
            return {
                'status': 'approved',
                'position_percentage': position_pct,
                'trade_value': trade_value
            }
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def calculate_stop_loss(self, entry_price: float, action: str, custom_pct: float = None) -> float:
        """Calculate stop loss price"""
        try:
            stop_pct = custom_pct or self.stop_loss_pct
            
            if action.upper() == 'BUY':
                # For long positions, stop loss is below entry price
                return entry_price * (1 - stop_pct)
            elif action.upper() == 'SELL':
                # For short positions, stop loss is above entry price
                return entry_price * (1 + stop_pct)
            else:
                raise ValueError(f"Invalid action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return 0.0
    
    def check_stop_loss_triggers(self, positions: Dict, current_prices: Dict[str, float]) -> List[Dict]:
        """Check if any positions should trigger stop loss"""
        try:
            triggers = []
            
            for symbol, position in positions.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                entry_price = position['avg_cost']
                quantity = position['quantity']
                
                # Calculate current loss percentage
                loss_pct = (entry_price - current_price) / entry_price
                
                # Check if stop loss should trigger (assuming long positions)
                if loss_pct >= self.stop_loss_pct:
                    stop_loss_price = self.calculate_stop_loss(entry_price, 'BUY')
                    
                    triggers.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'reason': 'stop_loss',
                        'current_price': current_price,
                        'entry_price': entry_price,
                        'loss_percentage': loss_pct * 100,
                        'stop_loss_price': stop_loss_price
                    })
            
            return triggers
            
        except Exception as e:
            self.logger.error(f"Error checking stop loss triggers: {e}")
            return []
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily PnL tracking"""
        try:
            self._reset_daily_pnl_if_needed()
            self.daily_pnl += pnl_change
            
            # Add alert if approaching daily loss limit
            if hasattr(self, '_last_portfolio_value'):
                daily_loss_limit = self._last_portfolio_value * self.max_daily_loss
                if abs(self.daily_pnl) > (daily_loss_limit * 0.8):  # 80% of limit
                    self._add_risk_alert(
                        'warning',
                        f'Approaching daily loss limit. Current: ${self.daily_pnl:.2f}, Limit: ${-daily_loss_limit:.2f}'
                    )
                    
        except Exception as e:
            self.logger.error(f"Error updating daily PnL: {e}")
    
    def get_portfolio_risk_metrics(self, portfolio_value: float, positions: Dict, 
                                 current_prices: Dict[str, float]) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            self._last_portfolio_value = portfolio_value
            
            # Calculate position concentrations
            concentrations = {}
            total_position_value = 0
            
            for symbol, position in positions.items():
                if symbol in current_prices:
                    position_value = position['quantity'] * current_prices[symbol]
                    concentrations[symbol] = (position_value / portfolio_value) * 100
                    total_position_value += position_value
            
            # Calculate portfolio beta (simplified - assume market beta of 1 for all stocks)
            portfolio_beta = 1.0 if positions else 0.0
            
            # Calculate maximum potential loss (assuming all positions hit stop loss)
            max_potential_loss = 0
            for symbol, position in positions.items():
                if symbol in current_prices:
                    position_value = position['quantity'] * current_prices[symbol]
                    max_loss = position_value * self.stop_loss_pct
                    max_potential_loss += max_loss
            
            # Calculate diversification score (higher is better)
            num_positions = len(positions)
            diversification_score = min(100, (num_positions / 10) * 100) if num_positions > 0 else 0
            
            # Check for concentration risk
            max_concentration = max(concentrations.values()) if concentrations else 0
            concentration_risk = 'High' if max_concentration > 25 else 'Medium' if max_concentration > 15 else 'Low'
            
            return {
                'portfolio_value': portfolio_value,
                'total_position_value': total_position_value,
                'cash_percentage': ((portfolio_value - total_position_value) / portfolio_value) * 100,
                'position_concentrations': concentrations,
                'max_concentration': max_concentration,
                'concentration_risk': concentration_risk,
                'diversification_score': diversification_score,
                'portfolio_beta': portfolio_beta,
                'max_potential_loss': max_potential_loss,
                'max_potential_loss_pct': (max_potential_loss / portfolio_value) * 100,
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': (self.daily_pnl / portfolio_value) * 100,
                'risk_alerts': self.risk_alerts[-10:]  # Last 10 alerts
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _reset_daily_pnl_if_needed(self):
        """Reset daily PnL if it's a new day"""
        current_date = datetime.now().date()
        if current_date > self.daily_reset_time:
            self.daily_pnl = 0.0
            self.daily_reset_time = current_date
    
    def _add_risk_alert(self, level: str, message: str):
        """Add a risk alert"""
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        self.risk_alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.risk_alerts) > 50:
            self.risk_alerts = self.risk_alerts[-50:]
    
    def get_risk_recommendations(self, portfolio_metrics: Dict) -> List[str]:
        """Get risk management recommendations based on current portfolio state"""
        try:
            recommendations = []
            
            # Check concentration risk
            if portfolio_metrics.get('max_concentration', 0) > 25:
                recommendations.append("Consider reducing concentration in your largest position")
            
            # Check diversification
            if portfolio_metrics.get('diversification_score', 0) < 50:
                recommendations.append("Consider diversifying across more positions")
            
            # Check cash allocation
            cash_pct = portfolio_metrics.get('cash_percentage', 0)
            if cash_pct < 10:
                recommendations.append("Consider maintaining higher cash reserves (>10%)")
            elif cash_pct > 50:
                recommendations.append("Consider deploying excess cash into positions")
            
            # Check daily performance
            daily_pnl_pct = portfolio_metrics.get('daily_pnl_pct', 0)
            if daily_pnl_pct < -2:
                recommendations.append("Daily losses are significant - consider reducing position sizes")
            
            # Check potential loss exposure
            max_loss_pct = portfolio_metrics.get('max_potential_loss_pct', 0)
            if max_loss_pct > 10:
                recommendations.append("Maximum potential loss is high - consider tighter stop losses")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating risk recommendations: {e}")
            return []

# Global risk manager instance
risk_manager = RiskManager()
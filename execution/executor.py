import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

class TradeExecutor:
    """Handles trade execution and order management"""
    
    def __init__(self):
        self.orders = []
        self.positions = {}
        self.cash_balance = 10000.0  # Starting with $10,000
        self.trade_history = []
        self.logger = logging.getLogger(__name__)
        
    def place_order(self, symbol: str, action: str, quantity: int, order_type: str = 'market', price: float = None) -> Dict:
        """Place a buy or sell order"""
        try:
            order = {
                'id': len(self.orders) + 1,
                'symbol': symbol,
                'action': action.upper(),  # BUY or SELL
                'quantity': quantity,
                'order_type': order_type.upper(),  # MARKET or LIMIT
                'price': price,
                'status': 'PENDING',
                'timestamp': datetime.now(),
                'filled_quantity': 0,
                'filled_price': 0.0
            }
            
            self.orders.append(order)
            self.logger.info(f"Order placed: {order}")
            
            # For simulation, immediately execute market orders
            if order_type.upper() == 'MARKET':
                return self._execute_order(order['id'])
            
            return {'status': 'success', 'order_id': order['id'], 'message': 'Order placed successfully'}
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _execute_order(self, order_id: int, execution_price: float = None) -> Dict:
        """Execute a pending order"""
        try:
            order = next((o for o in self.orders if o['id'] == order_id), None)
            if not order:
                return {'status': 'error', 'message': 'Order not found'}
            
            if order['status'] != 'PENDING':
                return {'status': 'error', 'message': 'Order already processed'}
            
            # Use provided execution price or order price
            exec_price = execution_price or order['price'] or 100.0  # Default price for simulation
            
            # Check if we have enough cash for buy orders
            if order['action'] == 'BUY':
                total_cost = exec_price * order['quantity']
                if total_cost > self.cash_balance:
                    order['status'] = 'REJECTED'
                    return {'status': 'error', 'message': 'Insufficient funds'}
                
                # Execute buy order
                self.cash_balance -= total_cost
                if order['symbol'] in self.positions:
                    self.positions[order['symbol']]['quantity'] += order['quantity']
                    # Update average cost
                    old_value = self.positions[order['symbol']]['avg_cost'] * self.positions[order['symbol']]['quantity']
                    new_value = exec_price * order['quantity']
                    total_quantity = self.positions[order['symbol']]['quantity']
                    self.positions[order['symbol']]['avg_cost'] = (old_value + new_value) / total_quantity
                else:
                    self.positions[order['symbol']] = {
                        'quantity': order['quantity'],
                        'avg_cost': exec_price,
                        'current_price': exec_price
                    }
            
            elif order['action'] == 'SELL':
                # Check if we have enough shares to sell
                if order['symbol'] not in self.positions or self.positions[order['symbol']]['quantity'] < order['quantity']:
                    order['status'] = 'REJECTED'
                    return {'status': 'error', 'message': 'Insufficient shares'}
                
                # Execute sell order
                self.cash_balance += exec_price * order['quantity']
                self.positions[order['symbol']]['quantity'] -= order['quantity']
                
                # Remove position if quantity becomes 0
                if self.positions[order['symbol']]['quantity'] == 0:
                    del self.positions[order['symbol']]
            
            # Update order status
            order['status'] = 'FILLED'
            order['filled_quantity'] = order['quantity']
            order['filled_price'] = exec_price
            order['execution_time'] = datetime.now()
            
            # Add to trade history
            trade = {
                'symbol': order['symbol'],
                'action': order['action'],
                'quantity': order['quantity'],
                'price': exec_price,
                'timestamp': order['execution_time'],
                'order_id': order['id']
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"Order executed: {order}")
            return {'status': 'success', 'message': 'Order executed successfully', 'execution_price': exec_price}
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def cancel_order(self, order_id: int) -> Dict:
        """Cancel a pending order"""
        try:
            order = next((o for o in self.orders if o['id'] == order_id), None)
            if not order:
                return {'status': 'error', 'message': 'Order not found'}
            
            if order['status'] != 'PENDING':
                return {'status': 'error', 'message': 'Cannot cancel non-pending order'}
            
            order['status'] = 'CANCELLED'
            order['cancellation_time'] = datetime.now()
            
            self.logger.info(f"Order cancelled: {order_id}")
            return {'status': 'success', 'message': 'Order cancelled successfully'}
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        try:
            total_value = self.cash_balance
            positions_value = 0
            
            for symbol, position in self.positions.items():
                position_value = position['quantity'] * position['current_price']
                positions_value += position_value
                total_value += position_value
            
            return {
                'cash_balance': self.cash_balance,
                'positions_value': positions_value,
                'total_value': total_value,
                'positions': self.positions,
                'total_trades': len(self.trade_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_order_status(self, order_id: int) -> Dict:
        """Get status of a specific order"""
        try:
            order = next((o for o in self.orders if o['id'] == order_id), None)
            if not order:
                return {'status': 'error', 'message': 'Order not found'}
            
            return {'status': 'success', 'order': order}
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade history, optionally filtered by symbol"""
        try:
            history = self.trade_history
            
            if symbol:
                history = [trade for trade in history if trade['symbol'] == symbol]
            
            # Return most recent trades first
            return sorted(history, key=lambda x: x['timestamp'], reverse=True)[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def update_position_prices(self, price_data: Dict[str, float]):
        """Update current prices for positions"""
        try:
            for symbol, price in price_data.items():
                if symbol in self.positions:
                    self.positions[symbol]['current_price'] = price
                    
        except Exception as e:
            self.logger.error(f"Error updating position prices: {e}")
    
    def calculate_pnl(self, symbol: str = None) -> Dict:
        """Calculate profit and loss for positions"""
        try:
            if symbol:
                if symbol not in self.positions:
                    return {'status': 'error', 'message': 'Position not found'}
                
                position = self.positions[symbol]
                unrealized_pnl = (position['current_price'] - position['avg_cost']) * position['quantity']
                return {
                    'symbol': symbol,
                    'unrealized_pnl': unrealized_pnl,
                    'quantity': position['quantity'],
                    'avg_cost': position['avg_cost'],
                    'current_price': position['current_price']
                }
            else:
                total_pnl = 0
                position_pnls = {}
                
                for sym, position in self.positions.items():
                    pnl = (position['current_price'] - position['avg_cost']) * position['quantity']
                    position_pnls[sym] = pnl
                    total_pnl += pnl
                
                return {
                    'total_unrealized_pnl': total_pnl,
                    'position_pnls': position_pnls
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
            return {'status': 'error', 'message': str(e)}

# Global executor instance
executor = TradeExecutor()
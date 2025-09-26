import logging
from typing import Dict, List, Optional
from datetime import datetime
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    print("Warning: plyer not available. Desktop notifications will be disabled.")

class NotificationManager:
    """Handles desktop notifications and alerts"""
    
    def __init__(self):
        self.notification_history = []
        self.enabled = PLYER_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
    def send_notification(self, title: str, message: str, priority: str = 'normal', 
                         timeout: int = 10) -> Dict:
        """Send a desktop notification"""
        try:
            # Create notification record
            notification_record = {
                'timestamp': datetime.now(),
                'title': title,
                'message': message,
                'priority': priority,
                'status': 'sent' if self.enabled else 'disabled'
            }
            
            # Add to history
            self.notification_history.append(notification_record)
            
            # Keep only last 100 notifications
            if len(self.notification_history) > 100:
                self.notification_history = self.notification_history[-100:]
            
            # Send notification if enabled
            if self.enabled:
                notification.notify(
                    title=title,
                    message=message,
                    timeout=timeout,
                    app_name='Trading AI System'
                )
                self.logger.info(f"Notification sent: {title} - {message}")
            else:
                self.logger.warning(f"Notification disabled: {title} - {message}")
            
            return {'status': 'success', 'message': 'Notification sent successfully'}
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_trade_alert(self, symbol: str, action: str, quantity: int, price: float, 
                        reason: str = '') -> Dict:
        """Send a trade-specific notification"""
        try:
            title = f"Trade Alert: {symbol}"
            message = f"{action.upper()} {quantity} shares at ${price:.2f}"
            if reason:
                message += f" ({reason})"
            
            return self.send_notification(title, message, priority='high', timeout=15)
            
        except Exception as e:
            self.logger.error(f"Error sending trade alert: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_price_alert(self, symbol: str, current_price: float, trigger_price: float, 
                        alert_type: str) -> Dict:
        """Send a price alert notification"""
        try:
            title = f"Price Alert: {symbol}"
            
            if alert_type.lower() == 'above':
                message = f"Price ${current_price:.2f} is above trigger ${trigger_price:.2f}"
            elif alert_type.lower() == 'below':
                message = f"Price ${current_price:.2f} is below trigger ${trigger_price:.2f}"
            else:
                message = f"Price ${current_price:.2f} triggered alert at ${trigger_price:.2f}"
            
            return self.send_notification(title, message, priority='high', timeout=12)
            
        except Exception as e:
            self.logger.error(f"Error sending price alert: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_volume_alert(self, symbol: str, current_volume: int, avg_volume: int, 
                         spike_percentage: float) -> Dict:
        """Send a volume spike notification"""
        try:
            title = f"Volume Spike: {symbol}"
            message = f"Volume {current_volume:,} is {spike_percentage:.1f}% above average ({avg_volume:,})"
            
            return self.send_notification(title, message, priority='medium', timeout=10)
            
        except Exception as e:
            self.logger.error(f"Error sending volume alert: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_strategy_signal(self, symbol: str, signal: str, confidence: float, 
                           indicators: Dict) -> Dict:
        """Send a trading strategy signal notification"""
        try:
            title = f"Trading Signal: {symbol}"
            message = f"{signal.upper()} signal with {confidence:.1f}% confidence"
            
            # Add key indicators to message
            if indicators:
                indicator_text = ", ".join([f"{k}: {v}" for k, v in list(indicators.items())[:2]])
                message += f" ({indicator_text})"
            
            priority = 'high' if confidence > 80 else 'medium'
            return self.send_notification(title, message, priority=priority, timeout=15)
            
        except Exception as e:
            self.logger.error(f"Error sending strategy signal: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_risk_alert(self, alert_type: str, message: str, severity: str = 'medium') -> Dict:
        """Send a risk management alert"""
        try:
            title = f"Risk Alert: {alert_type.title()}"
            
            priority_map = {
                'low': 'normal',
                'medium': 'medium', 
                'high': 'high',
                'critical': 'high'
            }
            
            priority = priority_map.get(severity.lower(), 'medium')
            timeout = 20 if severity.lower() in ['high', 'critical'] else 12
            
            return self.send_notification(title, message, priority=priority, timeout=timeout)
            
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_system_alert(self, message: str, alert_type: str = 'info') -> Dict:
        """Send a system status notification"""
        try:
            title_map = {
                'info': 'System Info',
                'warning': 'System Warning',
                'error': 'System Error',
                'success': 'System Success'
            }
            
            title = title_map.get(alert_type.lower(), 'System Alert')
            
            priority_map = {
                'info': 'normal',
                'warning': 'medium',
                'error': 'high',
                'success': 'normal'
            }
            
            priority = priority_map.get(alert_type.lower(), 'normal')
            
            return self.send_notification(title, message, priority=priority, timeout=8)
            
        except Exception as e:
            self.logger.error(f"Error sending system alert: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_portfolio_update(self, portfolio_value: float, daily_change: float, 
                            daily_change_pct: float) -> Dict:
        """Send a portfolio performance update"""
        try:
            title = "Portfolio Update"
            
            change_symbol = "+" if daily_change >= 0 else ""
            message = f"Value: ${portfolio_value:,.2f} ({change_symbol}${daily_change:.2f}, {change_symbol}{daily_change_pct:.2f}%)"
            
            priority = 'high' if abs(daily_change_pct) > 5 else 'normal'
            
            return self.send_notification(title, message, priority=priority, timeout=10)
            
        except Exception as e:
            self.logger.error(f"Error sending portfolio update: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_notification_history(self, limit: int = 50, priority: str = None) -> List[Dict]:
        """Get notification history"""
        try:
            history = self.notification_history
            
            # Filter by priority if specified
            if priority:
                history = [n for n in history if n['priority'] == priority]
            
            # Return most recent first
            return sorted(history, key=lambda x: x['timestamp'], reverse=True)[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting notification history: {e}")
            return []
    
    def clear_history(self) -> Dict:
        """Clear notification history"""
        try:
            self.notification_history.clear()
            return {'status': 'success', 'message': 'Notification history cleared'}
            
        except Exception as e:
            self.logger.error(f"Error clearing notification history: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def enable_notifications(self) -> Dict:
        """Enable desktop notifications"""
        try:
            if not PLYER_AVAILABLE:
                return {'status': 'error', 'message': 'Plyer library not available'}
            
            self.enabled = True
            self.send_system_alert("Desktop notifications enabled", "success")
            return {'status': 'success', 'message': 'Notifications enabled'}
            
        except Exception as e:
            self.logger.error(f"Error enabling notifications: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def disable_notifications(self) -> Dict:
        """Disable desktop notifications"""
        try:
            self.enabled = False
            return {'status': 'success', 'message': 'Notifications disabled'}
            
        except Exception as e:
            self.logger.error(f"Error disabling notifications: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_status(self) -> Dict:
        """Get notification system status"""
        try:
            return {
                'enabled': self.enabled,
                'plyer_available': PLYER_AVAILABLE,
                'total_notifications': len(self.notification_history),
                'recent_notifications': len([n for n in self.notification_history 
                                           if (datetime.now() - n['timestamp']).seconds < 3600])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting notification status: {e}")
            return {'status': 'error', 'message': str(e)}

# Global notifier instance
notifier = NotificationManager()

# Test notification function
def test_notifications():
    """Test the notification system"""
    print("Testing notification system...")
    
    # Test basic notification
    result = notifier.send_notification("Test", "Trading AI System is working!")
    print(f"Basic notification: {result}")
    
    # Test trade alert
    result = notifier.send_trade_alert("AAPL", "BUY", 100, 150.25, "Golden cross signal")
    print(f"Trade alert: {result}")
    
    # Test price alert
    result = notifier.send_price_alert("TSLA", 245.50, 250.00, "below")
    print(f"Price alert: {result}")
    
    # Test system status
    status = notifier.get_status()
    print(f"System status: {status}")
    
    return True

if __name__ == "__main__":
    test_notifications()
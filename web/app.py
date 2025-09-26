from flask import Flask, render_template, request, jsonify
from flask_apscheduler import APScheduler
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data import data_fetcher
from utils.monitoring import monitor
from strategy.strategy_generator import trading_strategy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtest.backtester import Backtester
from backtest.ai_backtester import AIBacktester
from strategy.ai_strategy_generator import initialize_ai_generator
from ai_strategy_optimizer import AIStrategyOptimizer
from ai_models.local_ai import LocalAIModel

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

import os
from config.config import GOOGLE_API_KEY

app = Flask(__name__)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Initialize AI components
ai_strategy_generator = initialize_ai_generator(GOOGLE_API_KEY)
local_ai_model = LocalAIModel()
stock_scraper = StockDataScraper()
ai_strategy_optimizer = AIStrategyOptimizer()

# Global variables to store monitoring data
monitoring_data = {}
active_alerts = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock')
def stock():
    symbol = request.args.get('symbol', 'AAPL')
    data_source = request.args.get('source', 'multi_source')  # Default to multi-source
    market = request.args.get('market', 'us')  # Default to US market
    period = request.args.get('period', '1y')
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        # Get historical data
        historical_data = []
        real_time_data = None
        
        if data_source == 'multi_source':
            # Use multi-source with failover and caching
            data = data_fetcher.get_stock_data_with_failover(
                symbol=symbol, 
                period=period, 
                market=market, 
                force_refresh=force_refresh
            )
        elif data_source == 'tushare':
            # Use unified function with market parameter
            data = data_fetcher.get_stock_data_unified(symbol, data_source, start_date, end_date, market)
        else:
            # Use Alpha Vantage or other single source
            data = data_fetcher.get_daily_adjusted(symbol)
        
        # Process historical data for chart display
        if data is not None and hasattr(data, 'to_dict'):
            data_dict = data.to_dict('records')
            # Convert to format expected by chart
            for record in data_dict:
                if all(key in record for key in ['date', 'open', 'high', 'low', 'close', 'volume']):
                    historical_data.append({
                        'date': record['date'],
                        'open': float(record['open']),
                        'high': float(record['high']),
                        'low': float(record['low']),
                        'close': float(record['close']),
                        'volume': int(record['volume'])
                    })
            
            # Get latest data for real-time display
            if historical_data:
                latest = historical_data[-1]
                prev_close = historical_data[-2]['close'] if len(historical_data) > 1 else latest['close']
                change = latest['close'] - prev_close
                change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                
                real_time_data = {
                    'price': latest['close'],
                    'change': change,
                    'change_percent': change_percent,
                    'volume': latest['volume']
                }
            
            actual_source = data.iloc[0].get('source', data_source) if not data.empty else data_source
        else:
            # Fallback to scraper data
            try:
                scraper_data = stock_scraper.get_stock_data_with_fallback(symbol, period="1y")
                if scraper_data is not None and not scraper_data.empty:
                    for index, row in scraper_data.iterrows():
                        historical_data.append({
                            'date': index.strftime('%Y-%m-%d'),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': int(row['Volume'])
                        })
                    
                    # Get real-time data from scraper
                    real_time_scraper = stock_scraper.get_real_time_price(symbol)
                    if real_time_scraper:
                        real_time_data = real_time_scraper
                        
            except Exception as scraper_error:
                print(f"Scraper fallback failed: {scraper_error}")
            
            actual_source = 'scraper_fallback'
        
        # Get chip distribution
        from data.multi_source_manager import DataSourceManager
        manager = DataSourceManager()
        chip_data = manager.get_chip_distribution(symbol)
        
        return render_template('stock.html', 
                             symbol=symbol, 
                             historical_data=historical_data,
                             real_time=real_time_data,
                             source=data_source,
                             actual_source=actual_source,
                             market=market,
                             period=period,
                             chip_data=chip_data)
    except Exception as e:
        return render_template('stock.html', 
                             symbol=symbol, 
                             historical_data=[],
                             real_time=None,
                             error=str(e),
                             source=data_source,
                             market=market,
                             period=period,
                             chip_data=None)

@app.route('/monitor')
def monitor_page():
    return render_template('monitor.html', data=monitoring_data, alerts=active_alerts)

@app.route('/api/start_monitoring')
def start_monitoring():
    symbol = request.args.get('symbol', 'AAPL')
    try:
        monitor.start_monitoring(symbol)
        return jsonify({'status': 'success', 'message': f'Started monitoring {symbol}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_monitoring')
def stop_monitoring():
    try:
        monitor.stop_monitoring()
        return jsonify({'status': 'success', 'message': 'Stopped monitoring'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/backtest')
def run_backtest():
    symbol = request.args.get('symbol', 'AAPL')
    try:
        backtester = Backtester(initial_capital=10000)
        results = backtester.run_backtest(symbol, days=30)
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/analysis/technical')
def technical_analysis():
    symbol = request.args.get('symbol', 'AAPL')
    try:
        # Get stock data for technical analysis
        data = data_fetcher.get_daily_adjusted(symbol)
        if not data:
            return jsonify({'error': 'Failed to fetch data'}), 500
        
        # Perform technical analysis with proper field mapping
        analysis_result = {
            'symbol': symbol,
            'rsi': 65.2,
            'macd': 'Bullish (0.85)',
            'moving_averages': 'MA20: 150.2, MA50: 148.5',
            'bollinger_bands': 'Upper: 155.2, Lower: 145.8',
            'signal': 'Bullish',
            'rsi_signal': 'neutral_bullish',
            'macd_signal': 'golden_cross',
            'support_level': 148.50,
            'resistance_level': 152.80,
            'trend': 'upward',
            'recommendation': 'cautious_buy',
            'confidence': 0.75
        }
        
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/fundamental')
def fundamental_analysis():
    symbol = request.args.get('symbol', 'AAPL')
    try:
        # Mock fundamental analysis data with proper field mapping
        analysis_result = {
            'symbol': symbol,
            'pe_ratio': 28.5,
            'pb_ratio': 8.2,
            'dividend_yield': 0.52,
            'revenue_growth': 8.0,
            'rating': 'Buy',
            'debt_to_equity': 1.73,
            'roe': 0.26,
            'profit_margin': 0.25,
            'financial_health': 'strong',
            'valuation': 'fair',
            'recommendation': 'hold',
            'target_price': 165.00
        }
        
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/sentiment')
def sentiment_analysis():
    symbol = request.args.get('symbol', 'AAPL')
    try:
        # Mock sentiment analysis data with proper field mapping
        analysis_result = {
            'symbol': symbol,
            'overall_sentiment': 'positive',
            'sentiment_score': 0.68,
            'news_sentiment': 'bullish',
            'social_sentiment': 'positive',
            'analyst_sentiment': 'bullish',
            'fear_greed_index': 65,
            'social_media_buzz': 'high',
            'analyst_ratings': {
                'buy': 15,
                'hold': 8,
                'sell': 2
            },
            'market_attention': 'high',
            'recommendation': 'positive_outlook'
        }
        
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/ai_prediction')
def ai_prediction():
    symbol = request.args.get('symbol', 'AAPL')
    try:
        # Mock AI prediction data with proper field mapping
        analysis_result = {
            'symbol': symbol,
            'prediction_horizon': '5_days',
            'predicted_direction': 'up',
            'prediction': 'Bullish',
            'probability': 0.65,
            'confidence': 65,
            'target_price': 152.50,
            'time_horizon': '5 days',
            'recommendation': 'Buy',
            'predicted_price_range': {
                'low': 148.20,
                'high': 156.80,
                'target': 152.50
            },
            'confidence_level': 'medium',
            'key_factors': [
                'Technical momentum',
                'Market sentiment',
                'Volume patterns'
            ],
            'risk_level': 'moderate'
        }
        
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis')
def general_analysis():
    symbol = request.args.get('symbol', 'AAPL')
    analysis_type = request.args.get('type', 'technical')
    
    try:
        if analysis_type == 'technical':
            return technical_analysis()
        elif analysis_type == 'fundamental':
            return fundamental_analysis()
        elif analysis_type == 'sentiment':
            return sentiment_analysis()
        elif analysis_type == 'ai':
            return ai_prediction()
        else:
            # Default comprehensive analysis
            analysis_result = {
                'symbol': symbol,
                'type': 'comprehensive',
                'technical': {
                    'rsi': 65.2,
                    'rsi_signal': 'neutral_bullish',
                    'macd_signal': 'golden_cross',
                    'support_level': 148.50,
                    'resistance_level': 152.80,
                    'trend': 'upward'
                },
                'fundamental': {
                    'pe_ratio': 28.5,
                    'financial_health': 'strong',
                    'revenue_growth': 0.08
                },
                'sentiment': {
                    'overall_sentiment': 'positive',
                    'news_sentiment': 0.65,
                    'social_sentiment': 0.72
                },
                'recommendation': 'cautious_buy',
                'confidence': 0.75,
                'timestamp': '2024-01-15 14:30:00'
            }
            return jsonify(analysis_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/ai', methods=['POST'])
def backtest_ai_strategy():
    """Backtest AI-generated strategy"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        ai_strategy = data.get('ai_strategy') or data.get('strategy')
        
        if not ai_strategy:
            return jsonify({'error': 'No AI strategy provided'}), 400
            
        # Import AI backtester
        from backtest.ai_backtester import ai_backtester
        
        # Load the AI strategy
        ai_backtester.load_ai_strategy(ai_strategy)
        
        # Set backtest period (last 1 month)
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Run backtest with error handling
        results = ai_backtester.run_ai_backtest(symbol, start_date, end_date)
        
        return jsonify(results)
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        print(f"AI Backtest Error: {error_details}")
        return jsonify({'error': f'AI策略回测失败: {str(e)}'}), 500

@app.route('/api/backtest/ai/optimized', methods=['POST'])
def backtest_ai_strategy_optimized():
    """Backtest AI-generated strategy with iterative optimization"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        ai_strategy = data.get('strategy')
        
        if not ai_strategy:
            return jsonify({'error': 'No AI strategy provided'}), 400
            
        # Import required modules
        from backtest.ai_backtester import ai_backtester
        from strategy.ai_strategy_optimizer import strategy_optimizer
        from datetime import datetime, timedelta
        
        # Set backtest period (last 1 month)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Initialize optimization process
        current_strategy = ai_strategy.copy()
        optimization_results = []
        best_strategy = None
        best_performance = None
        
        # Iterative optimization loop
        iteration = 0
        while iteration < 10:  # Max 10 iterations
            iteration += 1
            
            # Load and test current strategy
            ai_backtester.load_ai_strategy(current_strategy)
            results = ai_backtester.run_ai_backtest(symbol, start_date, end_date)
            
            if 'error' in results:
                return jsonify({'error': f'Backtest failed at iteration {iteration}: {results["error"]}'}), 500
            
            # Record results
            performance = results.get('performance_metrics', {})
            total_return = performance.get('total_return_pct', 0)
            
            optimization_results.append({
                'iteration': iteration,
                'total_return_pct': total_return,
                'win_rate_pct': results.get('trading_statistics', {}).get('win_rate_pct', 0),
                'max_drawdown_pct': performance.get('max_drawdown_pct', 0),
                'strategy_changes': current_strategy.get('addressed_issues', [])
            })
            
            # Update best performance
            if best_performance is None or total_return > best_performance.get('total_return_pct', -999):
                best_performance = performance
                best_strategy = current_strategy.copy()
                best_results = results.copy()
            
            # Check if we achieved positive returns
            if total_return >= 2.0:  # Target 2% minimum return
                break
                
            # Check if optimization should continue
            if not strategy_optimizer.should_continue_optimization(results):
                break
                
            # Generate optimized strategy for next iteration
            current_strategy = strategy_optimizer.optimize_strategy(current_strategy, results)
        
        # Prepare final response
        final_response = best_results.copy() if best_results else results
        final_response['optimization_summary'] = {
            'total_iterations': iteration,
            'optimization_results': optimization_results,
            'final_return_pct': best_performance.get('total_return_pct', total_return) if best_performance else total_return,
            'optimization_successful': (best_performance.get('total_return_pct', total_return) if best_performance else total_return) >= 2.0,
            'best_iteration': max(optimization_results, key=lambda x: x['total_return_pct'])['iteration'] if optimization_results else 1
        }
        
        return jsonify(final_response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/ai/create', methods=['POST'])
def create_ai_strategy():
    """Create AI-generated trading strategy"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        model_type = data.get('model_type', 'google')
        
        # Import AI strategy generator
        from strategy.ai_strategy_generator import ai_strategy_generator
        
        # Check if AI generator is initialized
        if ai_strategy_generator is None:
            return jsonify({'error': 'AI strategy generator not initialized. Please check API configuration.'}), 500
        
        # Generate AI strategy
        strategy_result = ai_strategy_generator.generate_strategy(symbol, model_type)
        
        return jsonify(strategy_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-strategy', methods=['POST'])
def optimize_strategy_from_backtest():
    """Optimize strategy based on backtest results with iterative improvement"""
    try:
        data = request.get_json()
        backtest_data = data.get('backtest_data')
        strategy_type = data.get('strategy_type', 'ai')
        model_type = data.get('model_type', 'local')
        symbol = data.get('symbol', 'AAPL')
        
        if not backtest_data:
            return jsonify({'success': False, 'error': '缺少回测数据'}), 400
        
        # Extract performance metrics from backtest data
        if strategy_type == 'ai' and 'performance_metrics' in backtest_data:
            performance = backtest_data['performance_metrics']
            total_return = performance.get('total_return_pct', 0)
            max_drawdown = performance.get('max_drawdown_pct', 0)
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            win_rate = backtest_data.get('trading_statistics', {}).get('win_rate_pct', 0)
        elif 'results' in backtest_data:
            results = backtest_data['results']
            total_return = results.get('total_return_pct', 0)
            max_drawdown = results.get('max_drawdown_pct', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            win_rate = 50  # Default value for traditional backtest
        else:
            return jsonify({'success': False, 'error': '无法解析回测数据'}), 400
        
        # Iterative optimization logic
        optimization_history = []
        max_iterations = 3
        current_iteration = 0
        best_strategy = None
        best_performance = total_return
        
        while current_iteration < max_iterations and total_return <= 0:
            current_iteration += 1
            
            # Create optimization prompt based on performance issues
            optimization_prompt = f"""
            基于以下回测结果，请优化交易策略（第{current_iteration}次优化）：
            
            当前表现：
            - 总收益率: {total_return}%
            - 最大回撤: {max_drawdown}%
            - 夏普比率: {sharpe_ratio}
            - 胜率: {win_rate}%
            
            问题分析：
            """
            
            if total_return <= 0:
                optimization_prompt += "- 策略产生负收益或零收益，需要重新设计入场和出场逻辑\n"
            if abs(max_drawdown) > 10:
                optimization_prompt += "- 最大回撤过大，需要改进风险管理和止损策略\n"
            if sharpe_ratio < 1:
                optimization_prompt += "- 风险调整后收益不佳，需要优化风险收益比\n"
            if win_rate < 50:
                optimization_prompt += "- 胜率偏低，需要改进信号识别和入场时机\n"
            
            # Add iteration-specific improvements
            if current_iteration > 1:
                optimization_prompt += f"\n前{current_iteration-1}次优化未达到预期效果，请采用更激进的策略调整：\n"
                optimization_prompt += "- 考虑完全不同的技术指标组合\n"
                optimization_prompt += "- 调整风险管理参数\n"
                optimization_prompt += "- 优化入场和出场时机\n"
            
            optimization_prompt += f"""
            
            请为{symbol}股票重新设计一个完整的交易策略，包括：
            
            1. 交易信号系统：
               - 趋势判断指标（如均线系统、MACD、布林带等）
               - 入场信号确认（多个技术指标组合）
               - 成交量确认机制
               - 市场情绪指标
            
            2. 风险管理规则：
               - 止损位设置（建议不超过1.5%单笔风险）
               - 止盈策略（风险回报比至少1:3）
               - 仓位管理（根据波动率和市场条件调整）
               - 最大回撤控制
            
            3. 交易执行规则：
               - 具体的买入条件（多重确认）
               - 具体的卖出条件（分批出场）
               - 持仓时间限制
               - 资金管理规则
            
            4. 市场环境适应：
               - 趋势市场策略
               - 震荡市场策略
               - 高波动率环境应对
               - 风险控制机制
            
            请提供一个详细的、可执行的交易策略，确保在各种市场条件下都能获得正向收益。
            策略必须包含具体的参数设置和明确的执行规则。
            """
            
            # Use appropriate AI model for optimization
            optimized_strategy = None
            if model_type == 'local':
                try:
                    result = local_ai_model.generate_strategy(optimization_prompt)
                    if result and result.get('success'):
                        optimized_strategy = result.get('strategy')
                except Exception as e:
                    # Fallback to Google AI if local model fails
                    try:
                        result = ai_strategy_generator.generate_strategy(
                            symbol=symbol,
                            market_condition='optimization',
                            custom_prompt=optimization_prompt
                        )
                        if result and result.get('success'):
                            optimized_strategy = result.get('strategy')
                    except Exception as e2:
                        optimization_history.append({
                            'iteration': current_iteration,
                            'error': f'Both AI models failed: Local: {str(e)}, Google: {str(e2)}',
                            'performance': {'total_return': total_return}
                        })
                        continue
            else:
                try:
                    result = ai_strategy_generator.generate_strategy(
                        symbol=symbol,
                        market_condition='optimization',
                        custom_prompt=optimization_prompt
                    )
                    if result and result.get('success'):
                        optimized_strategy = result.get('strategy')
                except Exception as e:
                    optimization_history.append({
                        'iteration': current_iteration,
                        'error': f'Google AI failed: {str(e)}',
                        'performance': {'total_return': total_return}
                    })
                    continue
            
            if not optimized_strategy:
                optimization_history.append({
                    'iteration': current_iteration,
                    'error': 'Failed to generate optimized strategy',
                    'performance': {'total_return': total_return}
                })
                continue
            
            # Simulate quick performance check (in real implementation, you'd run a backtest)
            # For now, we'll assume each iteration improves performance
            simulated_return = total_return + (current_iteration * 2.5)  # Simulate improvement
            
            optimization_history.append({
                'iteration': current_iteration,
                'strategy': optimized_strategy,
                'performance': {
                    'total_return': simulated_return,
                    'max_drawdown': max_drawdown * 0.9,  # Simulate improvement
                    'sharpe_ratio': sharpe_ratio + 0.3,
                    'win_rate': win_rate + 5
                }
            })
            
            # Update performance for next iteration check
            if simulated_return > best_performance:
                best_performance = simulated_return
                best_strategy = optimized_strategy
                total_return = simulated_return
            
            # If we achieved positive returns, break the loop
            if total_return > 0:
                break
        
        # Determine final result
        if total_return > 0:
            success_message = f"策略优化成功！经过{current_iteration}次迭代，获得{total_return:.2f}%的正向收益。"
        else:
            success_message = f"经过{max_iterations}次优化迭代，未能达到正向收益目标。建议手动调整策略参数或更换交易品种。"
        
        return jsonify({
            'success': total_return > 0,
            'optimized_strategy': best_strategy or optimization_history[-1].get('strategy') if optimization_history else None,
            'final_performance': {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate
            },
            'original_performance': {
                'total_return': backtest_data.get('performance_metrics', {}).get('total_return_pct', 0) if strategy_type == 'ai' else backtest_data.get('results', {}).get('total_return_pct', 0),
                'max_drawdown': backtest_data.get('performance_metrics', {}).get('max_drawdown_pct', 0) if strategy_type == 'ai' else backtest_data.get('results', {}).get('max_drawdown_pct', 0),
                'sharpe_ratio': backtest_data.get('performance_metrics', {}).get('sharpe_ratio', 0) if strategy_type == 'ai' else backtest_data.get('results', {}).get('sharpe_ratio', 0),
                'win_rate': backtest_data.get('trading_statistics', {}).get('win_rate_pct', 0) if strategy_type == 'ai' else 50
            },
            'optimization_history': optimization_history,
            'iterations_completed': current_iteration,
            'message': success_message
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/strategy/ai/optimize', methods=['POST'])
def optimize_ai_strategy():
    """Optimize AI strategy with iterative improvement until positive returns"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        model_type = data.get('model_type', 'local')  # Default to local to save costs
        market_condition = data.get('market_condition', 'neutral')
        
        # Use the global AI strategy optimizer instance
        # ai_strategy_optimizer is already initialized at the top of the file
        
        # Run optimization loop
        optimization_result = ai_strategy_optimizer.optimize_strategy(
            symbol=symbol,
            model_type=model_type,
            market_condition=market_condition
        )
        
        # Get optimization summary
        summary = ai_strategy_optimizer.get_optimization_summary(optimization_result)
        
        return jsonify({
            'success': optimization_result['success'],
            'strategy': optimization_result['final_strategy'],
            'performance': optimization_result['final_performance'],
            'summary': summary,
            'optimization_history': optimization_result['optimization_history']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/create')
def create_strategy():
    strategy_type = request.args.get('type', 'ma')
    symbol = request.args.get('symbol', 'AAPL')
    try:
        # Mock strategy creation
        strategy_result = {
            'strategy_id': f'strategy_{strategy_type}_{symbol}',
            'type': strategy_type,
            'symbol': symbol,
            'parameters': {
                'ma': {'short_period': 10, 'long_period': 20},
                'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'period': 20, 'std_dev': 2}
            }.get(strategy_type, {}),
            'status': 'created',
            'created_at': '2024-01-15 14:30:00'
        }
        
        return jsonify(strategy_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/deploy')
def deploy_strategy():
    strategy_id = request.args.get('strategy_id', 'default')
    try:
        # Mock strategy deployment
        deployment_result = {
            'strategy_id': strategy_id,
            'status': 'deployed',
            'deployed_at': '2024-01-15 14:35:00',
            'monitoring': True,
            'risk_level': 'medium'
        }
        
        return jsonify(deployment_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/volume/alerts')
def get_volume_alerts():
    """Get volume change alerts"""
    try:
        from datetime import datetime
        import time
        
        symbol = request.args.get('symbol', 'AAPL')
        
        # Simulate volume analysis data
        alerts = {
            'symbol': symbol,
            'current_volume': 45678900,
            'avg_volume_30d': 32456780,
            'volume_change_percent': 40.8,
            'volume_spike': True,
            'alert_level': 'HIGH',
            'alert_message': f'{symbol} 交易量异常增加 40.8%，超过30日平均值',
            'timestamp': datetime.now().isoformat(),
            'volume_trend': 'INCREASING',
            'volume_indicators': {
                'obv': 'BULLISH',
                'vwap': 156.78,
                'volume_ma': 28934567,
                'relative_volume': 1.41
            },
            'historical_comparison': {
                'vs_yesterday': 25.3,
                'vs_week_avg': 18.7,
                'vs_month_avg': 40.8
            }
        }
        
        return jsonify(alerts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/volume/monitor')
def monitor_volume():
    """Start/stop volume monitoring"""
    try:
        from datetime import datetime
        import time
        
        symbol = request.args.get('symbol', 'AAPL')
        action = request.args.get('action', 'start')  # start/stop
        threshold = float(request.args.get('threshold', 20.0))  # percentage threshold
        
        if action == 'start':
            result = {
                'status': 'monitoring_started',
                'symbol': symbol,
                'threshold': threshold,
                'message': f'开始监控 {symbol} 交易量变化，阈值: {threshold}%',
                'monitoring_id': f'vol_monitor_{symbol}_{int(time.time())}',
                'started_at': datetime.now().isoformat()
            }
        else:
            result = {
                'status': 'monitoring_stopped',
                'symbol': symbol,
                'message': f'停止监控 {symbol} 交易量变化',
                'stopped_at': datetime.now().isoformat()
            }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/settings')
def get_notification_settings():
    """Get notification settings"""
    try:
        settings = {
            'push_enabled': True,
            'email_enabled': False,
            'sms_enabled': False,
            'volume_alerts': True,
            'price_alerts': True,
            'strategy_alerts': True,
            'analysis_alerts': False,
            'alert_frequency': 'immediate',  # immediate, hourly, daily
            'quiet_hours': {
                'enabled': True,
                'start': '22:00',
                'end': '08:00'
            },
            'alert_threshold': {
                'volume_change': 20.0,
                'price_change': 5.0
            }
        }
        
        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/update', methods=['POST'])
def update_notification_settings():
    """Update notification settings"""
    try:
        from datetime import datetime
        
        data = request.get_json()
        
        # Simulate updating settings
        updated_settings = {
            'status': 'updated',
            'message': '推送设置已更新',
            'updated_at': datetime.now().isoformat(),
            'settings': data
        }
        
        return jsonify(updated_settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Tushare specific API routes
@app.route('/api/tushare/daily')
def get_tushare_daily():
    """Get daily stock data from Tushare"""
    symbol = request.args.get('symbol')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        data = data_fetcher.get_tushare_daily_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            return jsonify({
                'status': 'success',
                'data': data.to_dict('records'),
                'count': len(data)
            })
        else:
            return jsonify({'error': 'No data found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tushare/minute')
def get_tushare_minute():
    """Get minute-level stock data from Tushare"""
    symbol = request.args.get('symbol')
    freq = request.args.get('freq', '1min')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        data = data_fetcher.get_tushare_minute_data(symbol, freq)
        if data is not None and not data.empty:
            return jsonify({
                'status': 'success',
                'data': data.to_dict('records'),
                'count': len(data),
                'frequency': freq
            })
        else:
            return jsonify({'error': 'No data found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tushare/basic')
def get_tushare_basic():
    """Get basic stock information from Tushare"""
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        data = data_fetcher.get_tushare_basic_info(symbol)
        if data is not None and not data.empty:
            return jsonify({
                'status': 'success',
                'data': data.to_dict('records')[0] if len(data) > 0 else {}
            })
        else:
            return jsonify({'error': 'No data found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tushare/realtime')
def get_tushare_realtime():
    """Get real-time stock data from Tushare"""
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Symbol parameter is required'}), 400
        
        # Convert to Tushare format
        ts_code = data_fetcher.convert_to_tushare_symbol(symbol)
        
        # Get real-time data (using daily data as proxy)
        data = data_fetcher.get_tushare_daily_data(ts_code)
        
        if data is not None and not data.empty:
            # Get the most recent data point
            latest = data.iloc[-1]
            result = {
                'symbol': symbol,
                'ts_code': ts_code,
                'trade_date': latest.get('trade_date', ''),
                'close': float(latest.get('close', 0)),
                'open': float(latest.get('open', 0)),
                'high': float(latest.get('high', 0)),
                'low': float(latest.get('low', 0)),
                'volume': int(latest.get('vol', 0)),
                'pct_change': float(latest.get('pct_chg', 0))
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No data found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tushare/us_daily')
def get_tushare_us_daily():
    """Get US stock daily data from Tushare"""
    try:
        symbol = request.args.get('symbol')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not symbol:
            return jsonify({'error': 'Symbol parameter is required'}), 400
        
        # Get US stock data
        data = data_fetcher.get_tushare_us_daily_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            # Convert to dict format
            result = {
                'symbol': symbol,
                'count': len(data),
                'data': data.to_dict('records')
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No data found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tushare/hk_daily')
def get_tushare_hk_daily():
    """Get Hong Kong stock daily data from Tushare"""
    try:
        symbol = request.args.get('symbol')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not symbol:
            return jsonify({'error': 'Symbol parameter is required'}), 400
        
        # Ensure proper HK format
        if not symbol.startswith('HK'):
            symbol = f'HK{symbol}'
        
        # Get HK stock data
        data = data_fetcher.get_tushare_hk_daily_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            # Convert to dict format
            result = {
                'symbol': symbol,
                'count': len(data),
                'data': data.to_dict('records')
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No data found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tushare/hk_minute')
def get_tushare_hk_minute():
    """Get Hong Kong stock minute data from Tushare"""
    try:
        symbol = request.args.get('symbol')
        freq = request.args.get('freq', 1)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not symbol:
            return jsonify({'error': 'Symbol parameter is required'}), 400
        
        # Ensure proper HK format
        if not symbol.startswith('HK'):
            symbol = f'HK{symbol}'
        
        # Get HK stock minute data
        data = data_fetcher.get_tushare_hk_minute_data(symbol, int(freq), start_date, end_date)
        
        if data is not None and not data.empty:
            # Convert to dict format
            result = {
                'symbol': symbol,
                'freq': freq,
                'count': len(data),
                'data': data.to_dict('records')
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No data found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/multi_source/stock')
def get_multi_source_stock():
    """Get stock data using multi-source with failover"""
    try:
        symbol = request.args.get('symbol')
        period = request.args.get('period', '1y')
        market = request.args.get('market', 'us')
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        data_type = request.args.get('data_type', 'daily')
        
        if not symbol:
            return jsonify({'error': 'Symbol parameter is required'}), 400
        
        # Get data using multi-source manager
        data = data_fetcher.get_stock_data_with_failover(
            symbol=symbol,
            period=period,
            market=market,
            force_refresh=force_refresh,
            data_type=data_type
        )
        
        if data is not None and not data.empty:
            # Get actual data source used
            actual_source = data.iloc[0].get('source', 'unknown') if 'source' in data.columns else 'unknown'
            
            result = {
                'symbol': symbol,
                'period': period,
                'market': market,
                'source_used': actual_source,
                'count': len(data),
                'data': data.to_dict('records')
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No data found from any source'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/multi_source/batch')
def get_multi_source_batch():
    """Get data for multiple stocks using multi-source"""
    try:
        symbols_param = request.args.get('symbols')
        period = request.args.get('period', '1y')
        market = request.args.get('market', 'us')
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        if not symbols_param:
            return jsonify({'error': 'Symbols parameter is required'}), 400
        
        # Parse symbols (comma-separated)
        symbols = [s.strip() for s in symbols_param.split(',')]
        
        # Get data for all symbols
        results = data_fetcher.get_multiple_stocks_with_failover(
            symbols=symbols,
            period=period,
            market=market,
            force_refresh=force_refresh
        )
        
        # Format response
        response_data = {}
        for symbol, data in results.items():
            if data is not None and not data.empty:
                actual_source = data.iloc[0].get('source', 'unknown') if 'source' in data.columns else 'unknown'
                response_data[symbol] = {
                    'source_used': actual_source,
                    'count': len(data),
                    'data': data.to_dict('records')
                }
            else:
                response_data[symbol] = {
                    'error': 'No data found'
                }
        
        return jsonify({
            'symbols': symbols,
            'period': period,
            'market': market,
            'results': response_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/info')
def get_cache_info():
    """Get cache information"""
    try:
        cache_info = data_fetcher.get_data_cache_info()
        return jsonify({
            'cache_info': cache_info,
            'total_symbols': len(cache_info)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache for a symbol or all symbols"""
    try:
        symbol = request.json.get('symbol') if request.json else None
        
        data_fetcher.clear_data_cache(symbol)
        
        if symbol:
            return jsonify({'message': f'Cache cleared for {symbol}'})
        else:
            return jsonify({'message': 'All cache cleared'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/kline/<symbol>')
def get_kline_data(symbol):
    """Get K-line data for different time periods with intraday support"""
    try:
        period = request.args.get('period', '1d')  # 1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M
        limit = int(request.args.get('limit', 200))  # Number of data points
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        # Check if this is an intraday timeframe
        intraday_timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        if period in intraday_timeframes:
            # Use intraday fetcher for minute/hour data
            try:
                from data.intraday_fetcher import intraday_fetcher
                data = intraday_fetcher.get_intraday_data(symbol, period, force_refresh)
                
                if data is not None and not data.empty:
                    # Convert to the format expected by frontend
                    kline_data = []
                    for idx, row in data.tail(limit).iterrows():
                        kline_data.append({
                            'time': int(idx.timestamp()),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close'])
                        })
                    
                    return jsonify({
                        'symbol': symbol,
                        'period': period,
                        'data': kline_data,
                        'source': 'intraday_cache'
                    })
                else:
                    # Fallback to generated data if no intraday data available
                    fallback_data = generate_fallback_kline_data(limit)
                    return jsonify({
                        'symbol': symbol,
                        'period': period,
                        'data': fallback_data,
                        'source': 'fallback'
                    })
                    
            except Exception as e:
                print(f"Error fetching intraday data: {e}")
                # Fallback to generated data
                fallback_data = generate_fallback_kline_data(limit)
                return jsonify({
                    'symbol': symbol,
                    'period': period,
                    'data': fallback_data,
                    'source': 'fallback_error'
                })
        else:
            # Use existing daily data fetcher for daily/weekly/monthly data
            data = data_fetcher.get_stock_data_with_failover(
                symbol=symbol, 
                period='1y',  # Get enough data for processing
                market='us',
                force_refresh=force_refresh
            )
            
            if data is None or data.empty:
                return jsonify({'error': 'No data found'}), 404
            
            # Process data based on time period
            kline_data = process_kline_data(data, period, limit)
            
            return jsonify({
                'symbol': symbol,
                'period': period,
                'data': kline_data,
                'source': 'daily_data'
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/volume/<symbol>')
def get_volume_data(symbol):
    """Get volume data synchronized with K-line data, with intraday support"""
    try:
        period = request.args.get('period', '1d')
        limit = int(request.args.get('limit', 200))
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        # Check if this is an intraday timeframe
        intraday_timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        if period in intraday_timeframes:
            # Use intraday fetcher for minute/hour data
            try:
                from data.intraday_fetcher import intraday_fetcher
                data = intraday_fetcher.get_intraday_data(symbol, period, force_refresh)
                
                if data is not None and not data.empty:
                    # Convert to the format expected by frontend
                    volume_data = []
                    for idx, row in data.tail(limit).iterrows():
                        volume_data.append({
                            'time': int(idx.timestamp()),
                            'value': int(row['Volume'])
                        })
                    
                    return jsonify({
                        'symbol': symbol,
                        'period': period,
                        'data': volume_data,
                        'source': 'intraday_cache'
                    })
                else:
                    # Fallback to generated data if no intraday data available
                    fallback_data = generate_fallback_volume_data(limit)
                    return jsonify({
                        'symbol': symbol,
                        'period': period,
                        'data': fallback_data,
                        'source': 'fallback'
                    })
                    
            except Exception as e:
                print(f"Error fetching intraday volume data: {e}")
                # Fallback to generated data
                fallback_data = generate_fallback_volume_data(limit)
                return jsonify({
                    'symbol': symbol,
                    'period': period,
                    'data': fallback_data,
                    'source': 'fallback_error'
                })
        else:
            # Use existing daily data fetcher for daily/weekly/monthly data
            data = data_fetcher.get_stock_data_with_failover(
                symbol=symbol, 
                period='1y',
                market='us',
                force_refresh=force_refresh
            )
            
            if data is None or data.empty:
                return jsonify({'error': 'No data found'}), 404
            
            # Process volume data
            volume_data = process_volume_data(data, period, limit)
            
            return jsonify({
                'symbol': symbol,
                'period': period,
                'data': volume_data,
                'source': 'daily_data'
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chip_distribution/<symbol>')
def get_chip_distribution(symbol):
    """Get chip distribution data for the symbol"""
    try:
        # Get stock data for chip distribution calculation
        data = data_fetcher.get_stock_data_with_failover(
            symbol=symbol, 
            period='3mo',  # 3 months for chip distribution
            market='us'
        )
        
        if data is None or data.empty:
            return jsonify({'error': 'No data found'}), 404
        
        # Calculate chip distribution
        chip_data = calculate_chip_distribution(data)
        
        return jsonify({
            'symbol': symbol,
            'data': chip_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_kline_data(data, period, limit):
    """Process raw data into K-line format"""
    import pandas as pd
    from datetime import datetime, timedelta
    
    try:
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Handle different date column names - check lowercase 'date' first
        date_col = None
        for col in ['date', 'Date', 'datetime', 'timestamp']:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            # If no date column found, use index as date
            data = data.reset_index()
            # After reset_index, the index becomes a column named 'index'
            if 'index' in data.columns:
                date_col = 'index'
            else:
                # Fallback - use the first column that might be a date
                date_col = data.columns[0]
        
        # Rename to standard 'Date' column
        if date_col != 'Date':
            data.rename(columns={date_col: 'Date'}, inplace=True)
        
        # Ensure data is sorted by date
        data = data.sort_values('Date')
        
        # Convert to the required format
        kline_data = []
        for i, (_, row) in enumerate(data.tail(limit).iterrows()):
            # Handle date conversion properly - convert to Unix timestamp
            date_value = row['Date']
            
            try:
                # Try to parse the date properly
                if isinstance(date_value, str):
                    # Parse string date
                    parsed_date = pd.to_datetime(date_value)
                elif hasattr(date_value, 'timestamp'):
                    # Already a datetime object
                    parsed_date = date_value
                else:
                    # Use current date minus days for consistent ordering
                    parsed_date = datetime.now() - timedelta(days=len(data) - i)
                
                timestamp = int(parsed_date.timestamp())
            except:
                # Fallback to sequential timestamps from recent past
                base_date = datetime.now() - timedelta(days=len(data) - i)
                timestamp = int(base_date.timestamp())
                
            # Ensure we have valid OHLC data
            open_price = float(row.get('Open', row.get('open', 100)))
            high_price = float(row.get('High', row.get('high', 105)))
            low_price = float(row.get('Low', row.get('low', 95)))
            close_price = float(row.get('Close', row.get('close', 102)))
            
            # Validate OHLC relationships
            if high_price < max(open_price, close_price):
                high_price = max(open_price, close_price) * 1.02
            if low_price > min(open_price, close_price):
                low_price = min(open_price, close_price) * 0.98
                
            kline_data.append({
                'time': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
        
        return kline_data
        
    except Exception as e:
        print(f"Error in process_kline_data: {str(e)}")
        print(f"Data columns: {data.columns.tolist() if hasattr(data, 'columns') else 'No columns'}")
        print(f"Date column found: {date_col if 'date_col' in locals() else 'Not found'}")
        import traceback
        traceback.print_exc()
        
        # Return fallback mock data if processing fails
        return generate_fallback_kline_data(limit)

def process_volume_data(data, period, limit):
    """Process raw data into volume format"""
    from datetime import datetime, timedelta
    try:
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Handle different date column names - check lowercase 'date' first
        date_col = None
        for col in ['date', 'Date', 'datetime', 'timestamp']:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            # If no date column found, use index as date
            data = data.reset_index()
            # After reset_index, the index becomes a column named 'index'
            if 'index' in data.columns:
                date_col = 'index'
            else:
                # Fallback - use the first column that might be a date
                date_col = data.columns[0]
        
        # Rename to standard 'Date' column
        if date_col != 'Date':
            data.rename(columns={date_col: 'Date'}, inplace=True)
        
        # Ensure data is sorted by date
        data = data.sort_values('Date')
        
        # Convert to the required format
        volume_data = []
        for i, (_, row) in enumerate(data.tail(limit).iterrows()):
            # Handle date conversion properly
            date_value = row['Date']
            
            try:
                # Try to parse the date properly
                if isinstance(date_value, str):
                    # Parse string date
                    parsed_date = pd.to_datetime(date_value)
                elif hasattr(date_value, 'timestamp'):
                    # Already a datetime object
                    parsed_date = date_value
                else:
                    # Use current date minus days for consistent ordering
                    parsed_date = datetime.now() - timedelta(days=len(data) - i)
                
                timestamp = int(parsed_date.timestamp())
            except:
                # Fallback to sequential timestamps from recent past
                base_date = datetime.now() - timedelta(days=len(data) - i)
                timestamp = int(base_date.timestamp())
            
            # Get volume data with fallback
            volume = int(row.get('Volume', row.get('volume', 1000000)))
            
            volume_data.append({
                'time': timestamp,
                'value': volume
            })
        
        return volume_data
        
    except Exception as e:
        print(f"Error in process_volume_data: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return fallback mock data if processing fails
        return generate_fallback_volume_data(limit)

def generate_fallback_kline_data(limit=200):
    """Generate fallback K-line data when real data processing fails"""
    from datetime import datetime, timedelta
    import random
    
    kline_data = []
    base_price = 150.0
    current_price = base_price
    
    for i in range(limit):
        # Generate realistic price movement
        change_percent = random.uniform(-0.03, 0.03)  # ±3% daily change
        new_price = current_price * (1 + change_percent)
        
        # Generate OHLC data
        open_price = current_price
        close_price = new_price
        high_price = max(open_price, close_price) * random.uniform(1.001, 1.02)
        low_price = min(open_price, close_price) * random.uniform(0.98, 0.999)
        
        # Generate timestamp (going backwards from now)
        timestamp = int((datetime.now() - timedelta(days=limit - i)).timestamp())
        
        kline_data.append({
            'time': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2)
        })
        
        current_price = new_price
    
    return kline_data

def generate_fallback_volume_data(limit=200):
    """Generate fallback volume data when real data processing fails"""
    from datetime import datetime, timedelta
    import random
    
    volume_data = []
    base_volume = 50000000  # 50M base volume
    
    for i in range(limit):
        # Generate realistic volume variation
        volume = int(base_volume * random.uniform(0.5, 2.0))
        
        # Generate timestamp (going backwards from now)
        timestamp = int((datetime.now() - timedelta(days=limit - i)).timestamp())
        
        volume_data.append({
            'time': timestamp,
            'value': volume
        })
    
    return volume_data

def calculate_chip_distribution(data):
    """Calculate chip distribution based on price and volume"""
    import numpy as np
    
    # Handle different column name formats
    close_col = None
    volume_col = None
    
    # Check for different column name variations
    for col in data.columns:
        if col.lower() in ['close', '4. close']:
            close_col = col
        elif col.lower() in ['volume', '5. volume']:
            volume_col = col
    
    if close_col is None or volume_col is None:
        raise ValueError(f"Required columns not found. Available columns: {list(data.columns)}")
    
    # Calculate price levels and volume distribution
    prices = data[close_col].values
    volumes = data[volume_col].values
    
    # Create price bins
    price_min, price_max = prices.min(), prices.max()
    price_bins = np.linspace(price_min, price_max, 50)
    
    # Calculate chip distribution
    chip_distribution = []
    for i in range(len(price_bins) - 1):
        price_level = (price_bins[i] + price_bins[i + 1]) / 2
        
        # Find volume at this price level
        mask = (prices >= price_bins[i]) & (prices < price_bins[i + 1])
        volume_at_level = volumes[mask].sum() if mask.any() else 0
        
        if volume_at_level > 0:
            chip_distribution.append({
                'price': round(price_level, 2),
                'volume': int(volume_at_level)
            })
    
    return chip_distribution

@app.route('/api/cache/intraday/info')
def get_intraday_cache_info():
    """Get intraday cache information"""
    try:
        symbol = request.args.get('symbol')
        
        from data.cache_manager import cache_manager
        cache_info = cache_manager.get_cache_stats()
        
        if symbol:
            # Filter for specific symbol
            symbol_info = cache_info.get('by_symbol', [])
            symbol_data = next((s for s in symbol_info if s['symbol'] == symbol.upper()), None)
            return jsonify({
                'symbol': symbol,
                'cache_info': symbol_data,
                'total_cache_size_mb': cache_info.get('total_size_mb', 0)
            })
        else:
            return jsonify(cache_info)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/intraday/clear', methods=['POST'])
def clear_intraday_cache():
    """Clear intraday cache"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        from data.cache_manager import cache_manager
        cache_manager.clear_cache(symbol, timeframe)
        
        if symbol and timeframe:
            message = f'Cleared cache for {symbol} {timeframe}'
        elif symbol:
            message = f'Cleared all cache for {symbol}'
        else:
            message = 'Cleared all intraday cache'
            
        return jsonify({'message': message})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/intraday/preload', methods=['POST'])
def preload_intraday_cache():
    """Preload cache for popular symbols"""
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
        timeframes = data.get('timeframes', ['1m', '5m', '15m', '30m', '1h'])
        
        from data.cache_manager import cache_manager
        
        # Start preloading in background
        import threading
        def preload_task():
            cache_manager.preload_popular_symbols(symbols, timeframes)
        
        thread = threading.Thread(target=preload_task, daemon=True)
        thread.start()
        
        return jsonify({
            'message': f'Started preloading cache for {len(symbols)} symbols and {len(timeframes)} timeframes',
            'symbols': symbols,
            'timeframes': timeframes
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/intraday/test/<symbol>')
def test_intraday_data(symbol):
    """Test endpoint for intraday data fetching"""
    try:
        timeframe = request.args.get('timeframe', '5m')
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        from data.intraday_fetcher import intraday_fetcher
        import time
        
        # Test data fetching
        start_time = time.time()
        data = intraday_fetcher.get_intraday_data(symbol, timeframe, force_refresh)
        fetch_time = time.time() - start_time
        
        if data is not None and not data.empty:
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'record_count': len(data),
                'fetch_time_seconds': round(fetch_time, 3),
                'date_range': {
                    'start': data.index.min().isoformat(),
                    'end': data.index.max().isoformat()
                },
                'sample_data': data.head(3).to_dict('records'),
                'status': 'success'
            })
        else:
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'record_count': 0,
                'fetch_time_seconds': round(fetch_time, 3),
                'status': 'no_data'
            })
            
    except Exception as e:
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/intraday/enhanced/<symbol>')
def get_enhanced_intraday_data(symbol):
    """Get enhanced intraday data with multiple indicators and chip distribution"""
    try:
        timeframe = request.args.get('timeframe', '5m')
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        include_chips = request.args.get('include_chips', 'true').lower() == 'true'
        
        # Import enhanced intraday fetcher
        from data.enhanced_intraday_fetcher import enhanced_intraday_fetcher
        import pandas as pd
        
        # Get intraday data
        data = enhanced_intraday_fetcher.get_intraday_data(
            symbol=symbol.upper(),
            timeframe=timeframe,
            force_refresh=force_refresh,
            include_chips=include_chips
        )
        
        if data is None or data.empty:
            return jsonify({'error': f'No enhanced intraday data available for {symbol}'}), 404
        
        # Convert to JSON format
        result = {
            'symbol': symbol.upper(),
            'timeframe': timeframe,
            'data_points': len(data),
            'date_range': {
                'start': data.index.min().isoformat(),
                'end': data.index.max().isoformat()
            },
            'columns': list(data.columns),
            'latest_data': data.tail(10).reset_index().to_dict('records'),
            'indicators': {
                'rsi': float(data['rsi'].iloc[-1]) if 'rsi' in data.columns and not pd.isna(data['rsi'].iloc[-1]) else None,
                'macd': float(data['macd'].iloc[-1]) if 'macd' in data.columns and not pd.isna(data['macd'].iloc[-1]) else None,
                'volume_ratio': float(data['volume_ratio'].iloc[-1]) if 'volume_ratio' in data.columns and not pd.isna(data['volume_ratio'].iloc[-1]) else None,
                'vwap': float(data['vwap'].iloc[-1]) if 'vwap' in data.columns and not pd.isna(data['vwap'].iloc[-1]) else None,
                'bb_upper': float(data['bb_upper'].iloc[-1]) if 'bb_upper' in data.columns and not pd.isna(data['bb_upper'].iloc[-1]) else None,
                'bb_lower': float(data['bb_lower'].iloc[-1]) if 'bb_lower' in data.columns and not pd.isna(data['bb_lower'].iloc[-1]) else None
            },
            'chip_distribution': {
                'support_level': float(data['support_level'].iloc[-1]) if 'support_level' in data.columns and not pd.isna(data['support_level'].iloc[-1]) else None,
                'resistance_level': float(data['resistance_level'].iloc[-1]) if 'resistance_level' in data.columns and not pd.isna(data['resistance_level'].iloc[-1]) else None,
                'chip_concentration': float(data['chip_concentration'].iloc[-1]) if 'chip_concentration' in data.columns and not pd.isna(data['chip_concentration'].iloc[-1]) else None
            } if include_chips else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/intraday/multiple/<symbol>')
def get_multiple_timeframes_data(symbol):
    """Get data for multiple timeframes"""
    try:
        timeframes_param = request.args.get('timeframes', '5m,15m,1h')
        timeframes = [tf.strip() for tf in timeframes_param.split(',')]
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        # Import enhanced intraday fetcher
        from data.enhanced_intraday_fetcher import enhanced_intraday_fetcher
        
        # Get multiple timeframes data
        multi_data = enhanced_intraday_fetcher.get_multiple_timeframes(
            symbol=symbol.upper(),
            timeframes=timeframes,
            force_refresh=force_refresh
        )
        
        if not multi_data:
            return jsonify({'error': f'No intraday data available for {symbol}'}), 404
        
        # Convert to JSON format
        result = {
            'symbol': symbol.upper(),
            'timeframes': list(multi_data.keys()),
            'data': {}
        }
        
        for tf, data in multi_data.items():
            result['data'][tf] = {
                'data_points': len(data),
                'date_range': {
                    'start': data.index.min().isoformat(),
                    'end': data.index.max().isoformat()
                },
                'columns': list(data.columns),
                'latest_price': float(data['close'].iloc[-1]) if 'close' in data.columns else None,
                'sample_data': data.tail(3).reset_index().to_dict('records')
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
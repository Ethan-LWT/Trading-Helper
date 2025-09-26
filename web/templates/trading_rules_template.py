"""
Professional Trading Rules Template for AI Strategy Generation
Based on high-probability short-term trading signals and risk management principles
"""

TRADING_RULES_TEMPLATE = """
### Professional Short-Term Trading Strategy Template

## 1. TREND IDENTIFICATION SYSTEM

### Primary Trend Indicators:
- **Dual EMA System**: 5-period EMA and 20-period EMA
  - **Bullish Trend**: Price > 5EMA > 20EMA, with EMAs diverging upward
  - **Bearish Trend**: Price < 5EMA < 20EMA, with EMAs diverging downward
- **200EMA Filter**: Acts as bull/bear market divider
  - Above 200EMA: Only consider long positions
  - Below 200EMA: Prioritize short positions or stay in cash

## 2. HIGH-PROBABILITY ENTRY SIGNALS

### A. Trend Pullback Entry (High Win Rate Strategy)
**Scenario**: Clear uptrend with price pulling back to key support
**Entry Conditions**:
1. **Support Touch**: Price retraces to rising 20EMA or previous key price level
2. **Reversal Pattern**: Bullish pin bar (hammer), bullish engulfing, or morning star at support
3. **Indicator Confirmation**: RSI bouncing from oversold (<30) OR MACD bullish crossover above zero line
4. **Volume Confirmation**: Decreasing volume on pullback, increasing volume on bounce

### B. Breakout Entry (High Risk/Reward Strategy)  
**Scenario**: Price consolidation followed by directional breakout
**Entry Conditions**:
1. **Level Break**: Price breaks above key resistance (long) or below support (short) with volume
2. **Pattern Break**: Breakout from triangle, flag, pennant, or head & shoulders neckline
3. **Volume Surge**: Breakout MUST be accompanied by significantly higher volume
4. **Follow-through**: Next candle confirms direction with continued momentum

### C. Momentum Following Entry (Trend Continuation Strategy)
**Scenario**: Strong directional move with sustained momentum
**Entry Conditions**:
1. **Large Candle**: Candle size exceeds recent average range significantly
2. **Strong Indicators**: RSI sustained above 70 (can extend to 80 in strong trends), MACD with wide divergence
3. **Early Entry**: Must enter in early stages of momentum move
4. **Tight Stops**: Requires very close stop-loss due to higher risk

## 3. COMPLETE TRADING RULES FRAMEWORK

### Entry Rules (ALL Must Be Satisfied):
- **Trend Confirmation** + **Specific Entry Signal** + **Volume Confirmation**
- Execute with market or limit order without hesitation
- Never enter without all three confirmations

### Stop-Loss Rules (CRITICAL - Your Lifeline):
**Position Sizing Based on Stop Distance**:
- **Long Stop**: Below entry candle low or key support level
- **Short Stop**: Above entry candle high or key resistance level
- **Risk Per Trade**: Maximum 2% of total capital per trade
- **Position Size Formula**: `Position Size = (Total Capital × 2%) ÷ (Entry Price - Stop Price)`

### Take-Profit Rules:
**Risk/Reward Ratio**: Minimum 1.5:1, ideally 2:1 or 3:1
**Methods**:
1. **Fixed Ratio**: Set profit target at 2x or 3x stop distance
2. **Trailing Stops**: 
   - Move stop to breakeven after 1:1 ratio achieved
   - Trail stop below rising 5EMA or 10EMA in strong trends
   - Exit when EMA support breaks

### Position Management:
- **Single Trade Risk**: ≤ 2% of total capital
- **Total Portfolio Risk**: ≤ 6% across all open positions
- **Never Average Down**: Do not add to losing positions
- **Scale Out**: Consider taking partial profits at key resistance levels

### Time Frame and Instrument Selection:
- **Analysis Timeframes**: 1H and Daily charts for trend, 5min and 15min for entries
- **Instrument Criteria**: High liquidity stocks/ETFs with >5M daily volume
- **Avoid**: Penny stocks, low-volume securities, illiquid markets

### Risk Management and Psychology:
- **Pre-Market Planning**: All trade plans must be made before market open
- **Execution Only**: During market hours, only execute predetermined plans
- **Mandatory Break**: After 2-3 consecutive losses, step away from screens
- **Daily Review**: Analyze all trades post-market, document lessons learned
- **Emotional Control**: No revenge trading, no FOMO, no greed-driven decisions

## 4. COMPLETE TRADE EXAMPLE

**Setup**: Stock XYZ in daily uptrend (Price > 50EMA > 200EMA), pulling back to 20EMA on 15min chart

**Signal**: Bullish hammer candle at 20EMA support, RSI bouncing from 35 to 40+

**Execution**:
- **Entry**: Market buy at $100 (next 5min candle confirms with volume)
- **Stop Loss**: $98.50 (below hammer low)
- **Risk Per Share**: $1.50
- **Account Size**: $50,000
- **Max Loss**: $50,000 × 2% = $1,000
- **Position Size**: $1,000 ÷ $1.50 = 666 shares
- **Take Profit**: $103 (2:1 risk/reward ratio)

**Management**: 
- At $101.50 (1:1 ratio): Move stop to breakeven ($100)
- Let profits run to $103 target or trailing stop trigger

## 5. STRATEGY OPTIMIZATION CRITERIA

**Minimum Performance Standards**:
- **Win Rate**: >45% for pullback strategies, >35% for breakout strategies
- **Average Risk/Reward**: >1.5:1 across all trades
- **Maximum Drawdown**: <15% of account value
- **Profit Factor**: >1.3 (Gross Profit ÷ Gross Loss)

**Backtesting Requirements**:
- Test across multiple market conditions (bull, bear, sideways)
- Minimum 100 trades for statistical significance
- Include transaction costs and slippage
- Test on different timeframes and instruments

**Strategy Rejection Criteria**:
- Negative total return over any 12-month period
- Win rate <30% consistently
- Average loss > 2x average win
- Maximum single loss >5% of account

This template provides the foundation for generating profitable, risk-managed trading strategies that can be systematically backtested and optimized.
"""

def get_trading_rules_prompt(symbol: str, market_condition: str = "neutral") -> str:
    """
    Generate a specific prompt for AI strategy creation based on trading rules template
    """
    
    base_prompt = f"""
You are a professional quantitative trading strategist. Create a specific trading strategy for {symbol} using the following professional trading framework:

{TRADING_RULES_TEMPLATE}

**Current Market Context**: {market_condition}
**Target Symbol**: {symbol}

**Requirements**:
1. Generate specific entry and exit conditions based on the template above
2. Include exact technical indicator parameters (EMA periods, RSI levels, etc.)
3. Define precise risk management rules (stop-loss, position sizing)
4. Specify minimum risk/reward ratios
5. Include volume and momentum filters
6. Ensure strategy can be backtested programmatically

**Output Format**:
Return a JSON object with the following structure:
{{
    "strategy_name": "Descriptive strategy name",
    "description": "Brief strategy description",
    "entry_conditions": {{
        "trend_filter": "Specific trend requirements",
        "technical_setup": "Exact technical conditions",
        "volume_filter": "Volume requirements",
        "additional_filters": "Any other conditions"
    }},
    "exit_conditions": {{
        "stop_loss": "Stop loss rules",
        "take_profit": "Take profit rules", 
        "trailing_stop": "Trailing stop rules if applicable"
    }},
    "risk_management": {{
        "max_risk_per_trade": "2%",
        "position_sizing": "Risk-based position sizing formula",
        "max_portfolio_risk": "6%"
    }},
    "parameters": {{
        "ema_fast": 5,
        "ema_slow": 20,
        "ema_filter": 200,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "volume_ma_period": 20,
        "min_risk_reward": 1.5
    }},
    "timeframes": {{
        "analysis": ["1h", "1d"],
        "entry": ["5m", "15m"]
    }},
    "expected_performance": {{
        "win_rate": "Expected win rate percentage",
        "avg_risk_reward": "Expected average risk/reward ratio",
        "max_drawdown": "Expected maximum drawdown"
    }}
}}

Create a strategy that follows these professional trading principles and has a high probability of generating positive returns when backtested.
"""
    
    return base_prompt

def validate_strategy_performance(backtest_results: dict) -> dict:
    """
    Validate if a strategy meets minimum performance criteria
    """
    
    total_return = backtest_results.get('total_return', 0)
    win_rate = backtest_results.get('win_rate', 0)
    avg_risk_reward = backtest_results.get('avg_risk_reward', 0)
    max_drawdown = backtest_results.get('max_drawdown', 0)
    profit_factor = backtest_results.get('profit_factor', 0)
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'recommendations': []
    }
    
    # Check minimum performance criteria
    if total_return <= 0:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Negative total return")
        validation_results['recommendations'].append("Adjust entry/exit conditions to improve profitability")
    
    if win_rate < 30:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Win rate too low: {win_rate}%")
        validation_results['recommendations'].append("Improve entry signal quality or add filters")
    
    if avg_risk_reward < 1.2:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Risk/reward ratio too low: {avg_risk_reward}")
        validation_results['recommendations'].append("Adjust take-profit targets or tighten stop-losses")
    
    if max_drawdown > 20:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Maximum drawdown too high: {max_drawdown}%")
        validation_results['recommendations'].append("Implement better risk management or position sizing")
    
    if profit_factor < 1.1:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Profit factor too low: {profit_factor}")
        validation_results['recommendations'].append("Reduce trade frequency or improve signal quality")
    
    return validation_results

def generate_strategy_improvement_prompt(backtest_results: dict, validation_results: dict) -> str:
    """
    Generate prompt for AI to improve strategy based on backtest results
    """
    
    issues = validation_results.get('issues', [])
    recommendations = validation_results.get('recommendations', [])
    
    improvement_prompt = f"""
The previous trading strategy showed the following performance issues:

**Backtest Results**:
- Total Return: {backtest_results.get('total_return', 0):.2f}%
- Win Rate: {backtest_results.get('win_rate', 0):.1f}%
- Average Risk/Reward: {backtest_results.get('avg_risk_reward', 0):.2f}
- Maximum Drawdown: {backtest_results.get('max_drawdown', 0):.1f}%
- Profit Factor: {backtest_results.get('profit_factor', 0):.2f}

**Identified Issues**:
{chr(10).join([f"- {issue}" for issue in issues])}

**Improvement Recommendations**:
{chr(10).join([f"- {rec}" for rec in recommendations])}

Please create an improved trading strategy that addresses these issues while following the professional trading rules template. Focus on:

1. **Improving Entry Quality**: Add more selective filters to reduce false signals
2. **Optimizing Risk/Reward**: Adjust stop-loss and take-profit levels
3. **Better Risk Management**: Implement more conservative position sizing
4. **Market Condition Filters**: Add filters to avoid trading in unfavorable conditions

Use the same JSON output format as before, but with improved parameters and conditions that should result in positive returns when backtested.
"""
    
    return improvement_prompt
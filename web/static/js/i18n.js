/**
 * Internationalization (i18n) module for Trading AI Dashboard
 * Supports Chinese and English language switching
 */

// Language translations
const translations = {
    zh: {
        // Header and Navigation
        'trading_ai_dashboard': 'Trading AI 仪表板',
        'push_notifications': '推送功能',
        'watchlist_title': '自选股票',
        'remove_stock': '删除股票',
        
        // Chart and Data
        'price_chart': '价格走势图',
        'timeframe_1m': '1分',
        'timeframe_5m': '5分',
        'timeframe_15m': '15分',
        'timeframe_30m': '30分',
        'timeframe_1h': '1时',
        'timeframe_1d': '日线',
        'timeframe_1w': '周线',
        'timeframe_1M': '月线',
        'current_price': '当前价格',
        'price_change': '涨跌幅',
        'volume': '成交量',
        'open_price': '开盘价',
        'high_price': '最高价',
        'low_price': '最低价',
        
        // Volume Monitoring
        'volume_monitoring': '交易量变化检测',
        'monitoring_threshold': '监控阈值:',
        'start_monitoring': '启动监控',
        'stop_monitoring': '停止监控',
        'click_to_start_monitoring': '点击启动交易量监控',
        'volume_monitoring_started': '交易量监控已启动',
        'abnormal_volume': '异常交易量:',
        
        // Strategy
        'strategy_creation': '策略制定',
        'target_stock': '目标股票',
        'enter_stock_code': '输入股票代码或使用当前选中股票',
        'strategy_type': '策略类型',
        'moving_average': '移动平均线',
        'rsi_strategy': 'RSI策略',
        'macd_strategy': 'MACD策略',
        'bollinger_strategy': '布林带策略',
        'ai_strategy': 'AI智能策略',
        'ai_model_selection': 'AI模型选择',
        'local_ollama': '本地Ollama (Llama3)',
        'intelligent_analysis': '智能分析',
        'create_strategy': '创建策略',
        'backtest_verification': '回测验证',
        'deploy_strategy': '部署策略',
        'stop_strategy': '停止策略',
        'strategy_status_inactive': '策略状态：未激活',
        
        // Backtest
        'backtest_price_chart': '回测价格走势图',
        'overall_performance': '整体表现',
        'total_return': '总收益率',
        'max_drawdown': '最大回撤',
        'sharpe_ratio': '夏普比率',
        'annualized_volatility': '年化波动率',
        'trading_statistics': '交易统计',
        'total_trades': '总交易次数',
        'win_rate': '胜率',
        'avg_trade_return': '平均交易收益',
        'avg_holding_days': '平均持仓天数',
        'trade_start_date': '交易开始日期',
        'trade_end_date': '交易结束日期',
        'daily_analysis': '每日分析',
        'daily_trading_frequency': '每日交易频率',
        'daily_profit_frequency': '每日盈利频率',
        'days_with_trades': '有交易的天数',
        'days_with_profits': '有盈利的天数',
        'avg_daily_return': '平均每日收益',
        'strategy_evaluation': '策略评估',
        'sufficient_trading_frequency': '交易频率充足',
        'low_trading_frequency': '交易频率偏低',
        'good_profit_frequency': '盈利频率良好',
        'need_improve_profit_frequency': '盈利频率需改善',
        'good_win_rate': '胜率表现良好',
        'need_improve_win_rate': '胜率需要提升',
        
        // Analysis Modal
        'intelligent_analysis_results': '智能分析结果',
        'technical_analysis': '技术分析',
        'fundamental_analysis': '基本面分析',
        'sentiment_analysis': '市场情绪分析',
        'ai_prediction': 'AI预测',
        'analyzing': '正在分析...',
        'close': '关闭',
        
        // AI Optimization
        'ai_strategy_optimization': 'AI策略优化',
        'stock_code': '股票代码:',
        'ai_model': 'AI模型:',
        'local_ai_recommended': '本地AI (推荐)',
        'market_condition': '市场状况:',
        'neutral': '中性',
        'bullish': '看涨',
        'bearish': '看跌',
        'volatile': '波动',
        'start_optimization': '开始优化',
        'generate_simple_strategy': '生成简单策略',
        'optimizing_strategy': '正在优化策略... 这可能需要几分钟时间。',
        'initializing_optimization': '初始化优化...',
        
        // Add Stock Modal
        'add_stock_code': '添加股票代码',
        'select_market': '选择市场:',
        'us_market': '美股 (如: AAPL, TSLA)',
        'hk_market': '港股 (如: 0700, 0941)',
        'cn_market': 'A股 (如: 000001, 600036)',
        'enter_stock_code_placeholder': '请输入股票代码',
        'us_format_hint': '美股格式: 字母代码 (如: AAPL, TSLA)',
        'hk_format_hint': '港股格式: 4位数字 (如: 0700, 0941)',
        'cn_format_hint': 'A股格式: 6位数字 (如: 000001, 600036)',
        'confirm': '确认',
        'cancel': '取消',
        
        // Messages and Alerts
        'please_select_stock': '请选择股票',
        'analyzing_target': '正在进行{type}分析 {target}...',
        'analysis_failed': '分析失败: {error}',
        'strategy_creation_failed': '策略创建失败: {error}',
        'backtest_failed': '回测失败，请重试: {error}',
        'ai_strategy_backtest_failed': 'AI策略回测失败: {error}',
        'no_strategy_to_optimize': '没有可优化的AI策略',
        'optimizing_strategy_wait': '正在进行策略优化...<br>这可能需要几分钟时间，请耐心等待',
        'strategy_optimization_failed': '策略优化失败: {error}',
        'strategy_deployment_failed': '策略部署失败: {error}',
        'strategy_stopped': '策略已停止<br>状态：未激活',
        'push_notifications_enabled': '推送已开启',
        'push_notifications_disabled': '推送功能',
        'chip_distribution': '筹码分布',
        'no_data': '暂无数据',
        
        // Validation Messages
        'enter_valid_us_code': '请输入有效的美股代码 (1-5个字母，如: AAPL)',
        'enter_valid_hk_code': '请输入有效的港股代码 (1-4位数字，如: 700 或 0700)',
        'enter_valid_cn_code': '请输入有效的A股代码 (6位数字，如: 000001)',
        'stock_added': '已添加{market}: {symbol}',
        'us_stock': '美股',
        'hk_stock': '港股',
        'cn_stock': 'A股',
        
        // Units and Formatting
        'times': '次',
        'days': '天',
        'percent': '%',
        'current_volume': '当前交易量: {volume}',
        'avg_volume_30d': '30日平均: {volume}',
        'volume_change': '变化: {change}%',
        'relative_volume': '相对交易量: {ratio}x',
        'vs_yesterday': 'vs昨日: {change}%',
        'vs_week_avg': 'vs周均: {change}%',
        
        
        // Additional translations
        'optimization_complete': '优化完成！',
        'strategy_generation_complete': '策略生成完成！',
        'analyzing_target': '正在进行{type}分析 {target}...',
        'stock_added': '已添加{market}: {symbol}',
        'enter_stock_code_placeholder': '请输入股票代码',
        // Language Switch
        'language': '语言',
        'chinese': '中文',
        'english': 'English',
        
        // Additional translations for stock page
        'back_to_dashboard': 'Back to Dashboard',
        
        // Additional translations for stock page
        'back_to_dashboard': '返回仪表板'
    },
    
    en: {
        // Header and Navigation
        'trading_ai_dashboard': 'Trading AI Dashboard',
        'push_notifications': 'Push Notifications',
        'watchlist_title': 'Watchlist',
        'remove_stock': 'Remove Stock',
        
        // Chart and Data
        'price_chart': 'Price Chart',
        'timeframe_1m': '1m',
        'timeframe_5m': '5m',
        'timeframe_15m': '15m',
        'timeframe_30m': '30m',
        'timeframe_1h': '1h',
        'timeframe_1d': '1D',
        'timeframe_1w': '1W',
        'timeframe_1M': '1M',
        'current_price': 'Current Price',
        'price_change': 'Change',
        'volume': 'Volume',
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        
        // Volume Monitoring
        'volume_monitoring': 'Volume Change Detection',
        'monitoring_threshold': 'Threshold:',
        'start_monitoring': 'Start Monitoring',
        'stop_monitoring': 'Stop Monitoring',
        'click_to_start_monitoring': 'Click to start volume monitoring',
        'volume_monitoring_started': 'Volume monitoring started',
        'abnormal_volume': 'Abnormal Volume:',
        
        // Strategy
        'strategy_creation': 'Strategy Creation',
        'target_stock': 'Target Stock',
        'enter_stock_code': 'Enter stock symbol or use current selection',
        'strategy_type': 'Strategy Type',
        'moving_average': 'Moving Average',
        'rsi_strategy': 'RSI Strategy',
        'macd_strategy': 'MACD Strategy',
        'bollinger_strategy': 'Bollinger Bands',
        'ai_strategy': 'AI Strategy',
        'ai_model_selection': 'AI Model',
        'local_ollama': 'Local Ollama (Llama3)',
        'intelligent_analysis': 'Smart Analysis',
        'create_strategy': 'Create Strategy',
        'backtest_verification': 'Backtest',
        'deploy_strategy': 'Deploy Strategy',
        'stop_strategy': 'Stop Strategy',
        'strategy_status_inactive': 'Strategy Status: Inactive',
        
        // Backtest
        'backtest_price_chart': 'Backtest Price Chart',
        'overall_performance': 'Overall Performance',
        'total_return': 'Total Return',
        'max_drawdown': 'Max Drawdown',
        'sharpe_ratio': 'Sharpe Ratio',
        'annualized_volatility': 'Annualized Volatility',
        'trading_statistics': 'Trading Statistics',
        'total_trades': 'Total Trades',
        'win_rate': 'Win Rate',
        'avg_trade_return': 'Avg Trade Return',
        'avg_holding_days': 'Avg Holding Days',
        'trade_start_date': 'Trade Start Date',
        'trade_end_date': 'Trade End Date',
        'daily_analysis': 'Daily Analysis',
        'daily_trading_frequency': 'Daily Trading Frequency',
        'daily_profit_frequency': 'Daily Profit Frequency',
        'days_with_trades': 'Days with Trades',
        'days_with_profits': 'Days with Profits',
        'avg_daily_return': 'Avg Daily Return',
        'strategy_evaluation': 'Strategy Evaluation',
        'sufficient_trading_frequency': 'Sufficient Trading Frequency',
        'low_trading_frequency': 'Low Trading Frequency',
        'good_profit_frequency': 'Good Profit Frequency',
        'need_improve_profit_frequency': 'Need Improve Profit Frequency',
        'good_win_rate': 'Good Win Rate',
        'need_improve_win_rate': 'Need Improve Win Rate',
        
        // Analysis Modal
        'intelligent_analysis_results': 'Smart Analysis Results',
        'technical_analysis': 'Technical Analysis',
        'fundamental_analysis': 'Fundamental Analysis',
        'sentiment_analysis': 'Sentiment Analysis',
        'ai_prediction': 'AI Prediction',
        'analyzing': 'Analyzing...',
        'close': 'Close',
        
        // AI Optimization
        'ai_strategy_optimization': 'AI Strategy Optimization',
        'stock_code': 'Stock Symbol:',
        'ai_model': 'AI Model:',
        'local_ai_recommended': 'Local AI (Recommended)',
        'market_condition': 'Market Condition:',
        'neutral': 'Neutral',
        'bullish': 'Bullish',
        'bearish': 'Bearish',
        'volatile': 'Volatile',
        'start_optimization': 'Start Optimization',
        'generate_simple_strategy': 'Generate Simple Strategy',
        'optimizing_strategy': 'Optimizing strategy... This may take a few minutes.',
        'initializing_optimization': 'Initializing optimization...',
        
        // Add Stock Modal
        'add_stock_code': 'Add Stock Symbol',
        'select_market': 'Select Market:',
        'us_market': 'US Market (e.g., AAPL, TSLA)',
        'hk_market': 'HK Market (e.g., 0700, 0941)',
        'cn_market': 'CN Market (e.g., 000001, 600036)',
        'enter_stock_code_placeholder': 'Enter stock symbol',
        'us_format_hint': 'US format: Letter code (e.g., AAPL, TSLA)',
        'hk_format_hint': 'HK format: 4-digit number (e.g., 0700, 0941)',
        'cn_format_hint': 'CN format: 6-digit number (e.g., 000001, 600036)',
        'confirm': 'Confirm',
        'cancel': 'Cancel',
        
        // Messages and Alerts
        'please_select_stock': 'Please select a stock',
        'analyzing_target': 'Analyzing {type} for {target}...',
        'analysis_failed': 'Analysis failed: {error}',
        'strategy_creation_failed': 'Strategy creation failed: {error}',
        'backtest_failed': 'Backtest failed, please retry: {error}',
        'ai_strategy_backtest_failed': 'AI strategy backtest failed: {error}',
        'no_strategy_to_optimize': 'No AI strategy to optimize',
        'optimizing_strategy_wait': 'Optimizing strategy...<br>This may take a few minutes, please wait',
        'strategy_optimization_failed': 'Strategy optimization failed: {error}',
        'strategy_deployment_failed': 'Strategy deployment failed: {error}',
        'strategy_stopped': 'Strategy stopped<br>Status: Inactive',
        'push_notifications_enabled': 'Push Enabled',
        'push_notifications_disabled': 'Push Notifications',
        'chip_distribution': 'Chip Distribution',
        'no_data': 'No Data',
        
        // Validation Messages
        'enter_valid_us_code': 'Enter valid US stock code (1-5 letters, e.g., AAPL)',
        'enter_valid_hk_code': 'Enter valid HK stock code (1-4 digits, e.g., 700 or 0700)',
        'enter_valid_cn_code': 'Enter valid CN stock code (6 digits, e.g., 000001)',
        'stock_added': 'Added {market}: {symbol}',
        'us_stock': 'US Stock',
        'hk_stock': 'HK Stock',
        'cn_stock': 'CN Stock',
        
        // Units and Formatting
        'times': ' times',
        'days': ' days',
        'percent': '%',
        'current_volume': 'Current Volume: {volume}',
        'avg_volume_30d': '30-day Avg: {volume}',
        'volume_change': 'Change: {change}%',
        'relative_volume': 'Relative Volume: {ratio}x',
        'vs_yesterday': 'vs Yesterday: {change}%',
        'vs_week_avg': 'vs Week Avg: {change}%',
        
        
        // Additional translations
        'optimization_complete': 'Optimization Complete!',
        'strategy_generation_complete': 'Strategy Generation Complete!',
        'analyzing_target': 'Analyzing {type} for {target}...',
        'stock_added': 'Added {market}: {symbol}',
        'enter_stock_code_placeholder': 'Enter stock symbol',
        // Language Switch
        'language': 'Language',
        'chinese': '中文',
        'english': 'English'
    }
};

// Current language (default: Chinese)
let currentLanguage = localStorage.getItem('trading_ai_language') || 'zh';

/**
 * Get translation for a key
 * @param {string} key - Translation key
 * @param {object} params - Parameters for string interpolation
 * @returns {string} Translated text
 */
function t(key, params = {}) {
    let text = translations[currentLanguage][key] || translations['zh'][key] || key;
    
    // Simple string interpolation
    Object.keys(params).forEach(param => {
        text = text.replace(new RegExp(`{${param}}`, 'g'), params[param]);
    });
    
    return text;
}

/**
 * Toggle between languages (simplified version)
 */
function toggleLanguage() {
    const newLang = currentLanguage === 'zh' ? 'en' : 'zh';
    switchLanguage(newLang);
}

/**
 * Update language toggle display
 */
function updateLanguageToggle() {
    const langText = document.getElementById('currentLangText');
    if (langText) {
        langText.textContent = currentLanguage === 'zh' ? 'CN' : 'EN';
    }
}

/**
 * Switch language
 * @param {string} lang - Language code ('zh' or 'en')
 */
function switchLanguage(lang) {
    if (lang !== 'zh' && lang !== 'en') {
        console.warn('Unsupported language:', lang);
        return;
    }
    
    currentLanguage = lang;
    localStorage.setItem('trading_ai_language', lang);
    
    // Update all translatable elements
    updatePageTranslations();
    
    // Update language toggle display
    updateLanguageToggle();
    
    // Update language selector (if exists)
    updateLanguageSelector();
    
    console.log('Language switched to:', lang);
}

/**
 * Update all translatable elements on the page
 */
function updatePageTranslations() {
    // Update elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        element.textContent = t(key);
    });
    
    // Update elements with data-i18n-placeholder attribute
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        element.placeholder = t(key);
    });
    
    // Update elements with data-i18n-title attribute
    document.querySelectorAll('[data-i18n-title]').forEach(element => {
        const key = element.getAttribute('data-i18n-title');
        element.title = t(key);
    });
    
    // Update elements with data-i18n-html attribute (for HTML content)
    document.querySelectorAll('[data-i18n-html]').forEach(element => {
        const key = element.getAttribute('data-i18n-html');
        element.innerHTML = t(key);
    });
}

/**
 * Update language selector UI
 */
function updateLanguageSelector() {
    const selector = document.getElementById('languageSelector');
    if (selector) {
        selector.value = currentLanguage;
    }
}

/**
 * Initialize i18n system
 */
function initI18n() {
    // Apply initial translations
    updatePageTranslations();
    
    // Update language toggle display
    updateLanguageToggle();
    
    // Update language selector (if exists)
    updateLanguageSelector();
    
    console.log('i18n initialized with language:', currentLanguage);
}

/**
 * Create language selector in header
 */
function createLanguageSelector() {
    const header = document.querySelector('.header');
    if (!header) return;
    
    // Check if selector already exists
    if (document.getElementById('languageSelector')) return;
    
    const languageContainer = document.createElement('div');
    languageContainer.className = 'language-selector';
    languageContainer.innerHTML = `
        <select id="languageSelector" onchange="switchLanguage(this.value)" style="
            background: #2b3139;
            color: #ffffff;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 12px;
            cursor: pointer;
        ">
            <option value="zh">中文</option>
            <option value="en">English</option>
        </select>
    `;
    
    header.appendChild(languageContainer);
}

// Export functions for global use
window.t = t;
window.switchLanguage = switchLanguage;
window.toggleLanguage = toggleLanguage;
window.initI18n = initI18n;
window.currentLanguage = () => currentLanguage;
# 股票分钟级数据源指南

## 概述
本指南提供了获取1分钟、5分钟、15分钟、30分钟和60分钟股票价格、交易量和筹码分布数据的完整解决方案。

## 🎯 已实现的数据获取系统

### 增强型分钟级数据获取器 (Enhanced Intraday Fetcher)
位置: `data/enhanced_intraday_fetcher.py`

**支持的时间框架:**
- 1分钟 (1m)
- 5分钟 (5m) 
- 15分钟 (15m)
- 30分钟 (30m)
- 1小时 (1h)

**数据内容:**
- **基础OHLCV数据**: 开盘价、最高价、最低价、收盘价、成交量
- **技术指标**: RSI, MACD, 布林带, 移动平均线, EMA
- **成交量指标**: 成交量比率, 成交量移动平均
- **筹码分布**: VWAP, 支撑阻力位, 筹码集中度

## 📊 数据源配置

### 1. Yahoo Finance (yfinance) - 免费
**优点:**
- 完全免费
- 支持全球股票市场
- 数据质量较好
- 无需API密钥

**限制:**
- 有请求频率限制
- 1分钟数据仅限最近30天

**使用示例:**
```python
from data.enhanced_intraday_fetcher import enhanced_intraday_fetcher

# 获取5分钟数据
data = enhanced_intraday_fetcher.get_intraday_data('AAPL', '5m')
```

### 2. Alpha Vantage - 免费层级
**优点:**
- 高质量的金融数据
- 支持多种时间框架
- 包含技术指标

**限制:**
- 免费版: 5次调用/分钟, 500次调用/天
- 需要API密钥

**设置方法:**
```bash
# 设置环境变量
export ALPHA_VANTAGE_API_KEY='your_api_key_here'
```

**获取API密钥:**
1. 访问: https://www.alphavantage.co/support/#api-key
2. 免费注册获取API密钥

### 3. Polygon.io - 付费服务
**优点:**
- 高质量实时数据
- 支持所有时间框架
- 包含详细的成交量数据

**限制:**
- 需要付费订阅
- 免费层级功能有限

**设置方法:**
```bash
export POLYGON_API_KEY='your_api_key_here'
```

### 4. Twelve Data - 免费层级
**优点:**
- 免费层级: 8次调用/分钟
- 支持多种时间框架
- 数据质量良好

**限制:**
- 免费版数据量有限
- 需要注册

### 5. Finnhub - 免费层级
**优点:**
- 免费层级: 60次调用/分钟
- 实时数据支持
- 多种金融数据

**设置方法:**
```bash
export FINNHUB_API_KEY='your_api_key_here'
```

## 🚀 API接口使用

### Web API端点

#### 1. 获取增强型分钟级数据
```
GET /api/intraday/enhanced/{symbol}?timeframe=5m&include_chips=true
```

**参数:**
- `symbol`: 股票代码 (如: AAPL, TSLA, GOOGL)
- `timeframe`: 时间框架 (1m, 5m, 15m, 30m, 1h)
- `include_chips`: 是否包含筹码分布数据 (true/false)
- `force_refresh`: 强制刷新数据 (true/false)

**响应示例:**
```json
{
  "symbol": "AAPL",
  "timeframe": "5m",
  "data_points": 4032,
  "date_range": {
    "start": "2025-08-27T04:00:00",
    "end": "2025-09-25T19:55:00"
  },
  "indicators": {
    "rsi": 45.2,
    "macd": -0.15,
    "volume_ratio": 1.2,
    "vwap": 150.25
  },
  "chip_distribution": {
    "support_level": 148.50,
    "resistance_level": 152.30,
    "chip_concentration": 0.85
  }
}
```

#### 2. 获取多时间框架数据
```
GET /api/intraday/multiple/{symbol}?timeframes=1m,5m,15m,1h
```

### Python API使用

```python
from data.enhanced_intraday_fetcher import enhanced_intraday_fetcher

# 获取单个时间框架数据
data = enhanced_intraday_fetcher.get_intraday_data(
    symbol='AAPL',
    timeframe='5m',
    force_refresh=False,
    include_chips=True
)

# 获取多个时间框架数据
multi_data = enhanced_intraday_fetcher.get_multiple_timeframes(
    symbol='AAPL',
    timeframes=['1m', '5m', '15m', '30m', '1h'],
    force_refresh=False
)

# 查看缓存信息
cache_info = enhanced_intraday_fetcher.get_cache_info()

# 清理缓存
enhanced_intraday_fetcher.clear_cache(symbol='AAPL', timeframe='5m')
```

## 📈 数据字段说明

### 基础OHLCV字段
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

### 技术指标字段
- `sma_5`: 5期简单移动平均
- `sma_20`: 20期简单移动平均
- `ema_12`: 12期指数移动平均
- `ema_26`: 26期指数移动平均
- `rsi`: 相对强弱指数
- `macd`: MACD指标
- `macd_signal`: MACD信号线
- `macd_histogram`: MACD柱状图
- `bb_upper`: 布林带上轨
- `bb_middle`: 布林带中轨
- `bb_lower`: 布林带下轨

### 成交量指标字段
- `volume_sma`: 成交量移动平均
- `volume_ratio`: 成交量比率

### 筹码分布字段
- `vwap`: 成交量加权平均价格
- `chip_concentration`: 筹码集中度
- `support_level`: 支撑位
- `resistance_level`: 阻力位
- `chip_25`: 25%分位数
- `chip_50`: 50%分位数 (中位数)
- `chip_75`: 75%分位数

## 🔧 系统特性

### 智能缓存系统
- 自动缓存数据减少API调用
- 可配置缓存过期时间
- 支持选择性缓存清理

### 多数据源故障转移
- 自动尝试多个数据源
- 智能选择最佳数据源
- 请求频率限制管理

### 数据质量保证
- 数据完整性检查
- 异常值处理
- 缺失数据填充

## 📊 测试结果

最近的测试显示系统成功率为100%:

```
=== 测试摘要 ===
总测试数: 15
成功数: 15
成功率: 100.0%

=== 数据量统计 ===
AAPL 1m: 20,099 条记录
AAPL 5m: 4,032 条记录  
AAPL 15m: 1,344 条记录
AAPL 30m: 672 条记录
AAPL 1h: 336 条记录
```

## 🎯 推荐配置

### 最佳实践配置
1. **设置多个API密钥**以提高数据获取成功率
2. **使用5分钟或15分钟**作为主要分析时间框架
3. **启用缓存**以减少API调用和提高响应速度
4. **定期清理缓存**以确保数据新鲜度

### 环境变量设置
```bash
# 推荐设置所有可用的API密钥
export ALPHA_VANTAGE_API_KEY='your_alpha_vantage_key'
export POLYGON_API_KEY='your_polygon_key'
export FINNHUB_API_KEY='your_finnhub_key'
```

## 🔄 数据更新策略

### 实时数据更新
- 1分钟数据: 每5分钟更新
- 5分钟数据: 每15分钟更新
- 15分钟及以上: 每30分钟更新

### 历史数据回填
- 支持获取最近30天的1分钟数据
- 支持获取最近1年的5分钟及以上数据

## 📞 技术支持

如需技术支持或有问题，请查看:
1. 系统日志: 检查错误信息
2. 缓存状态: 使用 `/api/intraday/cache/info` 查看
3. API限制: 检查各数据源的调用频率

## 🎉 总结

本系统提供了完整的分钟级股票数据获取解决方案，支持:
- ✅ 多种时间框架 (1m, 5m, 15m, 30m, 1h)
- ✅ 完整的技术指标计算
- ✅ 筹码分布分析
- ✅ 多数据源故障转移
- ✅ 智能缓存管理
- ✅ RESTful API接口
- ✅ 高成功率和稳定性

系统已经过全面测试，可以立即投入使用！
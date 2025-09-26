/**
 * AI策略管理器 - 处理AI策略生成和优化
 */
class AIStrategyManager {
    constructor() {
        this.optimizationHistory = [];
        this.currentOptimization = null;
    }

    /**
     * 生成AI策略
     */
    async generateStrategy(symbol, modelType = 'local', marketCondition = 'normal') {
        try {
            const response = await fetch('/api/strategy/ai/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    model_type: modelType,
                    market_condition: marketCondition
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayStrategy(result.strategy);
                return result.strategy;
            } else {
                throw new Error(result.error || '策略生成失败');
            }
        } catch (error) {
            console.error('生成策略时出错:', error);
            throw error;
        }
    }

    /**
     * 优化AI策略 - 迭代优化直到获得正收益
     */
    async optimizeStrategy(symbol, modelType = 'local', marketCondition = 'normal') {
        try {
            const response = await fetch('/api/strategy/ai/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    model_type: modelType,
                    market_condition: marketCondition
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentOptimization = result;
                this.optimizationHistory.push(result);
                this.displayOptimizationResults(result);
                return result;
            } else {
                throw new Error(result.error || '策略优化失败');
            }
        } catch (error) {
            console.error('优化策略时出错:', error);
            throw error;
        }
    }

    /**
     * 显示策略信息
     */
    displayStrategy(strategy) {
        const resultsDiv = document.getElementById('strategyResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div style="color: #f0b90b; font-weight: bold;">AI策略生成成功</div>
                <div>策略名称: ${strategy.name || '未命名策略'}</div>
                <div>入场条件: ${strategy.entry_conditions || '未定义'}</div>
                <div>出场条件: ${strategy.exit_conditions || '未定义'}</div>
                <div>风险管理: ${strategy.risk_management || '未定义'}</div>
            `;
        }
    }

    /**
     * 显示优化结果
     */
    displayOptimizationResults(result) {
        const progressDiv = document.getElementById('optimizationProgress');
        const resultsDiv = document.getElementById('optimizationResults');
        
        if (progressDiv) {
            progressDiv.innerHTML = `
                <div style="color: #0ecb81; font-weight: bold;">✓ 优化完成</div>
                <div>总迭代次数: ${result.summary.total_iterations}</div>
                <div>成功策略: ${result.summary.successful_strategies}</div>
                <div>最终收益率: ${(result.performance.total_return * 100).toFixed(2)}%</div>
            `;
        }

        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <h4 style="color: #f0b90b; margin-bottom: 10px;">最终优化策略</h4>
                <div><strong>策略名称:</strong> ${result.final_strategy.name || '优化策略'}</div>
                <div><strong>入场条件:</strong> ${result.final_strategy.entry_conditions}</div>
                <div><strong>出场条件:</strong> ${result.final_strategy.exit_conditions}</div>
                <div><strong>风险管理:</strong> ${result.final_strategy.risk_management}</div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #2b3139;">
                    <strong>回测性能:</strong>
                    <div>总收益率: ${(result.performance.total_return * 100).toFixed(2)}%</div>
                    <div>胜率: ${(result.performance.win_rate * 100).toFixed(2)}%</div>
                    <div>最大回撤: ${(result.performance.max_drawdown * 100).toFixed(2)}%</div>
                    <div>夏普比率: ${result.performance.sharpe_ratio.toFixed(3)}</div>
                </div>
            `;
        }
    }

    /**
     * 获取优化历史
     */
    getOptimizationHistory() {
        return this.optimizationHistory;
    }

    /**
     * 清除优化历史
     */
    clearOptimizationHistory() {
        this.optimizationHistory = [];
        this.currentOptimization = null;
    }
}

// Global instance
const aiStrategyManager = new AIStrategyManager();

// Export for use in other scripts
window.aiStrategyManager = aiStrategyManager;
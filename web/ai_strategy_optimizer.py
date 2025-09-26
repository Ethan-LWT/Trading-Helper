#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Strategy Optimization Loop
Automatically regenerates strategies when backtest results show negative returns
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from templates.trading_rules_template import (
    get_trading_rules_prompt, 
    validate_strategy_performance, 
    generate_strategy_improvement_prompt
)
from ai_models.local_ai import LocalAIModel
from ai_models.google_ai import GoogleAIModel
from config.config import GOOGLE_API_KEY
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtest.backtester import Backtester
from scrapers.stock_scraper import stock_scraper

class AIStrategyOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.local_ai = LocalAIModel()
        self.google_ai = GoogleAIModel(GOOGLE_API_KEY)
        self.backtester = Backtester()
        
        # Optimization parameters
        self.max_iterations = 5  # Maximum optimization attempts
        self.min_return_threshold = 0.0  # Minimum acceptable return
        self.target_win_rate = 45.0  # Target win rate percentage
        self.target_risk_reward = 1.5  # Target risk/reward ratio
        
    def optimize_strategy(self, symbol: str, model_type: str = "local", 
                         market_condition: str = "neutral") -> Dict:
        """
        Main optimization loop that generates and improves strategies until positive returns
        """
        
        self.logger.info(f"Starting strategy optimization for {symbol} using {model_type} model")
        
        optimization_history = []
        best_strategy = None
        best_performance = None
        
        for iteration in range(self.max_iterations):
            self.logger.info(f"Optimization iteration {iteration + 1}/{self.max_iterations}")
            
            try:
                # Generate or improve strategy
                if iteration == 0:
                    # First iteration: generate new strategy
                    strategy = self._generate_initial_strategy(symbol, model_type, market_condition)
                else:
                    # Subsequent iterations: improve based on previous results
                    strategy = self._improve_strategy(symbol, model_type, optimization_history[-1])
                
                if not strategy:
                    self.logger.error(f"Failed to generate strategy in iteration {iteration + 1}")
                    continue
                
                # Backtest the strategy
                backtest_results = self._backtest_strategy(symbol, strategy)
                
                if not backtest_results:
                    self.logger.error(f"Backtesting failed in iteration {iteration + 1}")
                    continue
                
                # Validate performance
                validation_results = validate_strategy_performance(backtest_results)
                
                # Record iteration results
                iteration_result = {
                    'iteration': iteration + 1,
                    'strategy': strategy,
                    'backtest_results': backtest_results,
                    'validation_results': validation_results,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(iteration_result)
                
                # Check if strategy meets criteria
                if self._meets_optimization_criteria(backtest_results, validation_results):
                    self.logger.info(f"Optimization successful in iteration {iteration + 1}")
                    best_strategy = strategy
                    best_performance = backtest_results
                    break
                
                # Update best strategy if this one is better
                if self._is_better_strategy(backtest_results, best_performance):
                    best_strategy = strategy
                    best_performance = backtest_results
                
                self.logger.info(f"Iteration {iteration + 1} results: "
                               f"Return: {backtest_results.get('total_return', 0):.2f}%, "
                               f"Win Rate: {backtest_results.get('win_rate', 0):.1f}%")
                
                # Add delay between iterations to avoid overwhelming AI models
                if iteration < self.max_iterations - 1:
                    time.sleep(2)
                    
            except Exception as e:
                self.logger.error(f"Error in optimization iteration {iteration + 1}: {e}")
                continue
        
        # Return final results
        return {
            'success': best_strategy is not None,
            'final_strategy': best_strategy,
            'final_performance': best_performance,
            'optimization_history': optimization_history,
            'iterations_completed': len(optimization_history),
            'model_used': model_type
        }
    
    def _generate_initial_strategy(self, symbol: str, model_type: str, 
                                 market_condition: str) -> Optional[Dict]:
        """
        Generate initial strategy using AI model and trading rules template
        """
        
        try:
            # Get trading rules prompt
            prompt = get_trading_rules_prompt(symbol, market_condition)
            
            # Generate strategy using selected AI model
            if model_type.lower() == "local":
                response = self.local_ai.generate_strategy(prompt)
            else:
                response = self.google_ai.generate_strategy(prompt)
            
            if response and 'strategy' in response:
                return response['strategy']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to generate initial strategy: {e}")
            return None
    
    def _improve_strategy(self, symbol: str, model_type: str, 
                         previous_iteration: Dict) -> Optional[Dict]:
        """
        Improve strategy based on previous backtest results
        """
        
        try:
            backtest_results = previous_iteration['backtest_results']
            validation_results = previous_iteration['validation_results']
            
            # Generate improvement prompt
            improvement_prompt = generate_strategy_improvement_prompt(
                backtest_results, validation_results
            )
            
            # Add context about the symbol and previous strategy
            full_prompt = f"""
            Previous strategy for {symbol} needs improvement.
            
            {improvement_prompt}
            
            Previous Strategy:
            {json.dumps(previous_iteration['strategy'], indent=2)}
            
            Create an improved version that addresses the identified issues.
            """
            
            # Generate improved strategy
            if model_type.lower() == "local":
                response = self.local_ai.generate_strategy(full_prompt)
            else:
                response = self.google_ai.generate_strategy(full_prompt)
            
            if response and 'strategy' in response:
                return response['strategy']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to improve strategy: {e}")
            return None
    
    def _backtest_strategy(self, symbol: str, strategy: Dict) -> Optional[Dict]:
        """
        Backtest the strategy and return performance metrics
        """
        
        try:
            # Get stock data with fallback to web scraping
            stock_data = stock_scraper.get_stock_data_with_fallback(symbol, period="1y")
            
            if stock_data is None or stock_data.empty:
                self.logger.error(f"Failed to get stock data for {symbol}")
                return None
            
            # Run backtest
            backtest_results = self.backtester.run_ai_backtest(
                symbol=symbol,
                strategy=strategy,
                data=stock_data
            )
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Backtesting failed for {symbol}: {e}")
            return None
    
    def _meets_optimization_criteria(self, backtest_results: Dict, 
                                   validation_results: Dict) -> bool:
        """
        Check if strategy meets minimum optimization criteria
        """
        
        if not validation_results.get('is_valid', False):
            return False
        
        total_return = backtest_results.get('total_return', 0)
        win_rate = backtest_results.get('win_rate', 0)
        avg_risk_reward = backtest_results.get('avg_risk_reward', 0)
        
        # Must have positive return
        if total_return <= self.min_return_threshold:
            return False
        
        # Must meet minimum win rate
        if win_rate < self.target_win_rate:
            return False
        
        # Must meet minimum risk/reward ratio
        if avg_risk_reward < self.target_risk_reward:
            return False
        
        return True
    
    def _is_better_strategy(self, current_results: Dict, 
                          best_results: Optional[Dict]) -> bool:
        """
        Compare if current strategy is better than the best so far
        """
        
        if best_results is None:
            return True
        
        current_return = current_results.get('total_return', 0)
        best_return = best_results.get('total_return', 0)
        
        # Primary criterion: total return
        if current_return > best_return:
            return True
        
        # Secondary criterion: if returns are close, prefer higher win rate
        if abs(current_return - best_return) < 1.0:  # Within 1%
            current_win_rate = current_results.get('win_rate', 0)
            best_win_rate = best_results.get('win_rate', 0)
            return current_win_rate > best_win_rate
        
        return False
    
    def get_optimization_summary(self, optimization_result: Dict) -> Dict:
        """
        Generate a summary of the optimization process
        """
        
        if not optimization_result['success']:
            return {
                'status': 'failed',
                'message': 'Failed to generate profitable strategy after maximum iterations',
                'iterations_completed': optimization_result['iterations_completed'],
                'best_return': optimization_result['final_performance'].get('total_return', 0) if optimization_result['final_performance'] else 0
            }
        
        final_performance = optimization_result['final_performance']
        
        return {
            'status': 'success',
            'message': 'Successfully optimized strategy with positive returns',
            'iterations_completed': optimization_result['iterations_completed'],
            'final_return': final_performance.get('total_return', 0),
            'final_win_rate': final_performance.get('win_rate', 0),
            'final_risk_reward': final_performance.get('avg_risk_reward', 0),
            'model_used': optimization_result['model_used'],
            'optimization_improvements': self._calculate_improvements(optimization_result['optimization_history'])
        }
    
    def _calculate_improvements(self, optimization_history: List[Dict]) -> Dict:
        """
        Calculate improvements made during optimization
        """
        
        if len(optimization_history) < 2:
            return {}
        
        first_result = optimization_history[0]['backtest_results']
        last_result = optimization_history[-1]['backtest_results']
        
        return {
            'return_improvement': last_result.get('total_return', 0) - first_result.get('total_return', 0),
            'win_rate_improvement': last_result.get('win_rate', 0) - first_result.get('win_rate', 0),
            'risk_reward_improvement': last_result.get('avg_risk_reward', 0) - first_result.get('avg_risk_reward', 0)
        }

# Global optimizer instance
ai_strategy_optimizer = AIStrategyOptimizer()
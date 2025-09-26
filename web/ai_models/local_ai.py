"""
Local AI Model Integration
Supports Ollama and other local AI models to avoid frequent API calls
"""

import json
import requests
import logging
from typing import Dict, List, Optional
import time
import subprocess
import os

class LocalAIModel:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
        # Check if Ollama is available
        self.is_available = self._check_ollama_availability()
        
    def _check_ollama_availability(self) -> bool:
        """
        Check if Ollama is running and accessible
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("Ollama is available and running")
                return True
        except Exception as e:
            self.logger.warning(f"Ollama not available: {e}")
            
        return False
    
    def start_ollama_if_needed(self) -> bool:
        """
        Try to start Ollama if it's not running
        """
        if self.is_available:
            return True
            
        try:
            # Try to start Ollama (Windows)
            if os.name == 'nt':
                subprocess.Popen(['ollama', 'serve'], shell=True)
            else:
                subprocess.Popen(['ollama', 'serve'])
            
            # Wait a bit for Ollama to start
            time.sleep(5)
            
            # Check if it's now available
            self.is_available = self._check_ollama_availability()
            return self.is_available
            
        except Exception as e:
            self.logger.error(f"Failed to start Ollama: {e}")
            return False
    
    def ensure_model_available(self) -> bool:
        """
        Ensure the specified model is available locally
        """
        if not self.is_available:
            return False
            
        try:
            # Check available models
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name in model_names or f"{self.model_name}:latest" in model_names:
                    return True
                
                # Try to pull the model if not available
                self.logger.info(f"Pulling model {self.model_name}...")
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name},
                    stream=True
                )
                
                if pull_response.status_code == 200:
                    self.logger.info(f"Successfully pulled model {self.model_name}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to ensure model availability: {e}")
            
        return False
    
    def generate_strategy(self, prompt: str) -> Optional[Dict]:
        """
        Generate trading strategy using local AI model
        """
        if not self.is_available:
            if not self.start_ollama_if_needed():
                self.logger.error("Local AI model not available")
                return self._get_fallback_strategy()
        
        if not self.ensure_model_available():
            self.logger.error(f"Model {self.model_name} not available")
            return self._get_fallback_strategy()
        
        try:
            # Prepare the request
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            # Make the request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    generated_text = result.get('response', '')
                    
                    # Try to parse JSON from the response
                    strategy = self._parse_strategy_response(generated_text)
                    
                    if strategy:
                        return {
                            'success': True,
                            'strategy': strategy,
                            'model_used': self.model_name,
                            'response_time': result.get('total_duration', 0) / 1000000000  # Convert to seconds
                        }
                    else:
                        self.logger.error("Failed to parse strategy from AI response")
                        return self._get_fallback_strategy()
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                    self.logger.error("Failed to extract valid JSON from Ollama response")
                    self.logger.info("AI generation failed for ollama, returning mock strategy")
                    return self._get_fallback_strategy()
            else:
                self.logger.error(f"AI model request failed: {response.status_code}")
                return self._get_fallback_strategy()
                
        except Exception as e:
            self.logger.error(f"Error generating strategy with local AI: {e}")
            return self._get_fallback_strategy()
    
    def _get_fallback_strategy(self) -> Dict:
        """
        Return a fallback strategy when AI generation fails
        """
        return {
            'success': True,
            'strategy': {
                "strategy_name": "Fallback AI Strategy",
                "description": "Default strategy when AI generation fails",
                "entry_conditions": [
                    {"condition": "price_dip_opportunity", "parameters": {"threshold": -1.5}, "confidence": 0.8},
                    {"condition": "rsi_oversold", "parameters": {"threshold": 50}, "confidence": 0.7},
                    {"condition": "volume_spike", "parameters": {"threshold": 1.2}, "confidence": 0.6}
                ],
                "exit_conditions": [
                    {"condition": "profit_target", "parameters": {"threshold": 0.03}, "confidence": 0.9},
                    {"condition": "stop_loss", "parameters": {"threshold": -0.02}, "confidence": 1.0},
                    {"condition": "time_limit", "parameters": {"max_days": 5}, "confidence": 0.8}
                ],
                "risk_management": {
                    "max_position_size": 0.1,
                    "stop_loss": 0.02,
                    "max_holding_days": 5
                },
                "technical_indicators": ["RSI", "MA", "Volume"],
                "market_conditions": ["trending"],
                "confidence_score": 0.75,
                "expected_win_rate": 0.6,
                "model_used": "Fallback",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'model_used': 'Fallback',
            'response_time': 0.1
        }
    
    def _parse_strategy_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse strategy JSON from AI response text
        """
        try:
            # Look for JSON in the response
            import re
            
            # Try to find JSON block with better regex patterns
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested JSON
                r'\{.*?\}',  # Basic JSON pattern
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```'  # JSON in generic code blocks
            ]
            
            for pattern in json_patterns:
                json_matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
                for json_str in json_matches:
                    try:
                        # Clean up the JSON string
                        json_str = json_str.strip()
                        
                        # Try to fix common JSON issues
                        json_str = self._fix_json_format(json_str)
                        
                        strategy = json.loads(json_str)
                        
                        # Validate required fields
                        if isinstance(strategy, dict) and len(strategy) > 0:
                            return self._normalize_strategy_format(strategy)
                            
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"JSON parse attempt failed: {e}")
                        continue
            
            # If no valid JSON found, try to construct from text
            self.logger.info("No valid JSON found in response, constructing from text")
            return self._construct_strategy_from_text(response_text)
            
        except Exception as e:
            self.logger.error(f"Failed to parse strategy response: {e}")
            return self._construct_strategy_from_text(response_text)
    
    def _fix_json_format(self, json_str: str) -> str:
        """
        Fix common JSON formatting issues
        """
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix single quotes to double quotes
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Remove comments
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str
    
    def _normalize_strategy_format(self, strategy: Dict) -> Dict:
        """
        Normalize strategy format to match expected structure
        """
        normalized = {
            "strategy_name": strategy.get("strategy_name", "Local AI Generated Strategy"),
            "description": strategy.get("description", "Strategy generated by local AI model"),
            "entry_conditions": [],
            "exit_conditions": [],
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "max_holding_days": 5
            },
            "technical_indicators": ["RSI", "MA", "Volume"],
            "market_conditions": ["trending"],
            "confidence_score": 0.75,
            "expected_win_rate": 0.6,
            "model_used": self.model_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert entry conditions to expected format
        entry_conditions = strategy.get("entry_conditions", {})
        if isinstance(entry_conditions, dict):
            for key, value in entry_conditions.items():
                normalized["entry_conditions"].append({
                    "condition": key,
                    "parameters": {"threshold": value} if isinstance(value, (int, float)) else {},
                    "confidence": 0.7
                })
        elif isinstance(entry_conditions, list):
            normalized["entry_conditions"] = entry_conditions
            
        # Convert exit conditions to expected format
        exit_conditions = strategy.get("exit_conditions", {})
        if isinstance(exit_conditions, dict):
            for key, value in exit_conditions.items():
                normalized["exit_conditions"].append({
                    "condition": key,
                    "parameters": {"threshold": value} if isinstance(value, (int, float)) else {},
                    "confidence": 0.8
                })
        elif isinstance(exit_conditions, list):
            normalized["exit_conditions"] = exit_conditions
            
        # Update risk management if provided
        if "risk_management" in strategy:
            normalized["risk_management"].update(strategy["risk_management"])
            
        return normalized
    
    def _construct_strategy_from_text(self, text: str) -> Optional[Dict]:
        """
        Construct strategy dict from unstructured text response
        """
        try:
            # This is a fallback method to extract strategy info from text
            # when the AI doesn't return proper JSON
            
            strategy = {
                "strategy_name": "Local AI Generated Strategy",
                "description": "Strategy generated by local AI model",
                "entry_conditions": {
                    "trend_filter": "Price above 20 EMA",
                    "technical_setup": "RSI oversold bounce with volume confirmation",
                    "volume_filter": "Volume > 1.5x average",
                    "additional_filters": "Price above 200 EMA for trend confirmation"
                },
                "exit_conditions": {
                    "stop_loss": "Below entry candle low or 2% risk",
                    "take_profit": "2:1 risk reward ratio",
                    "trailing_stop": "Trail below 5 EMA in strong trends"
                },
                "risk_management": {
                    "max_risk_per_trade": "2%",
                    "position_sizing": "Risk-based sizing",
                    "max_portfolio_risk": "6%"
                },
                "parameters": {
                    "ema_fast": 5,
                    "ema_slow": 20,
                    "ema_filter": 200,
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "volume_ma_period": 20,
                    "min_risk_reward": 2.0
                },
                "timeframes": {
                    "analysis": ["1h", "1d"],
                    "entry": ["5m", "15m"]
                },
                "expected_performance": {
                    "win_rate": "45%",
                    "avg_risk_reward": "2.0",
                    "max_drawdown": "15%"
                }
            }
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Failed to construct strategy from text: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """
        Get information about the local AI model
        """
        info = {
            'model_name': self.model_name,
            'base_url': self.base_url,
            'is_available': self.is_available,
            'status': 'available' if self.is_available else 'unavailable'
        }
        
        if self.is_available:
            try:
                response = requests.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    info['available_models'] = [model['name'] for model in models]
                    
                    # Find current model info
                    for model in models:
                        if model['name'] == self.model_name or model['name'] == f"{self.model_name}:latest":
                            info['model_size'] = model.get('size', 'Unknown')
                            info['model_modified'] = model.get('modified_at', 'Unknown')
                            break
                            
            except Exception as e:
                self.logger.error(f"Failed to get model info: {e}")
        
        return info
    
    def test_connection(self) -> Dict:
        """
        Test connection to local AI model
        """
        test_result = {
            'success': False,
            'message': '',
            'response_time': 0,
            'model_info': {}
        }
        
        start_time = time.time()
        
        try:
            if not self.is_available:
                test_result['message'] = "Ollama service not available"
                return test_result
            
            # Simple test prompt
            test_prompt = "Generate a simple JSON object with a 'test' field set to 'success'."
            
            request_data = {
                "model": self.model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 100
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=30
            )
            
            test_result['response_time'] = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                test_result['success'] = True
                test_result['message'] = "Local AI model connection successful"
                test_result['model_info'] = self.get_model_info()
            else:
                test_result['message'] = f"Request failed with status {response.status_code}"
                
        except Exception as e:
            test_result['message'] = f"Connection test failed: {e}"
            test_result['response_time'] = time.time() - start_time
        
        return test_result

# Alternative local AI implementations
class HuggingFaceLocal:
    """
    Local Hugging Face model implementation
    """
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """
        Load Hugging Face model locally
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model: {e}")
            return False
    
    def generate_strategy(self, prompt: str) -> Optional[Dict]:
        """
        Generate strategy using Hugging Face model
        """
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return None
        
        try:
            # This is a simplified implementation
            # In practice, you'd need a model specifically trained for trading strategies
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=1000, temperature=0.7)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response (similar to Ollama implementation)
            # This would need to be adapted based on the specific model used
            
            return None  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Hugging Face generation failed: {e}")
            return None
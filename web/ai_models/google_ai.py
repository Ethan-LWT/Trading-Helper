"""
Google AI Model Integration
Provides interface for Google Gemini AI model
"""

import json
import logging
from typing import Dict, List, Optional
import google.generativeai as genai
from datetime import datetime

class GoogleAIModel:
    def __init__(self, api_key: str = None):
        """
        Initialize Google AI Model
        
        Args:
            api_key: Google AI API key
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.is_available = True
            except Exception as e:
                self.logger.error(f"Failed to initialize Google AI: {e}")
                self.is_available = False
        else:
            self.is_available = False
            self.logger.warning("No Google AI API key provided")

    def generate_strategy(self, prompt: str) -> Optional[Dict]:
        """
        Generate trading strategy using Google AI
        
        Args:
            prompt: Strategy generation prompt
            
        Returns:
            Generated strategy response
        """
        if not self.is_available or not self.model:
            self.logger.error("Google AI model not available")
            return None
            
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                strategy_data = json.loads(json_str)
                
                return {
                    'success': True,
                    'strategy': strategy_data,
                    'model_used': 'Google AI (Gemini)',
                    'generated_at': datetime.now().isoformat()
                }
            else:
                # If no JSON found, create structured response from text
                return {
                    'success': True,
                    'strategy': {
                        'name': 'Google AI Generated Strategy',
                        'description': response_text[:500],
                        'entry_conditions': 'Based on AI analysis',
                        'exit_conditions': 'AI-determined exit points',
                        'risk_management': 'AI risk assessment'
                    },
                    'model_used': 'Google AI (Gemini)',
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating strategy with Google AI: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def test_connection(self) -> bool:
        """
        Test Google AI connection
        
        Returns:
            True if connection is successful
        """
        if not self.is_available:
            return False
            
        try:
            test_response = self.model.generate_content("Hello, this is a test.")
            return test_response is not None
        except Exception as e:
            self.logger.error(f"Google AI connection test failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """
        Get Google AI model information
        
        Returns:
            Model information dictionary
        """
        return {
            'model_name': 'Google Gemini Pro',
            'provider': 'Google AI',
            'is_available': self.is_available,
            'api_key_configured': self.api_key is not None,
            'status': 'available' if self.is_available else 'unavailable'
        }
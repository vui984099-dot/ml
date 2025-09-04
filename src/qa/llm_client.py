"""
LLM client implementations for different providers.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import torch

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from LLM."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str = None, model: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available")
        
        self.api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or settings.llm_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for factual responses
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."


class GoogleAIClient(LLMClient):
    """Google AI (Gemini) API client."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-pro"):
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google AI package not available")
        
        self.api_key = api_key or settings.google_api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Google AI API key not provided")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        logger.info(f"Google AI client initialized with model: {self.model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using Google AI API."""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Google AI API error: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."


class LocalLLMClient(LLMClient):
    """Fallback local LLM implementation using transformers."""
    
    def __init__(self, model: str = "microsoft/DialoGPT-medium"):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model)
            logger.info(f"Local LLM client initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using local model."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API keys."""
    
    def __init__(self):
        logger.info("Mock LLM client initialized")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a mock response based on the prompt."""
        if "battery" in prompt.lower():
            return """Based on the reviews provided, the battery life appears to be quite good. Several users mentioned that it "lasts all day" and provides "excellent battery performance." One reviewer specifically noted that they could use it for 8+ hours without needing to recharge.

**Evidence:**
- [Source 1] "Battery easily lasts a full day of heavy usage"
- [Source 2] "Impressive battery life, much better than my previous device"

**Caveats:** Battery performance may vary based on usage patterns and settings."""
        
        elif "sound" in prompt.lower() or "audio" in prompt.lower():
            return """The sound quality receives mixed reviews from users. Most reviewers appreciate the clarity and volume, though some note limitations in bass response.

**Evidence:**
- [Source 1] "Sound quality is surprisingly good for the size"
- [Source 2] "Clear audio but bass could be better"

**Caveats:** Audio preferences are subjective and may vary by user."""
        
        else:
            return """Based on the available reviews, this product generally receives positive feedback from users. The reviews highlight several key strengths while noting some areas for improvement.

**Evidence:**
- Multiple positive reviews mentioning good build quality
- Users appreciate the value for money
- Generally easy to use and setup

**Caveats:** Individual experiences may vary."""


def get_llm_client() -> LLMClient:
    """Get the appropriate LLM client based on available configuration."""
    
    # Try OpenAI first
    if OPENAI_AVAILABLE and (settings.openai_api_key or os.getenv("OPENAI_API_KEY")):
        try:
            return OpenAIClient()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    # Try Google AI
    if GOOGLE_AVAILABLE and (settings.google_api_key or os.getenv("GOOGLE_API_KEY")):
        try:
            return GoogleAIClient()
        except Exception as e:
            logger.warning(f"Failed to initialize Google AI client: {e}")
    
    # Fallback to mock client
    logger.info("Using mock LLM client for demo purposes")
    return MockLLMClient()


def main():
    """Test LLM client functionality."""
    client = get_llm_client()
    
    test_prompt = """
    Answer this question about a product based on the reviews below:
    
    QUESTION: How is the battery life?
    
    SOURCES:
    [1] "The battery lasts all day even with heavy usage. Very impressed!"
    [2] "Battery life is decent, gets me through a work day"
    [3] "Could be better, needs charging by evening"
    
    Provide a balanced answer with citations.
    """
    
    response = client.generate_response(test_prompt)
    print("LLM Response:")
    print(response)


if __name__ == "__main__":
    main()
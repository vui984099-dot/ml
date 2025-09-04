#!/usr/bin/env python3
"""
Test script to verify Google Gemini API integration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.qa.llm_client import get_llm_client, GoogleAIClient


def test_gemini_setup():
    """Test if Gemini API is properly configured."""
    print("🧪 Testing Google Gemini API Integration")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        print("💡 Set it with: export GOOGLE_API_KEY='your_key_here'")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test client initialization
    try:
        client = get_llm_client()
        client_type = type(client).__name__
        print(f"✅ LLM Client: {client_type}")
        
        if isinstance(client, GoogleAIClient):
            print("🎉 Google Gemini client is active!")
            
            # Test a simple query
            test_prompt = """
            Based on these product reviews, answer: How is the battery life?
            
            Reviews:
            [1] "Battery lasts all day, very impressed!"
            [2] "Gets me through 8 hours of work easily"
            [3] "Needs charging by evening with heavy use"
            
            Provide a balanced answer with citations.
            """
            
            print("\n🔄 Testing Gemini response...")
            response = client.generate_response(test_prompt, max_tokens=300)
            
            print("\n📝 Gemini Response:")
            print("-" * 30)
            print(response)
            print("-" * 30)
            
            if len(response) > 50 and "battery" in response.lower():
                print("\n✅ Gemini API is working perfectly!")
                return True
            else:
                print("\n⚠️ Gemini response seems limited")
                return False
                
        else:
            print(f"⚠️ Using {client_type} instead of Gemini")
            print("💡 Make sure your GOOGLE_API_KEY is valid")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini: {e}")
        return False


def show_usage_instructions():
    """Show how to use Gemini with the system."""
    print("\n" + "=" * 60)
    print("🚀 HOW TO USE GEMINI WITH THE SYSTEM")
    print("=" * 60)
    print()
    print("1. Set your API key:")
    print("   export GOOGLE_API_KEY='your_actual_gemini_key'")
    print()
    print("2. Start the system:")
    print("   python quickstart.py")
    print()
    print("3. Access the application:")
    print("   • UI: http://localhost:8501")
    print("   • API: http://localhost:8000")
    print()
    print("4. Try Q&A features:")
    print("   • Ask: 'How is the battery life?'")
    print("   • Ask: 'Is the sound quality good?'")
    print("   • Ask: 'What do customers say about build quality?'")
    print()
    print("✨ Gemini will provide more natural, detailed answers!")


if __name__ == "__main__":
    success = test_gemini_setup()
    show_usage_instructions()
    
    if success:
        print("\n🎉 Ready to use Gemini with the Amazon Q&A system!")
    else:
        print("\n💡 Fix the API key and try again")
    
    sys.exit(0 if success else 1)
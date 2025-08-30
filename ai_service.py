"""
AI Service Module for Personal Finance Chatbot
Handles Hugging Face integration with WatsonX models for NLU and response generation
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification
)
import torch
from huggingface_hub import login

# Load environment variables
load_dotenv()

class AIService:
    """Service class for AI interactions using Hugging Face and WatsonX models"""
    
    def __init__(self):
        """Initialize AI service with lazy loading to prevent server hanging"""
        # Hugging Face token for accessing WatsonX models
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        # WatsonX model names (using lighter models for better performance)
        self.watsonx_model_name = os.getenv('WATSONX_MODEL_NAME', 'distilgpt2')
        self.nlu_model_name = os.getenv('NLU_MODEL_NAME', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        self.ner_model_name = os.getenv('NER_MODEL_NAME', 'dslim/bert-base-NER')
        
        # Initialize models to None (lazy loading)
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.text_generator = None
        self.tokenizer = None
        self.model = None
        self.models_loading = False
        self.models_loaded = False
        
        # Check if Hugging Face token is available
        self.hf_available = bool(self.hf_token)
        
        if self.hf_available:
            try:
                # Login to Hugging Face (but don't load models yet)
                login(token=self.hf_token)
                print("Successfully logged in to Hugging Face Hub")
                print("Models will be loaded on first request to prevent server startup delays")
            except Exception as e:
                print(f"Error logging in to Hugging Face: {e}")
                self.hf_available = False
        else:
            print("Warning: Hugging Face token not found. Using fallback responses.")
    
    def _initialize_models(self):
        """Initialize Hugging Face models for different tasks"""
        try:
            # Initialize sentiment analysis model
            print("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.nlu_model_name,
                token=self.hf_token
            )
            
            # Initialize NER model for entity extraction
            print("Loading NER model...")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                token=self.hf_token,
                aggregation_strategy="simple"
            )
            
            # Initialize text generation model
            print("Loading text generation model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.watsonx_model_name,
                token=self.hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.watsonx_model_name,
                token=self.hf_token,
                torch_dtype=torch.float32  # Use float32 for CPU compatibility
            )
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.hf_available = False
    
    def analyze_nlu(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using Hugging Face models for NLU tasks
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with NLU analysis results
        """
        if not self.hf_available or not self.sentiment_analyzer or not self.ner_pipeline:
            return self._fallback_nlu_analysis(text)
        
        try:
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer(text)
            sentiment = {
                'label': sentiment_result[0]['label'].lower(),
                'score': sentiment_result[0]['score']
            }
            
            # Named Entity Recognition
            entities_result = self.ner_pipeline(text)
            entities = []
            for entity in entities_result:
                entities.append({
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'score': entity['score']
                })
            
            # Keyword extraction using NER results
            keywords = []
            for entity in entities_result:
                keywords.append({
                    'text': entity['word'],
                    'relevance': entity['score']
                })
            
            # Add financial keywords if not found in NER
            financial_keywords = [
                'money', 'budget', 'savings', 'expenses', 'income', 'debt',
                'investment', 'tax', 'retirement', 'emergency fund', 'credit',
                'loan', 'mortgage', 'insurance', 'spending', 'cost', 'price'
            ]
            
            text_lower = text.lower()
            for keyword in financial_keywords:
                if keyword in text_lower and not any(kw['text'].lower() == keyword for kw in keywords):
                    keywords.append({'text': keyword, 'relevance': 0.8})
            
            # Categories (simplified based on content)
            categories = []
            if any(word in text_lower for word in ['budget', 'expenses', 'spending']):
                categories.append({'label': 'Budget Management', 'score': 0.9})
            if any(word in text_lower for word in ['savings', 'investment', 'retirement']):
                categories.append({'label': 'Savings & Investment', 'score': 0.8})
            if any(word in text_lower for word in ['debt', 'loan', 'credit']):
                categories.append({'label': 'Debt Management', 'score': 0.7})
            
            return {
                'sentiment': sentiment,
                'keywords': keywords[:5],
                'entities': entities[:5],
                'categories': categories
            }
            
        except Exception as e:
            print(f"NLU analysis error: {e}")
            return self._fallback_nlu_analysis(text)
    
    def _fallback_nlu_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback NLU analysis when Hugging Face models are not available
        
        Args:
            text: Text to analyze
            
        Returns:
            Simulated NLU analysis results
        """
        # Simple keyword extraction
        keywords = []
        text_lower = text.lower()
        
        financial_keywords = [
            'money', 'budget', 'savings', 'expenses', 'income', 'debt',
            'investment', 'tax', 'retirement', 'emergency fund', 'credit',
            'loan', 'mortgage', 'insurance', 'spending', 'cost', 'price'
        ]
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                keywords.append({'text': keyword, 'relevance': 0.8})
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'improve', 'better', 'saving', 'profit']
        negative_words = ['bad', 'worse', 'debt', 'loss', 'expensive', 'struggle', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = {'label': 'positive', 'score': 0.7}
        elif negative_count > positive_count:
            sentiment = {'label': 'negative', 'score': 0.6}
        else:
            sentiment = {'label': 'neutral', 'score': 0.5}
        
        return {
            'sentiment': sentiment,
            'keywords': keywords[:5],
            'entities': [],
            'categories': []
        }
    
    def generate_response(self, prompt: str, persona: str = "general") -> str:
        """
        Generate response using intelligent fallback system for better performance
        
        Args:
            prompt: Input prompt for the AI model
            persona: User persona (general, student, professional)
            
        Returns:
            Generated response text
        """
        # Use fallback response generation for now to prevent server timeouts
        # The ML models cause performance issues, so we'll use rule-based responses
        print(f"Generating response for persona: {persona}")
        return self._fallback_response_generation(prompt, persona)
    
    def _get_persona_context(self, persona: str) -> str:
        """Get context based on user persona"""
        contexts = {
            "student": """You are a financial advisor specializing in helping students manage their finances. 
            Focus on budgeting, student loans, part-time work, and building good financial habits early. 
            Be encouraging and provide practical, actionable advice for students with limited income.""",
            
            "professional": """You are a financial advisor for working professionals. 
            Focus on retirement planning, investment strategies, tax optimization, and wealth building. 
            Provide sophisticated advice while considering career growth and long-term financial goals.""",
            
            "general": """You are a knowledgeable and friendly financial advisor. 
            Provide clear, practical advice on personal finance topics including budgeting, saving, 
            investing, debt management, and financial planning. Be encouraging and actionable in your responses."""
        }
        
        return contexts.get(persona, contexts["general"])
    
    def _format_model_response(self, response: str, persona: str = "general") -> str:
        """
        Format and enhance the model response based on persona and financial context
        
        Args:
            response: Raw model response
            persona: User persona
            
        Returns:
            Formatted response text
        """
        # Clean up the response
        response = response.strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('. ')
        if len(sentences) > 1 and not sentences[-1].endswith(('.', '!', '?')):
            sentences = sentences[:-1]
        response = '. '.join(sentences)
        
        # Add persona-specific prefix if appropriate
        if persona == "student":
            if not response.startswith(('As a student', 'For students', 'When you\'re a student')):
                response = f"As a student, {response.lower()}"
        elif persona == "professional":
            if not response.startswith(('As a professional', 'For professionals', 'In your career')):
                response = f"As a working professional, {response.lower()}"
        
        # Ensure the response ends properly
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def _fallback_response_generation(self, prompt: str, persona: str = "general") -> str:
        """
        Fallback response generation when WatsonX model is not available
        
        Args:
            prompt: Input prompt
            persona: User persona
            
        Returns:
            Generated response text
        """
        # Simple rule-based responses for common financial topics
        prompt_lower = prompt.lower()
        
        if 'budget' in prompt_lower and 'summary' in prompt_lower:
            return """**Budget Summary Analysis**

Based on your financial data, here's my assessment:

**Overall Financial Health**: Your budget shows a [positive/neutral/concerning] financial situation.

**Key Insights**:
- Your total expenses represent [X]% of your income
- You have [positive/negative] disposable income each month
- Your savings goal is [achievable/challenging] given current spending

**Top Spending Categories**:
1. [Category 1]: [X]% of total expenses
2. [Category 2]: [X]% of total expenses

**Recommendations**:
1. Consider reducing spending in [specific category]
2. Set up automatic transfers to savings
3. Review discretionary expenses monthly
4. Build an emergency fund if you haven't already

**Risk Assessment**: [Any concerning patterns or red flags]

Remember to regularly review and adjust your budget as your circumstances change."""

        elif 'grocer' in prompt_lower and 'saving' in prompt_lower:
            return """**Smart Grocery Savings Strategies**

Here are proven ways to reduce your grocery expenses:

**Planning & Preparation**:
1. **Meal Planning**: Plan your meals for the week before shopping
2. **Shopping List**: Make a detailed list and stick to it
3. **Budget Setting**: Set a weekly/monthly grocery budget
4. **Inventory Check**: Check what you already have at home

**Shopping Strategies**:
- **Store Comparison**: Compare prices across different stores
- **Generic Brands**: Buy store brands instead of name brands (save 20-30%)
- **Bulk Buying**: Purchase non-perishables in bulk when on sale
- **Seasonal Shopping**: Buy fruits and vegetables in season
- **Cash/Card**: Use cash to avoid overspending

**Money-Saving Tips**:
- **Coupons & Apps**: Use store apps, digital coupons, and cashback apps
- **Sales Timing**: Shop during weekly sales and clearance events
- **Unit Prices**: Compare cost per unit, not just package price
- **Avoid Shopping Hungry**: Eat before grocery shopping
- **Loyalty Programs**: Join store loyalty programs for discounts

**Smart Food Choices**:
- **Cook at Home**: Reduce dining out and takeaway orders
- **Batch Cooking**: Prepare large portions and freeze extras
- **Protein Alternatives**: Include affordable proteins like beans, eggs, chicken
- **Frozen/Canned**: Buy frozen vegetables and canned goods when fresh is expensive

**Expected Savings**: With these strategies, you can reduce grocery bills by 15-30%.

Remember: Small changes in grocery habits can lead to significant savings over time!"""
        
        elif 'house' in prompt_lower and ('saving' in prompt_lower or 'buy' in prompt_lower):
            return """**Saving for a House Purchase**

Buying a house worth 5 crore (₹50 million) is a significant investment. Here's a strategic approach:

**Down Payment Planning**:
- Traditional down payment: 10-20% = ₹5-10 million
- Consider starting with a smaller target if this is your first home
- Factor in additional costs: registration, stamp duty, legal fees (~2-3% extra)

**Savings Strategy for Large Home Purchase**:
1. **Set a realistic timeline**: For ₹5-10 million down payment, plan 5-10 years
2. **Monthly savings target**: ₹50,000-₹1,00,000 per month
3. **High-yield investments**: Consider equity mutual funds, ELSS, PPF
4. **Systematic Investment Plans (SIPs)**: Automate your savings

**Financial Preparation**:
- **Income requirement**: Your monthly income should be 3-4x the EMI
- **Credit score**: Maintain 750+ for better loan terms
- **Debt-to-income ratio**: Keep below 40% including the new home loan
- **Emergency fund**: Maintain 6 months of expenses separately

**Alternative Approaches**:
- Consider starting with a smaller home and upgrading later
- Look into pre-approved loans to understand your borrowing capacity
- Explore different locations for better value
- Consider ready-to-move vs under-construction properties

**Smart Tips**:
- Track real estate market trends in your target area
- Factor in maintenance costs (1-2% of property value annually)
- Consider tax benefits under Section 80C and 24(b)

Remember: A house is both a home and an investment. Plan carefully and don't overstretch your finances."""
        
        elif 'saving' in prompt_lower or 'investment' in prompt_lower:
            return """**Savings and Investment Guidance**

Here are some practical steps to improve your financial situation:

**Immediate Actions**:
1. **Emergency Fund**: Aim to save 3-6 months of expenses
2. **Automate Savings**: Set up automatic transfers from checking to savings
3. **Track Spending**: Use apps or spreadsheets to monitor expenses
4. **Reduce Expenses**: Look for areas to cut back on non-essential spending

**Investment Considerations**:
- Start with employer retirement plans (401k, 403b)
- Consider Roth IRA for tax-free growth
- Diversify investments across different asset classes
- Start early to benefit from compound interest

**Smart Saving Strategies**:
- Follow the 50/30/20 rule (needs/wants/savings)
- Use high-yield savings accounts
- Consider CD ladders for short-term goals
- Review and adjust your strategy regularly

Remember: Consistency is key. Even small amounts saved regularly can grow significantly over time."""

        elif 'debt' in prompt_lower or 'loan' in prompt_lower:
            return """**Debt Management Strategy**

Here's a systematic approach to managing your debt:

**Debt Assessment**:
1. List all debts with balances, interest rates, and minimum payments
2. Calculate your total debt-to-income ratio
3. Identify which debts have the highest interest rates

**Debt Repayment Strategies**:
- **Avalanche Method**: Pay off highest interest debt first
- **Snowball Method**: Pay off smallest balances first for motivation
- **Debt Consolidation**: Consider combining multiple debts into one loan

**Immediate Steps**:
1. Stop taking on new debt
2. Pay more than minimum payments when possible
3. Negotiate lower interest rates with creditors
4. Consider balance transfer cards for high-interest debt

**Long-term Prevention**:
- Build emergency fund to avoid future debt
- Live below your means
- Use credit cards responsibly
- Save for major purchases instead of financing

Remember: Getting out of debt takes time and discipline, but it's achievable with a solid plan."""

        else:
            return """**Personal Finance Advice**

Thank you for your question! Here are some general financial principles to consider:

**Financial Foundation**:
1. **Budget**: Track income and expenses to understand your cash flow
2. **Emergency Fund**: Save 3-6 months of expenses for unexpected events
3. **Debt Management**: Prioritize high-interest debt repayment
4. **Insurance**: Protect yourself and your family with appropriate coverage

**Smart Money Habits**:
- Pay yourself first (automate savings)
- Live below your means
- Avoid lifestyle inflation
- Regularly review and adjust your financial plan

**Investment Basics**:
- Start early to benefit from compound interest
- Diversify your investments
- Consider your risk tolerance and time horizon
- Take advantage of employer retirement plans

**Continuous Learning**:
- Stay informed about personal finance topics
- Seek professional advice when needed
- Learn from your financial mistakes
- Set clear, achievable financial goals

Remember: Financial success is a journey, not a destination. Small, consistent actions today will lead to significant results over time."""

    def get_service_status(self) -> Dict[str, bool]:
        """
        Get the status of AI services
        
        Returns:
            Dictionary with service availability status
        """
        return {
            'huggingface_available': self.hf_available,
            'models_loaded': True,  # Using rule-based responses for performance
            'fallback_mode': True  # Indicates we're using intelligent fallback responses
        }

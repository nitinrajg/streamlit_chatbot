# Personal Finance Chatbot

An intelligent conversational AI system that provides personalized financial guidance for savings, taxes, and investments. Built with Streamlit, FastAPI, and IBM Watson AI services.

## 🎯 Project Overview

This Personal Finance Chatbot leverages IBM's generative AI models and Watson services to provide:

- **Personalized Financial Guidance** - Customized advice based on user profiles
- **AI-Generated Budget Summaries** - Automatic budget analysis and recommendations
- **Spending Insights and Suggestions** - Actionable recommendations for expense optimization
- **Demographic-Aware Communication** - Adapts tone based on user type (student vs professional)
- **Conversational NLP Experience** - Natural, context-aware interactions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   IBM Watson    │
│   Frontend      │◄──►│   Backend       │◄──►│   AI Services   │
│                 │    │                 │    │                 │
│ • Chat UI       │    │ • API Endpoints │    │ • NLU Analysis  │
│ • Forms         │    │ • Data Processing│   │ • Response Gen  │
│ • Visualizations│    │ • Validation    │    │ • Sentiment     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Features

### 1. General Chat Interface
- Ask any financial question
- Get personalized advice based on your profile
- View sentiment analysis and keyword extraction
- Support for different personas (student, professional, general)

### 2. Budget Summary Analysis
- Comprehensive budget breakdown
- Financial health assessment
- Top spending category identification
- Personalized recommendations
- Visual expense distribution charts

### 3. Spending Insights
- Deep spending pattern analysis
- Goal feasibility assessment
- Behavioral insights
- Actionable recommendations
- Category-wise spending breakdown

### 4. Text Analysis
- Sentiment analysis of financial text
- Keyword extraction
- Entity recognition
- Category classification
- Interactive visualizations

## 📋 Prerequisites

- Python 3.8 or higher
- IBM Cloud account (for Watson services)
- Basic understanding of personal finance concepts

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd personal-finance-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
# IBM Watson NLU Credentials
NLU_API_KEY=your_nlu_api_key_here
NLU_URL=https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/your_instance_id

# IBM Watsonx.ai Credentials
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=meta-llama/llama-2-70b-chat
PROJECT_ID=your_project_id_here

# Application Settings
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
```

### 5. Get IBM Watson Credentials

#### For Watson NLU:
1. Go to [IBM Cloud](https://cloud.ibm.com/)
2. Create a Natural Language Understanding service
3. Get your API key and URL from the service credentials

#### For Watsonx.ai:
1. Go to [IBM Watsonx.ai](https://www.ibm.com/products/watsonx-ai)
2. Set up your project and get the necessary credentials
3. Note your model ID and project ID

## 🚀 Running the Application

### 1. Start the Backend Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend Application
```bash
streamlit run streamlit_app.py
```

### 3. Access the Application
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📖 Usage Guide

### General Chat
1. Navigate to the "Chat" section
2. Select your profile type (Student, Professional, or General)
3. Type your financial question
4. Click "Send" to get personalized advice
5. View the analysis details in the expandable section

### Budget Summary
1. Go to the "Budget Summary" section
2. Enter your monthly income and savings goal
3. Fill in your monthly expenses
4. Select your profile type
5. Click "Analyze Budget" to get comprehensive insights
6. Review the charts and recommendations

### Spending Insights
1. Navigate to "Spending Insights"
2. Enter your income and financial goals
3. Provide detailed expense breakdown
4. Select your profile type
5. Click "Analyze Spending" for behavioral insights
6. Review action items and recommendations

### Text Analysis
1. Go to "Text Analysis"
2. Enter any financial text
3. Click "Analyze Text"
4. View sentiment, keywords, entities, and categories

## 🔧 API Endpoints

### Chat Endpoint
```http
POST /chat
{
  "message": "How can I save money?",
  "persona": "student"
}
```

### Budget Summary Endpoint
```http
POST /budget-summary
{
  "income": 5000.0,
  "expenses": {
    "rent": 1500.0,
    "groceries": 400.0
  },
  "savings_goal": 500.0,
  "persona": "professional"
}
```

### Spending Insights Endpoint
```http
POST /spending-insights
{
  "income": 5000.0,
  "expenses": {
    "rent": 1500.0,
    "groceries": 400.0
  },
  "goals": ["emergency fund", "vacation"],
  "persona": "general"
}
```

### NLU Analysis Endpoint
```http
POST /nlu
{
  "text": "I'm struggling with my budget"
}
```

## 🎨 Customization

### Adding New Personas
1. Modify the `build_basic_prompt()` function in `utils.py`
2. Add persona-specific logic in the prompt building functions
3. Update the frontend persona selection options

### Customizing Prompts
1. Edit the prompt templates in `utils.py`
2. Adjust the tone and complexity based on your needs
3. Test with different user inputs

### Adding New Features
1. Create new API endpoints in `main.py`
2. Add corresponding frontend interfaces in `streamlit_app.py`
3. Update the utility functions as needed

## 🔍 Troubleshooting

### Common Issues

#### Backend Not Starting
- Check if port 8000 is available
- Verify all dependencies are installed
- Check environment variables are set correctly

#### Frontend Connection Issues
- Ensure backend is running on http://localhost:8000
- Check CORS settings in `main.py`
- Verify API health endpoint is responding

#### AI Service Errors
- Verify IBM Watson credentials are correct
- Check API quotas and limits
- Review error logs for specific issues

#### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Performance Optimization

### For Production Deployment
1. Use production-grade WSGI server (Gunicorn)
2. Implement caching for AI responses
3. Add rate limiting for API endpoints
4. Use environment-specific configurations
5. Implement proper error handling and logging

### Scaling Considerations
1. Use load balancers for multiple instances
2. Implement database for user sessions
3. Add monitoring and analytics
4. Consider using Redis for caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IBM Watson AI services for natural language processing
- Streamlit for the user interface framework
- FastAPI for the backend API framework
- The open-source community for various libraries and tools

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at http://localhost:8000/docs

## 🔮 Future Enhancements

- Integration with real financial data sources
- Mobile app development
- Advanced analytics and reporting
- Multi-language support
- Voice interface integration
- Machine learning model fine-tuning
- Integration with financial institutions (with user consent)
- Real-time market data integration
- Goal tracking and progress monitoring
- Social features for financial communities

---

**Note**: This chatbot provides educational financial advice and should not be considered as professional financial consultation. Always consult with qualified financial advisors for personalized financial planning.

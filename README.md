# AI Government Assistance Platform 🏛️

An enterprise-level AI-powered web application that helps Indian citizens determine their eligibility for government schemes, startup funding, and subsidies using Machine Learning.

## 🎯 Overview

This platform transforms the basic eligibility screening into a comprehensive multi-page dashboard with 8 specialized pages, providing deep insights into government scheme eligibility, personalized recommendations, and analytics.

## ✨ Features

### 📊 **8-Page Enterprise Dashboard**

1. **Dashboard Overview** - Real-time analytics and metrics
2. **Eligibility Predictor** - AI-powered eligibility assessment with business metrics
3. **Scheme Recommendation Engine** - Personalized scheme matching
4. **User Profile Analyzer** - Comprehensive profile scoring and risk assessment
5. **Eligibility Score Breakdown** - Feature importance and factor analysis
6. **Government Scheme Database** - Searchable catalog of 15+ schemes
7. **Analytics Dashboard** - Data visualization and distribution analysis
8. **Admin Model Monitoring** - Model performance and system health

### 🚀 **Enterprise Features**

- **Multi-Page Navigation** - Clean sidebar navigation with 8 specialized pages
- **Advanced ML Model** - RandomForestClassifier with 9 features and 500+ training samples
- **Real-Time Analytics** - Track users, eligibility rates, and scheme popularity
- **Interactive Charts** - Plotly-powered visualizations (pie, bar, line, histogram)
- **Profile Scoring** - Risk level, financial strength, startup readiness, support eligibility
- **Scheme Database** - 15 government schemes with detailed information
- **Fast Performance** - Optimized with @st.cache_resource and @st.cache_data
- **Professional UI** - Modern gradient design with colored status indicators

### 🎨 **UI/UX Enhancements**

- Professional color scheme with gradients
- Success/Warning/Error/Info cards
- Metric cards with deltas
- Progress bars and score visualizations
- Responsive layout with wide mode
- Clean sidebar navigation
- Mobile-friendly design

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.28+
- **Machine Learning**: scikit-learn (RandomForestClassifier)
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly
- **Model Persistence**: joblib
- **Architecture**: Clean, modular design

## 📦 Project Structure

```
AI-Government-Scheme-Eligibility-Predictor/
├── app.py              # Main multi-page Streamlit application
├── model.py            # ML model training and prediction logic
├── data.py             # Synthetic data generation and scheme database
├── utils.py            # Helper functions, styling, and chart utilities
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # Documentation
```

## 🧠 Machine Learning Model

### Features (9 Total)

1. **Age** - 18-80 years
2. **Income** - Annual income in INR
3. **Education** - School/Graduate/Postgraduate
4. **Employment Status** - Employed/Unemployed
5. **Location** - Urban/Rural
6. **Gender** - Male/Female
7. **Startup Owner** - Yes/No
8. **Business Age** - 0-20 years
9. **Business Revenue** - Annual revenue in INR

### Model Specifications

- **Algorithm**: Random Forest Classifier
- **Training Data**: 500 synthetic samples using sklearn.make_classification
- **Estimators**: 100 trees
- **Max Depth**: 10
- **Features**: 9 (7 informative, 2 redundant)
- **Classes**: 2 (Eligible/Not Eligible)
- **Performance**: Optimized with caching for fast predictions

## 📊 Pages Detailed Description

### 1. Dashboard Overview
- Total users analyzed metric
- Eligible users percentage
- Average income analyzed
- Most recommended scheme
- Pie chart: Eligible vs Not Eligible
- Bar chart: Top 5 scheme popularity
- Line chart: Eligibility trend over time

### 2. Eligibility Predictor
- 9-field input form (age, income, education, employment, location, gender, startup details)
- Real-time eligibility prediction
- Eligibility score (0-100)
- Confidence percentage
- Top 3 recommended schemes
- Profile summary with metrics

### 3. Scheme Recommendation Engine
- Personalized scheme matching based on profile
- Up to 5 top-matched schemes
- Match score percentage
- Detailed benefits and eligibility criteria
- Income range compatibility

### 4. User Profile Analyzer
- Risk level assessment (Low/Medium/High)
- Financial strength score (0-100)
- Startup readiness score (0-100)
- Support eligibility score (0-100)
- Profile insights and recommendations
- Basic and business information summary

### 5. Eligibility Score Breakdown
- Feature importance visualization
- Horizontal bar chart showing impact of each factor
- Component score breakdown
- Detailed explanations of each feature
- User's personalized score analysis

### 6. Government Scheme Database
- Searchable database of 15+ schemes
- Filter by category (Startup, Finance, Skills, Women, Youth, etc.)
- Filter by income range
- Detailed scheme information
- Export to CSV functionality

### 7. Analytics Dashboard
- Income distribution histogram
- Education level pie chart
- Location distribution
- Employment status breakdown
- Eligibility rates by demographics
- Age group analysis

### 8. Admin Model Monitoring
- Model accuracy metric
- Total predictions counter
- Model version and status
- Performance metrics (Precision, Recall, F1)
- Feature importance table
- System health indicators
- Admin actions (refresh model, reset analytics)

## 📋 Installation

### Prerequisites

- Python 3.11+
- pip package manager

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ravigohel142996/AI-Government-Scheme-Eligibility-Predictor.git
   cd AI-Government-Scheme-Eligibility-Predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

## ☁️ Streamlit Cloud Deployment

This application is optimized for Streamlit Cloud:

1. Fork/Clone this repository
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app and connect to your repository
4. Set main file to `app.py`
5. Deploy!

The app uses only lightweight dependencies and requires no database or external services.

## 🎯 Usage Guide

### Step 1: Navigate to Eligibility Predictor
- Fill in all 9 fields in the form
- Click "Predict Eligibility"

### Step 2: Review Results
- Check your eligibility status
- View eligibility score and confidence
- Explore recommended schemes

### Step 3: Analyze Your Profile
- Navigate to "User Profile Analyzer"
- View your risk level, financial strength, startup readiness
- Get personalized insights

### Step 4: Explore Schemes
- Visit "Government Scheme Database"
- Search and filter schemes
- Download the database as CSV

### Step 5: View Analytics
- Check "Analytics Dashboard" for insights
- Monitor "Dashboard Overview" for aggregate metrics
- Review "Eligibility Score Breakdown" to understand factors

## 🎓 Government Schemes Included

1. **Startup India** - Startup funding and tax exemption
2. **Mudra Loan (Shishu, Kishore, Tarun)** - Micro and small business loans
3. **Skill India (PMKVY)** - Free skill training and certification
4. **Women Entrepreneurship Program** - Loans for women entrepreneurs
5. **Stand-Up India** - Funding for SC/ST/Women enterprises
6. **Youth Startup Program** - Youth entrepreneurship support
7. **Credit Guarantee Scheme** - Collateral-free credit
8. **PM Employment Generation Program** - Employment assistance
9. **Atal Innovation Mission** - Innovation funding
10. **Rural Self Employment** - Rural livelihood support
11. **Digital India Initiative** - Digital skills training
12. **Make in India** - Manufacturing incentives
13. **National Rural Livelihood Mission** - Rural development
14. And more...

## 🔒 Privacy & Security

- **No Data Storage**: All data is processed in real-time and not stored
- **No Database**: The app runs entirely in-memory
- **Client-Side Processing**: Information never leaves your session
- **Open Source**: Full transparency of the prediction logic

## 📊 Performance Optimization

- **Caching**: `@st.cache_resource` for model loading
- **Fast Loading**: <3 seconds load time
- **Efficient Rendering**: Optimized component rendering
- **Memory Efficient**: Minimal memory footprint

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available for educational and public service purposes.

## ⚠️ Disclaimer

This is an AI-based prediction tool for informational purposes only. Always verify eligibility with official government sources and portals before applying for any schemes.

## 👨‍💻 Author

Developed with ❤️ for empowering citizens through technology

## 🆕 Version History

### Version 2.0 (Current)
- ✅ Multi-page dashboard with 8 pages
- ✅ Enhanced ML model with 9 features
- ✅ Scheme recommendation engine
- ✅ Profile analyzer with scoring
- ✅ Analytics dashboard
- ✅ Admin monitoring panel
- ✅ Plotly visualizations
- ✅ Performance optimizations
- ✅ Enterprise-level UI/UX

### Version 1.0
- Basic single-page eligibility predictor
- 6 features ML model
- Simple matplotlib charts

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Note**: This application uses machine learning predictions based on historical patterns and should be used as a preliminary screening tool. Final eligibility should be confirmed through official government channels.

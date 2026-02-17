# AI Government Scheme Eligibility Predictor 🏛️

An AI-powered web application that helps Indian citizens determine their eligibility for government schemes, startup funding, and subsidies using Machine Learning.

## 🎯 Overview

Millions of people don't know which government schemes they qualify for, their loan eligibility, startup support opportunities, or subsidy eligibility. This AI system solves that problem by providing instant, accurate predictions based on demographic and socioeconomic factors.

## ✨ Features

- **Instant Predictions**: Get eligibility results in seconds
- **AI-Powered Accuracy**: Uses Decision Tree Classifier trained on eligibility patterns
- **User-Friendly Interface**: Modern, professional UI with Indian government aesthetic
- **Confidence Scores**: See how confident the AI is about its predictions
- **Educational Insights**: Learn what factors influence eligibility
- **Privacy First**: No data storage, all processing happens in real-time
- **Lightweight**: Fast performance with minimal dependencies

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn (DecisionTreeClassifier)
- **Data Processing**: pandas, numpy
- **Design**: Custom CSS with Indian national color palette

## 🎨 Design Philosophy

The application features a modern institutional aesthetic that balances professionalism with approachability:

- **Color Palette**: Indian national colors (saffron orange #E67E22, deep green #1B7F5C, beige #F4E6D7)
- **Typography**: Clean Poppins font family
- **Layout**: Card-based design with soft shadows and rounded corners
- **Interaction**: Smooth animations and responsive feedback

## 🧠 How It Works

### Machine Learning Model

The system uses a **Decision Tree Classifier** trained on a synthetic dataset that represents historical eligibility decisions. The model considers four key factors:

1. **Age**: Age of the applicant (18-80 years)
2. **Income**: Annual income in Indian Rupees
3. **Education**: Education level (School/Graduate/Postgraduate)
4. **Employment**: Current employment status (Employed/Unemployed)

### Prediction Logic

The model prioritizes:
- Unemployed individuals with lower income
- Citizens with higher education levels
- Age range between 18-60 years
- Annual income below ₹500,000

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
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

## 📁 Project Structure

```
AI-Government-Scheme-Eligibility-Predictor/
├── app.py              # Streamlit frontend application
├── model.py            # ML model and prediction logic
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## �� Usage

1. **Enter Your Details**:
   - Adjust the age slider to your current age
   - Set your annual income using the income slider
   - Select your highest education level
   - Choose your employment status

2. **Check Eligibility**:
   - Click the "Check Eligibility" button
   - View your eligibility status instantly

3. **Review Results**:
   - See if you're eligible or not
   - Check the confidence score
   - View your profile summary
   - Explore how the AI makes decisions

## 🎓 Educational Features

The application includes an educational component that explains:
- How the AI model works
- Which factors are most important for eligibility
- The technology behind the predictions

## 🔒 Privacy & Security

- **No Data Storage**: All data is processed in real-time and not stored
- **No Database**: The app runs entirely in-memory
- **Client-Side Processing**: Your information never leaves your session
- **Open Source**: Full transparency of the prediction logic

## 📊 Model Performance

The Decision Tree Classifier is trained on 50 data points covering various demographic profiles. The model uses:
- **Max Depth**: 5 (prevents overfitting)
- **Random State**: 42 (ensures reproducibility)
- **Features**: 4 (age, income, education, employment)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available for educational and public service purposes.

## ⚠️ Disclaimer

This is an AI-based prediction tool for informational purposes only. Always verify eligibility with official government sources and portals before applying for any schemes.

## 👨‍💻 Author

Developed with ❤️ for empowering citizens through technology

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Note**: This application uses machine learning predictions based on historical patterns and should be used as a preliminary screening tool. Final eligibility should be confirmed through official government channels.

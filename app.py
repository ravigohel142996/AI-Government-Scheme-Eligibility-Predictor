"""
AI Government Scheme Eligibility Predictor - Streamlit Application
-------------------------------------------------------------------
A modern, professional web application that helps citizens check their
eligibility for government schemes using Machine Learning.

Design: Indian government aesthetic with saffron, green, and beige colors
Technology: Streamlit + scikit-learn DecisionTreeClassifier
"""

import streamlit as st
from model import predict_eligibility, get_feature_importance
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="AI Government Scheme Eligibility Predictor",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Indian Government Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .main {
        background-color: #F5F7FA;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #E67E22 0%, #F2994A 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .subtitle {
        color: #FAFAFA;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 0;
    }
    
    /* Card Styling */
    .info-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid #1B7F5C;
    }
    
    /* Input Section */
    .stSlider {
        padding: 0.5rem 0;
    }
    
    .stSelectbox {
        padding: 0.5rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #F2994A 0%, #E67E22 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(246, 149, 74, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #E67E22 0%, #D35400 100%);
        box-shadow: 0 6px 16px rgba(246, 149, 74, 0.4);
        transform: translateY(-2px);
    }
    
    /* Success/Error Messages */
    .success-box {
        background-color: #D5F5E3;
        border-left: 5px solid #1B7F5C;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    .error-box {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7F8C8D;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #E0E0E0;
    }
    
    /* Feature Card */
    .feature-card {
        background-color: #F4E6D7;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #F2994A;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-container">
    <div class="main-title">🏛️ AI Government Scheme Eligibility Predictor</div>
    <div class="subtitle">Empowering Citizens Through Smart Technology</div>
</div>
""", unsafe_allow_html=True)

# Introduction Card
st.markdown("""
<div class="info-card">
    <h3 style="color: #2C2C2C; margin-top: 0;">About This Service</h3>
    <p style="color: #555555; line-height: 1.6;">
        Our AI-powered system helps you instantly determine your eligibility for various government schemes, 
        startup funding, and subsidies. Simply provide your details below and get accurate predictions 
        powered by machine learning technology.
    </p>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Information")
    
    # Age Input
    age = st.slider(
        "Age (years)",
        min_value=18,
        max_value=80,
        value=30,
        step=1,
        help="Select your current age"
    )
    
    # Income Input
    income = st.slider(
        "Annual Income (INR)",
        min_value=10000,
        max_value=2000000,
        value=300000,
        step=10000,
        format="₹%d",
        help="Select your annual income in Indian Rupees"
    )

with col2:
    st.markdown("### 🎓 Qualifications")
    
    # Education Input
    education = st.selectbox(
        "Education Level",
        options=["School", "Graduate", "Postgraduate"],
        index=1,
        help="Select your highest level of education"
    )
    
    # Employment Status Input
    employed = st.selectbox(
        "Employment Status",
        options=["Employed", "Unemployed"],
        index=0,
        help="Select your current employment status"
    )

# Spacing
st.markdown("<br>", unsafe_allow_html=True)

# Check Eligibility Button
if st.button("🔍 Check Eligibility"):
    # Show loading spinner while processing
    with st.spinner("Analyzing your eligibility..."):
        # Get prediction from ML model
        result, confidence = predict_eligibility(age, income, education, employed)
        
        # Display results based on prediction
        if result == "Eligible":
            st.success("✅ **Congratulations! You are ELIGIBLE for government schemes.**")
            st.markdown(f"""
            <div class="success-box">
                <h3 style="color: #1B7F5C; margin-top: 0;">Eligibility Confirmed</h3>
                <p style="color: #2C2C2C; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    Based on your profile, you qualify for government assistance programs.
                </p>
                <p style="color: #555555; font-size: 1.3rem; font-weight: 600;">
                    Confidence Score: {confidence:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional Information
            st.info("📋 **Next Steps:** Visit your nearest government office or official portal to apply for schemes.")
            
        else:
            st.error("❌ **You are currently NOT ELIGIBLE for government schemes.**")
            st.markdown(f"""
            <div class="error-box">
                <h3 style="color: #C0392B; margin-top: 0;">Eligibility Not Met</h3>
                <p style="color: #2C2C2C; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    Based on your current profile, you do not meet the eligibility criteria.
                </p>
                <p style="color: #555555; font-size: 1.3rem; font-weight: 600;">
                    Confidence Score: {confidence:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Suggestions
            st.info("💡 **Suggestion:** Some criteria may change over time. Consider checking again in the future or exploring other schemes.")
        
        # Show input summary
        st.markdown("---")
        st.markdown("### 📊 Your Profile Summary")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Age", f"{age} years")
        
        with col_b:
            st.metric("Income", f"₹{income:,}")
        
        with col_c:
            st.metric("Education", education)
        
        with col_d:
            st.metric("Employment", employed)

# Feature Importance Section (Optional Educational Component)
with st.expander("🔬 How does the AI make predictions?"):
    st.markdown("""
    <div class="feature-card">
        <p style="margin: 0; color: #2C2C2C; line-height: 1.6;">
            Our AI system uses a <strong>Decision Tree Classifier</strong> trained on historical eligibility data. 
            It analyzes multiple factors including your age, income, education level, and employment status 
            to make accurate predictions about scheme eligibility.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show feature importance
    importance = get_feature_importance()
    st.markdown("#### Key Factors Influencing Eligibility:")
    
    importance_df = pd.DataFrame({
        'Factor': importance.keys(),
        'Importance': [f"{v*100:.1f}%" for v in importance.values()]
    })
    st.table(importance_df)

# Information Section
st.markdown("---")
st.markdown("### ℹ️ Important Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #2C2C2C; margin-top: 0;">✨ Benefits</h4>
        <ul style="color: #555555; line-height: 1.8;">
            <li>Instant predictions</li>
            <li>AI-powered accuracy</li>
            <li>Free to use</li>
            <li>Privacy protected</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #2C2C2C; margin-top: 0;">🎯 Applicable Schemes</h4>
        <ul style="color: #555555; line-height: 1.8;">
            <li>Startup funding</li>
            <li>Education subsidies</li>
            <li>Healthcare schemes</li>
            <li>Employment programs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="margin: 0;">
        <strong>AI Government Scheme Eligibility Predictor</strong><br>
        Powered by Machine Learning | Built with Streamlit<br>
        © 2024 | For informational purposes only. Please verify with official sources.
    </p>
</div>
""", unsafe_allow_html=True)

"""
AI Government Scheme Eligibility Predictor - Streamlit Application
-------------------------------------------------------------------
A modern, professional web application that helps citizens check their
eligibility for government schemes using Machine Learning.

Design: Clean and professional layout
Technology: Streamlit + scikit-learn RandomForestClassifier
"""

import streamlit as st
from model import predict_eligibility, get_feature_importance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Government Scheme Eligibility Predictor",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Theme
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    
    /* Success/Error Messages */
    .success-box {
        background-color: #D5F5E3;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    .error-box {
        background-color: #FADBD8;
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
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
    <div class="subtitle">Powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
### Welcome to the AI Government Scheme Eligibility Predictor

This application uses advanced machine learning to help you determine your eligibility 
for various government schemes, funding, and subsidies. Simply fill in your details 
in the sidebar and click the **Predict** button to get instant results.
""")

# Sidebar for user inputs
st.sidebar.header("📋 Enter Your Details")
st.sidebar.markdown("---")

try:
    # Age Input
    age = st.sidebar.slider(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
        step=1,
        help="Select your current age"
    )
    
    # Income Input
    income = st.sidebar.slider(
        "Annual Income (₹)",
        min_value=10000,
        max_value=2000000,
        value=300000,
        step=10000,
        help="Select your annual income in Indian Rupees"
    )
    
    # Education Input
    education = st.sidebar.selectbox(
        "Education Level",
        options=["School", "Graduate", "Postgraduate"],
        index=1,
        help="Select your highest level of education"
    )
    
    # Employment Status Input
    employment_status = st.sidebar.selectbox(
        "Employment Status",
        options=["Employed", "Unemployed"],
        index=0,
        help="Select your current employment status"
    )
    
    # Location Input
    location = st.sidebar.selectbox(
        "Location",
        options=["Urban", "Rural"],
        index=0,
        help="Select your location type"
    )
    
    # Gender Input
    gender = st.sidebar.selectbox(
        "Gender",
        options=["Male", "Female"],
        index=0,
        help="Select your gender"
    )
    
    st.sidebar.markdown("---")
    
    # Predict Button
    predict_button = st.sidebar.button("🔍 Predict Eligibility", use_container_width=True)
    
    # Handle prediction
    if predict_button:
        # Show loading spinner while processing
        with st.spinner("Analyzing your eligibility..."):
            try:
                # Get prediction from ML model
                result, confidence = predict_eligibility(age, income, education, employment_status, location, gender)
                
                # Check if prediction was successful
                if result == "Error":
                    st.error("❌ An error occurred while making the prediction. Please try again.")
                else:
                    # Display results based on prediction
                    if result == "Eligible":
                        st.success("✅ **Congratulations! You are ELIGIBLE for government schemes.**")
                        st.markdown(f"""
                        <div class="success-box">
                            <h3 style="color: #28a745; margin-top: 0;">✅ Eligibility Confirmed</h3>
                            <p style="color: #2C2C2C; font-size: 1.1rem; margin-bottom: 0.5rem;">
                                Based on your profile, you qualify for government assistance programs.
                            </p>
                            <p style="color: #555555; font-size: 1.3rem; font-weight: 600;">
                                Confidence Score: {confidence:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ **You are currently NOT ELIGIBLE for government schemes.**")
                        st.markdown(f"""
                        <div class="error-box">
                            <h3 style="color: #dc3545; margin-top: 0;">❌ Eligibility Not Met</h3>
                            <p style="color: #2C2C2C; font-size: 1.1rem; margin-bottom: 0.5rem;">
                                Based on your current profile, you do not meet the eligibility criteria.
                            </p>
                            <p style="color: #555555; font-size: 1.3rem; font-weight: 600;">
                                Confidence Score: {confidence:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show prediction confidence chart
                    st.markdown("### 📊 Prediction Confidence")
                    
                    # Create matplotlib chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Determine colors based on result
                    if result == "Eligible":
                        colors = ['#28a745', '#dc3545']
                        labels = ['Eligible', 'Not Eligible']
                        values = [confidence, 100 - confidence]
                    else:
                        colors = ['#dc3545', '#28a745']
                        labels = ['Not Eligible', 'Eligible']
                        values = [confidence, 100 - confidence]
                    
                    # Create horizontal bar chart
                    bars = ax.barh(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                    
                    # Add value labels on bars
                    for i, (bar, value) in enumerate(zip(bars, values)):
                        ax.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=12, fontweight='bold')
                    
                    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
                    ax.set_xlim(0, 110)
                    ax.set_title('Model Prediction Confidence', fontsize=14, fontweight='bold', pad=15)
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Display the chart
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show input summary
                    st.markdown("---")
                    st.markdown("### 📝 Your Profile Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Age", f"{age} years")
                        st.metric("Education", education)
                    
                    with col2:
                        st.metric("Income", f"₹{income:,}")
                        st.metric("Employment", employment_status)
                    
                    with col3:
                        st.metric("Location", location)
                        st.metric("Gender", gender)
                    
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.info("Please check your inputs and try again.")
    
except Exception as e:
    st.error(f"❌ Error loading the application: {str(e)}")
    st.info("Please refresh the page or contact support if the issue persists.")

# Feature Importance Section
st.markdown("---")
with st.expander("🔬 How does the AI make predictions?"):
    try:
        st.markdown("""
        Our AI system uses a **Random Forest Classifier** trained on historical eligibility data. 
        It analyzes multiple factors including your age, income, education level, employment status,
        location, and gender to make accurate predictions about scheme eligibility.
        """)
        
        # Show feature importance
        importance = get_feature_importance()
        if importance:
            st.markdown("#### Key Factors Influencing Eligibility:")
            
            importance_df = pd.DataFrame({
                'Factor': importance.keys(),
                'Importance': [f"{v*100:.1f}%" for v in importance.values()]
            })
            st.table(importance_df)
    except Exception as e:
        st.warning("Unable to load feature importance information.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <strong>AI Government Scheme Eligibility Predictor</strong><br>
    Powered by Machine Learning | Built with Streamlit<br>
    © 2024 | For informational purposes only
</div>
""", unsafe_allow_html=True)

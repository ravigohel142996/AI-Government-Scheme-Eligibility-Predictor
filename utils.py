"""
Utility Module - AI Government Assistance Platform
--------------------------------------------------
Helper functions, styling, and caching utilities.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def apply_custom_css():
    """Apply custom CSS styling for the application."""
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
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #2C3E50;
        }

        [data-testid="stSidebar"] * {
            color: #FFFFFF;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #E8EEF9 !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 0.35rem 0.4rem;
            border-radius: 8px;
            transition: background-color 0.2s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background-color: rgba(255, 255, 255, 0.12);
        }

        [data-testid="stSidebar"] [role="radiogroup"] input:checked + div {
            color: #FFFFFF !important;
            font-weight: 600;
        }
        
        /* Metrics Styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.5rem 2rem;
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
            transform: translateY(-2px);
        }
        
        /* Success/Warning/Error Cards */
        .success-card {
            background-color: #D5F5E3;
            border-left: 5px solid #28a745;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .warning-card {
            background-color: #FFF3CD;
            border-left: 5px solid #ffc107;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .error-card {
            background-color: #FADBD8;
            border-left: 5px solid #dc3545;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .info-card {
            background-color: #D1ECF1;
            border-left: 5px solid #17a2b8;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 2rem;
            background-color: #F8F9FA;
            border-radius: 8px 8px 0 0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

def display_metric_card(label, value, delta=None, delta_color="normal"):
    """
    Display a metric card with optional delta.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        delta_color: Color for delta (normal, inverse, off)
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)

def display_success_card(title, message):
    """Display a success message card."""
    st.markdown(f"""
    <div class="success-card">
        <h3 style="color: #28a745; margin-top: 0;">✅ {title}</h3>
        <p style="color: #2C2C2C; font-size: 1rem;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def display_warning_card(title, message):
    """Display a warning message card."""
    st.markdown(f"""
    <div class="warning-card">
        <h3 style="color: #856404; margin-top: 0;">⚠️ {title}</h3>
        <p style="color: #2C2C2C; font-size: 1rem;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def display_error_card(title, message):
    """Display an error message card."""
    st.markdown(f"""
    <div class="error-card">
        <h3 style="color: #dc3545; margin-top: 0;">❌ {title}</h3>
        <p style="color: #2C2C2C; font-size: 1rem;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def display_info_card(title, message):
    """Display an info message card."""
    st.markdown(f"""
    <div class="info-card">
        <h3 style="color: #0c5460; margin-top: 0;">ℹ️ {title}</h3>
        <p style="color: #2C2C2C; font-size: 1rem;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def create_pie_chart(labels, values, title):
    """
    Create a Plotly pie chart.
    
    Args:
        labels: List of labels
        values: List of values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    colors = ['#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6c757d']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Poppins')),
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_bar_chart(x, y, title, x_label, y_label, color='#667eea'):
    """
    Create a Plotly bar chart.
    
    Args:
        x: X-axis data
        y: Y-axis data
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        color: Bar color
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Bar(
        x=x,
        y=y,
        marker=dict(color=color),
        text=y,
        textposition='auto',
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Poppins')),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        margin=dict(l=20, r=20, t=60, b=60),
        showlegend=False
    )
    
    return fig

def create_line_chart(x, y, title, x_label, y_label):
    """
    Create a Plotly line chart.
    
    Args:
        x: X-axis data
        y: Y-axis data
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2')
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Poppins')),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        margin=dict(l=20, r=20, t=60, b=60),
        showlegend=False
    )
    
    return fig

def create_horizontal_bar_chart(labels, values, title, color_scale='Blues'):
    """
    Create a Plotly horizontal bar chart.
    
    Args:
        labels: Y-axis labels
        values: X-axis values
        title: Chart title
        color_scale: Color scale name
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=values,
            colorscale=color_scale,
            showscale=False
        ),
        text=[f'{v:.2f}' for v in values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Poppins')),
        xaxis_title='Importance Score',
        height=400,
        margin=dict(l=150, r=20, t=60, b=60),
        showlegend=False
    )
    
    return fig

def create_histogram(data, title, x_label, nbins=30):
    """
    Create a Plotly histogram.
    
    Args:
        data: Data array
        title: Chart title
        x_label: X-axis label
        nbins: Number of bins
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Histogram(
        x=data,
        nbinsx=nbins,
        marker=dict(color='#667eea'),
        opacity=0.75
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Poppins')),
        xaxis_title=x_label,
        yaxis_title='Count',
        height=400,
        margin=dict(l=20, r=20, t=60, b=60),
        showlegend=False
    )
    
    return fig

def calculate_risk_level(income, employment_status, age):
    """
    Calculate risk level for user profile.
    
    Args:
        income: Annual income
        employment_status: Employment status
        age: User age
        
    Returns:
        Risk level string and score (0-100)
    """
    risk_score = 50  # Start at medium
    
    # Income factor
    if income < 200000:
        risk_score += 20
    elif income < 500000:
        risk_score += 10
    else:
        risk_score -= 10
    
    # Employment factor
    if employment_status == 'Unemployed':
        risk_score += 15
    else:
        risk_score -= 15
    
    # Age factor
    if age < 25 or age > 60:
        risk_score += 10
    else:
        risk_score -= 5
    
    # Clamp between 0 and 100
    risk_score = max(0, min(100, risk_score))
    
    if risk_score < 30:
        return "Low Risk", risk_score
    elif risk_score < 60:
        return "Medium Risk", risk_score
    else:
        return "High Risk", risk_score

def calculate_financial_strength(income, business_revenue):
    """
    Calculate financial strength score.
    
    Args:
        income: Annual income
        business_revenue: Annual business revenue
        
    Returns:
        Score (0-100)
    """
    # Base score from income
    income_score = min((income / 1000000) * 50, 50)
    
    # Additional score from business revenue
    revenue_score = min((business_revenue / 5000000) * 50, 50)
    
    total_score = income_score + revenue_score
    return min(100, total_score)

def calculate_startup_readiness(age, education, startup_owner, business_age):
    """
    Calculate startup readiness score.
    
    Args:
        age: User age
        education: Education level
        startup_owner: Whether user owns a startup
        business_age: Age of business
        
    Returns:
        Score (0-100)
    """
    score = 0
    
    # Age factor (prefer 25-45)
    if 25 <= age <= 45:
        score += 30
    elif age < 25:
        score += 20
    else:
        score += 15
    
    # Education factor
    if education == 'Postgraduate':
        score += 30
    elif education == 'Graduate':
        score += 25
    else:
        score += 15
    
    # Startup ownership
    if startup_owner == 'Yes':
        score += 25
        # Business maturity
        if business_age >= 2:
            score += 15
        elif business_age >= 1:
            score += 10
        else:
            score += 5
    else:
        score += 10
    
    return min(100, score)

def calculate_support_eligibility(income, employment_status, location):
    """
    Calculate support eligibility score.
    
    Args:
        income: Annual income
        employment_status: Employment status
        location: Location (Urban/Rural)
        
    Returns:
        Score (0-100)
    """
    score = 0
    
    # Income factor (lower income = higher eligibility)
    if income < 200000:
        score += 40
    elif income < 500000:
        score += 30
    elif income < 1000000:
        score += 20
    else:
        score += 10
    
    # Employment factor
    if employment_status == 'Unemployed':
        score += 35
    else:
        score += 15
    
    # Location factor
    if location == 'Rural':
        score += 25
    else:
        score += 15
    
    return min(100, score)

def format_currency(amount):
    """Format currency in Indian format."""
    return f"₹{amount:,.0f}"

def get_status_color(score):
    """Get color based on score."""
    if score >= 70:
        return "#28a745"  # Green
    elif score >= 40:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

"""
Data Module - AI Government Assistance Platform
------------------------------------------------
Handles synthetic data generation and government scheme database.
Uses scikit-learn's make_classification for creating training data.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from datetime import datetime, timedelta

# Global storage for analytics (simulated user data)
_user_analytics = {
    'total_users': 0,
    'eligible_users': 0,
    'total_income': 0,
    'scheme_recommendations': {},
    'eligibility_history': []
}

def generate_training_data(n_samples=500):
    """
    Generate synthetic training dataset using sklearn's make_classification.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with features and target variable
    """
    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=9,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        weights=[0.4, 0.6],  # 60% eligible
        random_state=42
    )
    
    # Convert to realistic feature ranges
    df = pd.DataFrame(X, columns=[
        'age_raw', 'income_raw', 'education_raw', 'employment_raw',
        'location_raw', 'gender_raw', 'startup_owner_raw',
        'business_age_raw', 'business_revenue_raw'
    ])
    
    # Transform to realistic ranges
    df['age'] = ((df['age_raw'] - df['age_raw'].min()) / 
                 (df['age_raw'].max() - df['age_raw'].min()) * 62 + 18).astype(int)
    
    df['income'] = ((df['income_raw'] - df['income_raw'].min()) / 
                    (df['income_raw'].max() - df['income_raw'].min()) * 1990000 + 10000).astype(int)
    
    df['education'] = pd.cut(df['education_raw'], bins=3, labels=['School', 'Graduate', 'Postgraduate'])
    df['employment_status'] = pd.cut(df['employment_raw'], bins=2, labels=['Unemployed', 'Employed'])
    df['location'] = pd.cut(df['location_raw'], bins=2, labels=['Rural', 'Urban'])
    df['gender'] = pd.cut(df['gender_raw'], bins=2, labels=['Male', 'Female'])
    df['startup_owner'] = pd.cut(df['startup_owner_raw'], bins=2, labels=['No', 'Yes'])
    
    df['business_age'] = ((df['business_age_raw'] - df['business_age_raw'].min()) / 
                          (df['business_age_raw'].max() - df['business_age_raw'].min()) * 20).astype(int)
    
    df['business_revenue'] = ((df['business_revenue_raw'] - df['business_revenue_raw'].min()) / 
                              (df['business_revenue_raw'].max() - df['business_revenue_raw'].min()) * 9900000 + 100000).astype(int)
    
    # Add target variable
    df['eligible'] = y
    
    # Select final columns
    final_df = df[['age', 'income', 'education', 'employment_status', 'location', 
                   'gender', 'startup_owner', 'business_age', 'business_revenue', 'eligible']]
    
    return final_df

def get_scheme_database():
    """
    Returns comprehensive government scheme database.
    
    Returns:
        DataFrame with scheme information
    """
    schemes = {
        'Scheme Name': [
            'Startup India',
            'Mudra Loan - Shishu',
            'Mudra Loan - Kishore',
            'Mudra Loan - Tarun',
            'Skill India (PMKVY)',
            'Women Entrepreneurship Program',
            'Stand-Up India',
            'Youth Startup Program',
            'Credit Guarantee Scheme',
            'PM Employment Generation Program',
            'Atal Innovation Mission',
            'Rural Self Employment',
            'Digital India Initiative',
            'Make in India',
            'National Rural Livelihood Mission'
        ],
        'Category': [
            'Startup Funding',
            'Micro Finance',
            'Small Business Loan',
            'Business Expansion',
            'Skill Development',
            'Women Empowerment',
            'SC/ST/Women Enterprise',
            'Youth Development',
            'Credit Support',
            'Employment',
            'Innovation',
            'Rural Development',
            'Digital Skills',
            'Manufacturing',
            'Rural Livelihood'
        ],
        'Min Income': [
            0, 0, 50000, 200000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100000, 0
        ],
        'Max Income': [
            500000, 300000, 500000, 1000000, 300000, 400000, 500000, 400000, 
            600000, 500000, 600000, 300000, 400000, 800000, 250000
        ],
        'Benefits': [
            'Tax exemption, funding up to ₹10 lakhs, mentorship',
            'Loan up to ₹50,000 for micro enterprises',
            'Loan from ₹50,000 to ₹5 lakhs for small business',
            'Loan from ₹5 lakhs to ₹10 lakhs for expansion',
            'Free skill training with certification and placement',
            'Loans up to ₹5 lakhs at subsidized rates for women',
            'Loans between ₹10 lakhs to ₹1 crore for SC/ST/Women',
            'Grants and mentorship for youth entrepreneurs',
            'Collateral-free credit guarantee up to ₹2 crores',
            'Margin money subsidy of 15-35% for projects',
            'Innovation funding, mentorship, and infrastructure',
            'Self-employment training and subsidy',
            'Digital literacy training and certification',
            'Manufacturing setup assistance and incentives',
            'Financial assistance and skill development'
        ],
        'Eligibility Criteria': [
            'Age 18-35, innovative business idea, registered startup',
            'Micro enterprise owner, good credit score',
            'Small business owner with 2+ years operation',
            'Established business looking to expand',
            'Age 18-45, unemployed or seeking upskilling',
            'Women entrepreneurs with viable business plan',
            'SC/ST/Women starting greenfield enterprise',
            'Age 18-35, first-time entrepreneur',
            'MSMEs with viable business model',
            'Age 18-60, unemployed or underemployed',
            'Age 18-40, innovative project proposal',
            'Rural resident seeking self-employment',
            'Basic literacy, willingness to learn digital skills',
            'Manufacturing business or startup',
            'Rural resident, BPL or marginalized community'
        ]
    }
    
    return pd.DataFrame(schemes)

def get_scheme_recommendations(age, income, education, employment_status, location, 
                               gender, startup_owner, business_age, business_revenue):
    """
    Get personalized scheme recommendations based on user profile.
    
    Returns:
        List of dictionaries with scheme recommendations
    """
    recommendations = []
    
    # Startup India
    if age <= 35 and startup_owner == 'Yes' and income <= 500000:
        recommendations.append({
            'scheme': 'Startup India',
            'eligibility': 85,
            'reason': 'Young entrepreneur with startup'
        })
    
    # Mudra Loan - Shishu
    if startup_owner == 'Yes' and business_revenue <= 300000:
        recommendations.append({
            'scheme': 'Mudra Loan - Shishu',
            'eligibility': 80,
            'reason': 'Micro enterprise with low revenue'
        })
    
    # Mudra Loan - Kishore
    if startup_owner == 'Yes' and 50000 <= income <= 500000 and business_age >= 2:
        recommendations.append({
            'scheme': 'Mudra Loan - Kishore',
            'eligibility': 75,
            'reason': 'Established small business'
        })
    
    # Skill India
    if age <= 45 and employment_status == 'Unemployed' and income <= 300000:
        recommendations.append({
            'scheme': 'Skill India (PMKVY)',
            'eligibility': 90,
            'reason': 'Unemployed and seeking skill development'
        })
    
    # Women Entrepreneurship
    if gender == 'Female' and startup_owner == 'Yes' and income <= 400000:
        recommendations.append({
            'scheme': 'Women Entrepreneurship Program',
            'eligibility': 85,
            'reason': 'Women entrepreneur with viable business'
        })
    
    # Youth Startup Program
    if age <= 35 and startup_owner == 'Yes' and income <= 400000:
        recommendations.append({
            'scheme': 'Youth Startup Program',
            'eligibility': 80,
            'reason': 'Young first-time entrepreneur'
        })
    
    # Stand-Up India
    if startup_owner == 'Yes' and income <= 500000:
        recommendations.append({
            'scheme': 'Stand-Up India',
            'eligibility': 70,
            'reason': 'New enterprise seeking funding'
        })
    
    # Rural Self Employment
    if location == 'Rural' and income <= 300000:
        recommendations.append({
            'scheme': 'Rural Self Employment',
            'eligibility': 75,
            'reason': 'Rural resident seeking self-employment'
        })
    
    # Default recommendation if none match
    if not recommendations:
        recommendations.append({
            'scheme': 'Credit Guarantee Scheme',
            'eligibility': 60,
            'reason': 'General credit support available'
        })
    
    # Sort by eligibility score
    recommendations.sort(key=lambda x: x['eligibility'], reverse=True)
    
    return recommendations[:5]  # Return top 5

def update_analytics(age, income, eligible, recommended_scheme):
    """
    Update analytics data (simulated).
    
    Args:
        age: User age
        income: User income
        eligible: Eligibility status (0 or 1)
        recommended_scheme: Recommended scheme name
    """
    _user_analytics['total_users'] += 1
    _user_analytics['eligible_users'] += eligible
    _user_analytics['total_income'] += income
    
    # Track scheme recommendations
    if recommended_scheme not in _user_analytics['scheme_recommendations']:
        _user_analytics['scheme_recommendations'][recommended_scheme] = 0
    _user_analytics['scheme_recommendations'][recommended_scheme] += 1
    
    # Add to history
    _user_analytics['eligibility_history'].append({
        'timestamp': datetime.now(),
        'eligible': eligible
    })

def get_analytics_summary():
    """
    Get analytics summary.
    
    Returns:
        Dictionary with analytics metrics
    """
    total = max(_user_analytics['total_users'], 1)  # Avoid division by zero
    
    summary = {
        'total_users': _user_analytics['total_users'],
        'eligible_users': _user_analytics['eligible_users'],
        'eligible_percentage': (_user_analytics['eligible_users'] / total) * 100,
        'average_income': _user_analytics['total_income'] / total if total > 0 else 0,
        'most_recommended': max(_user_analytics['scheme_recommendations'], 
                                key=_user_analytics['scheme_recommendations'].get)
                            if _user_analytics['scheme_recommendations'] else 'N/A',
        'scheme_counts': _user_analytics['scheme_recommendations'].copy(),
        'eligibility_history': _user_analytics['eligibility_history'].copy()
    }
    
    return summary

def reset_analytics():
    """Reset analytics data."""
    global _user_analytics
    _user_analytics = {
        'total_users': 0,
        'eligible_users': 0,
        'total_income': 0,
        'scheme_recommendations': {},
        'eligibility_history': []
    }

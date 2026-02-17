"""
AI Government Assistance Platform - ML Model
--------------------------------------------
Enhanced ML model with support for business metrics and multiple features.

Model: RandomForestClassifier (scikit-learn)
Features: age, income, education, employment, location, gender, startup_owner, business_age, business_revenue
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import streamlit as st

# Initialize label encoders for categorical features
education_encoder = LabelEncoder()
employment_encoder = LabelEncoder()
location_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
startup_encoder = LabelEncoder()

@st.cache_resource
def train_model():
    """
    Trains a RandomForestClassifier on the synthetic dataset.
    Uses caching to avoid retraining on every run.
    
    Returns:
        model: Trained RandomForestClassifier
        encoders: Dictionary containing label encoders for categorical features
    """
    # Import data module
    from data import generate_training_data
    
    # Load training data
    df = generate_training_data(n_samples=500)
    
    # Encode categorical features
    education_encoder.fit(['School', 'Graduate', 'Postgraduate'])
    employment_encoder.fit(['Employed', 'Unemployed'])
    location_encoder.fit(['Urban', 'Rural'])
    gender_encoder.fit(['Male', 'Female'])
    startup_encoder.fit(['No', 'Yes'])
    
    df['education_encoded'] = education_encoder.transform(df['education'])
    df['employed_encoded'] = employment_encoder.transform(df['employment_status'])
    df['location_encoded'] = location_encoder.transform(df['location'])
    df['gender_encoded'] = gender_encoder.transform(df['gender'])
    df['startup_encoded'] = startup_encoder.transform(df['startup_owner'])
    
    # Prepare features (X) and target (y)
    X = df[['age', 'income', 'education_encoded', 'employed_encoded', 
            'location_encoded', 'gender_encoded', 'startup_encoded',
            'business_age', 'business_revenue']]
    y = df['eligible']
    
    # Initialize and train the Random Forest Classifier
    # n_estimators=100 for robust predictions
    # max_depth=10 prevents overfitting
    # random_state=42 ensures reproducibility
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Save the model using joblib
    model_path = os.path.join(os.path.dirname(__file__), 'eligibility_model.pkl')
    joblib.dump(model, model_path)
    
    # Save encoders
    encoders_dict = {
        'education': education_encoder, 
        'employment': employment_encoder,
        'location': location_encoder,
        'gender': gender_encoder,
        'startup': startup_encoder
    }
    encoders_path = os.path.join(os.path.dirname(__file__), 'encoders.pkl')
    joblib.dump(encoders_dict, encoders_path)
    
    return model, encoders_dict

@st.cache_resource
def load_model():
    """
    Loads the trained model and encoders from disk, or trains a new one if not found.
    Uses caching to avoid reloading on every run.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'eligibility_model.pkl')
    encoders_path = os.path.join(os.path.dirname(__file__), 'encoders.pkl')
    
    try:
        # Try to load existing model
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
    except (FileNotFoundError, EOFError, Exception):
        # Train new model if files don't exist or are corrupted
        model, encoders = train_model()
    
    return model, encoders

# Initialize the global model
model, encoders = load_model()

def predict_eligibility(age, income, education, employment_status, location, gender, 
                       startup_owner='No', business_age=0, business_revenue=0):
    """
    Predicts government scheme eligibility for a given citizen profile.
    
    Parameters:
        age (int): Age of the applicant (18-80 years)
        income (int): Annual income in INR
        education (str): Education level ('School', 'Graduate', 'Postgraduate')
        employment_status (str): Employment status ('Employed', 'Unemployed')
        location (str): Location type ('Urban', 'Rural')
        gender (str): Gender ('Male', 'Female')
        startup_owner (str): Startup ownership ('Yes', 'No')
        business_age (int): Age of business in years
        business_revenue (int): Annual business revenue
    
    Returns:
        tuple: (prediction, confidence_score, eligibility_score)
            - prediction (str): 'Eligible' or 'Not Eligible'
            - confidence_score (float): Confidence percentage (0-100)
            - eligibility_score (float): Overall eligibility score (0-100)
    """
    try:
        # Encode categorical inputs
        education_encoded = encoders['education'].transform([education])[0]
        employed_encoded = encoders['employment'].transform([employment_status])[0]
        location_encoded = encoders['location'].transform([location])[0]
        gender_encoded = encoders['gender'].transform([gender])[0]
        startup_encoded = encoders['startup'].transform([startup_owner])[0]
        
        # Create input feature array
        input_features = np.array([[age, income, education_encoded, employed_encoded, 
                                   location_encoded, gender_encoded, startup_encoded,
                                   business_age, business_revenue]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Get probability scores for confidence calculation
        probabilities = model.predict_proba(input_features)[0]
        confidence_score = probabilities[prediction] * 100
        
        # Calculate overall eligibility score (0-100)
        eligibility_score = probabilities[1] * 100  # Probability of being eligible
        
        # Convert numeric prediction to human-readable format
        result = "Eligible" if prediction == 1 else "Not Eligible"
        
        return result, confidence_score, eligibility_score
    except Exception as e:
        # Log error and return error message
        print(f"Prediction error: {str(e)}")
        return "Error", 0.0, 0.0

# Optional: Function to get model feature importance
def get_feature_importance():
    """
    Returns the importance of each feature in the random forest.
    Useful for understanding which factors most influence eligibility.
    """
    try:
        feature_names = ['Age', 'Income', 'Education', 'Employment Status', 
                        'Location', 'Gender', 'Startup Owner', 'Business Age', 'Business Revenue']
        importances = model.feature_importances_
        
        return dict(zip(feature_names, importances))
    except Exception as e:
        # Log error and return empty dict
        print(f"Feature importance error: {str(e)}")
        return {}

def get_model_accuracy():
    """
    Calculate model accuracy on training data (for monitoring).
    
    Returns:
        float: Accuracy score (0-1)
    """
    try:
        from data import generate_training_data
        df = generate_training_data(n_samples=100)
        
        # Encode features
        df['education_encoded'] = encoders['education'].transform(df['education'])
        df['employed_encoded'] = encoders['employment'].transform(df['employment_status'])
        df['location_encoded'] = encoders['location'].transform(df['location'])
        df['gender_encoded'] = encoders['gender'].transform(df['gender'])
        df['startup_encoded'] = encoders['startup'].transform(df['startup_owner'])
        
        X = df[['age', 'income', 'education_encoded', 'employed_encoded', 
                'location_encoded', 'gender_encoded', 'startup_encoded',
                'business_age', 'business_revenue']]
        y = df['eligible']
        
        accuracy = model.score(X, y)
        return accuracy
    except Exception as e:
        print(f"Accuracy calculation error: {str(e)}")
        return 0.0

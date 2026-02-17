"""
AI Government Scheme Eligibility Predictor - ML Model
------------------------------------------------------
This module handles the machine learning logic for predicting
eligibility for government schemes based on citizen demographics.

Model: DecisionTreeClassifier (scikit-learn)
Features: age, income, education, employment status
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize label encoders for categorical features
education_encoder = LabelEncoder()
employment_encoder = LabelEncoder()

# Create training dataset
# This dataset represents historical eligibility decisions
def create_training_data():
    """
    Creates a synthetic training dataset for government scheme eligibility.
    
    Features:
    - age: Age of the applicant (18-80 years)
    - income: Annual income in INR (10,000 - 2,000,000)
    - education: Education level (School, Graduate, Postgraduate)
    - employed: Employment status (Employed, Unemployed)
    - eligible: Target variable (1 = Eligible, 0 = Not Eligible)
    
    Eligibility Logic:
    - Priority given to unemployed individuals with lower income
    - Education level influences eligibility positively
    - Age between 18-60 is preferred range
    - Income below 500,000 INR increases eligibility
    """
    data = {
        'age': [23, 45, 32, 28, 55, 19, 38, 42, 25, 35,
                50, 22, 40, 30, 48, 26, 44, 33, 29, 52,
                24, 39, 46, 27, 31, 60, 21, 37, 43, 34,
                49, 36, 41, 54, 20, 47, 56, 25, 30, 38,
                28, 33, 45, 51, 24, 40, 35, 29, 42, 26],
        'income': [250000, 800000, 450000, 180000, 1200000, 120000, 550000, 950000, 200000, 600000,
                   1500000, 150000, 700000, 300000, 1100000, 220000, 850000, 400000, 280000, 1300000,
                   190000, 620000, 1000000, 170000, 380000, 1800000, 140000, 500000, 900000, 420000,
                   1400000, 480000, 750000, 1600000, 130000, 1050000, 1700000, 210000, 320000, 580000,
                   260000, 440000, 920000, 1450000, 195000, 780000, 520000, 290000, 880000, 230000],
        'education': ['School', 'Graduate', 'Graduate', 'School', 'Postgraduate', 'School', 'Graduate', 
                      'Postgraduate', 'School', 'Graduate', 'Postgraduate', 'School', 'Graduate', 
                      'Graduate', 'Postgraduate', 'School', 'Graduate', 'Graduate', 'School', 'Postgraduate',
                      'School', 'Graduate', 'Postgraduate', 'School', 'Graduate', 'Postgraduate', 'School',
                      'Graduate', 'Graduate', 'Graduate', 'Postgraduate', 'Graduate', 'Graduate', 'Postgraduate',
                      'School', 'Graduate', 'Postgraduate', 'School', 'Graduate', 'Graduate',
                      'School', 'Graduate', 'Graduate', 'Postgraduate', 'School', 'Graduate', 'Graduate',
                      'School', 'Graduate', 'School'],
        'employed': ['Unemployed', 'Employed', 'Unemployed', 'Unemployed', 'Employed', 'Unemployed', 
                     'Employed', 'Employed', 'Unemployed', 'Employed', 'Employed', 'Unemployed', 
                     'Employed', 'Unemployed', 'Employed', 'Unemployed', 'Employed', 'Unemployed', 
                     'Unemployed', 'Employed', 'Unemployed', 'Employed', 'Employed', 'Unemployed',
                     'Unemployed', 'Employed', 'Unemployed', 'Employed', 'Employed', 'Unemployed',
                     'Employed', 'Employed', 'Employed', 'Employed', 'Unemployed', 'Employed', 'Employed',
                     'Unemployed', 'Unemployed', 'Employed', 'Unemployed', 'Unemployed', 'Employed',
                     'Employed', 'Unemployed', 'Employed', 'Employed', 'Unemployed', 'Employed', 'Unemployed'],
        'eligible': [1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                     0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
                     1, 0, 0, 1, 1, 0, 1, 0, 0, 1,
                     0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
                     1, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    }
    
    return pd.DataFrame(data)

# Train the model
def train_model():
    """
    Trains a DecisionTreeClassifier on the synthetic dataset.
    
    Returns:
        model: Trained DecisionTreeClassifier
        encoders: Dictionary containing label encoders for categorical features
    """
    # Load training data
    df = create_training_data()
    
    # Encode categorical features
    education_encoder.fit(['School', 'Graduate', 'Postgraduate'])
    employment_encoder.fit(['Employed', 'Unemployed'])
    
    df['education_encoded'] = education_encoder.transform(df['education'])
    df['employed_encoded'] = employment_encoder.transform(df['employed'])
    
    # Prepare features (X) and target (y)
    X = df[['age', 'income', 'education_encoded', 'employed_encoded']]
    y = df['eligible']
    
    # Initialize and train the Decision Tree Classifier
    # max_depth=5 prevents overfitting on small dataset
    # random_state=42 ensures reproducibility
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, {'education': education_encoder, 'employment': employment_encoder}

# Initialize the global model
model, encoders = train_model()

def predict_eligibility(age, income, education, employed):
    """
    Predicts government scheme eligibility for a given citizen profile.
    
    Parameters:
        age (int): Age of the applicant (18-80 years)
        income (int): Annual income in INR
        education (str): Education level ('School', 'Graduate', 'Postgraduate')
        employed (str): Employment status ('Employed', 'Unemployed')
    
    Returns:
        tuple: (prediction, confidence_score)
            - prediction (str): 'Eligible' or 'Not Eligible'
            - confidence_score (float): Confidence percentage (0-100)
    """
    # Encode categorical inputs
    education_encoded = encoders['education'].transform([education])[0]
    employed_encoded = encoders['employment'].transform([employed])[0]
    
    # Create input feature array
    input_features = np.array([[age, income, education_encoded, employed_encoded]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Get probability scores for confidence calculation
    # predict_proba returns probabilities for each class [Not Eligible, Eligible]
    probabilities = model.predict_proba(input_features)[0]
    confidence_score = probabilities[prediction] * 100  # Convert to percentage
    
    # Convert numeric prediction to human-readable format
    result = "Eligible" if prediction == 1 else "Not Eligible"
    
    return result, confidence_score

# Optional: Function to get model feature importance
def get_feature_importance():
    """
    Returns the importance of each feature in the decision tree.
    Useful for understanding which factors most influence eligibility.
    """
    feature_names = ['Age', 'Income', 'Education', 'Employment Status']
    importances = model.feature_importances_
    
    return dict(zip(feature_names, importances))

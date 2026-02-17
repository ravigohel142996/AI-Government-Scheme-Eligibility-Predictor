"""
AI Government Scheme Eligibility Predictor - ML Model
------------------------------------------------------
This module handles the machine learning logic for predicting
eligibility for government schemes based on citizen demographics.

Model: RandomForestClassifier (scikit-learn)
Features: age, income, education, employment status, location, gender
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Initialize label encoders for categorical features
education_encoder = LabelEncoder()
employment_encoder = LabelEncoder()
location_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

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
    - location: Location type (Urban, Rural)
    - gender: Gender (Male, Female)
    - eligible: Target variable (1 = Eligible, 0 = Not Eligible)
    
    Eligibility Logic:
    - Priority given to unemployed individuals with lower income
    - Education level influences eligibility positively
    - Age between 18-60 is preferred range
    - Income below 500,000 INR increases eligibility
    - Rural location may get priority in some schemes
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
        'location': ['Rural', 'Urban', 'Rural', 'Rural', 'Urban', 'Rural', 'Urban', 'Urban', 'Rural', 'Urban',
                     'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Rural', 'Urban',
                     'Rural', 'Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Rural', 'Urban', 'Urban', 'Rural',
                     'Urban', 'Urban', 'Urban', 'Urban', 'Rural', 'Urban', 'Urban', 'Rural', 'Rural', 'Urban',
                     'Rural', 'Rural', 'Urban', 'Urban', 'Rural', 'Urban', 'Urban', 'Rural', 'Urban', 'Rural'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male',
                   'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female',
                   'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male',
                   'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
                   'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male'],
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
    Trains a RandomForestClassifier on the synthetic dataset.
    
    Returns:
        model: Trained RandomForestClassifier
        encoders: Dictionary containing label encoders for categorical features
    """
    # Load training data
    df = create_training_data()
    
    # Encode categorical features
    education_encoder.fit(['School', 'Graduate', 'Postgraduate'])
    employment_encoder.fit(['Employed', 'Unemployed'])
    location_encoder.fit(['Urban', 'Rural'])
    gender_encoder.fit(['Male', 'Female'])
    
    df['education_encoded'] = education_encoder.transform(df['education'])
    df['employed_encoded'] = employment_encoder.transform(df['employed'])
    df['location_encoded'] = location_encoder.transform(df['location'])
    df['gender_encoded'] = gender_encoder.transform(df['gender'])
    
    # Prepare features (X) and target (y)
    X = df[['age', 'income', 'education_encoded', 'employed_encoded', 'location_encoded', 'gender_encoded']]
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
        'gender': gender_encoder
    }
    encoders_path = os.path.join(os.path.dirname(__file__), 'encoders.pkl')
    joblib.dump(encoders_dict, encoders_path)
    
    return model, encoders_dict

# Load or train the model
def load_model():
    """
    Loads the trained model and encoders from disk, or trains a new one if not found.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'eligibility_model.pkl')
    encoders_path = os.path.join(os.path.dirname(__file__), 'encoders.pkl')
    
    try:
        # Try to load existing model
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
    except:
        # Train new model if files don't exist
        model, encoders = train_model()
    
    return model, encoders

# Initialize the global model
model, encoders = load_model()

def predict_eligibility(age, income, education, employed, location, gender):
    """
    Predicts government scheme eligibility for a given citizen profile.
    
    Parameters:
        age (int): Age of the applicant (18-80 years)
        income (int): Annual income in INR
        education (str): Education level ('School', 'Graduate', 'Postgraduate')
        employed (str): Employment status ('Employed', 'Unemployed')
        location (str): Location type ('Urban', 'Rural')
        gender (str): Gender ('Male', 'Female')
    
    Returns:
        tuple: (prediction, confidence_score)
            - prediction (str): 'Eligible' or 'Not Eligible'
            - confidence_score (float): Confidence percentage (0-100)
    """
    try:
        # Encode categorical inputs
        education_encoded = encoders['education'].transform([education])[0]
        employed_encoded = encoders['employment'].transform([employed])[0]
        location_encoded = encoders['location'].transform([location])[0]
        gender_encoded = encoders['gender'].transform([gender])[0]
        
        # Create input feature array
        input_features = np.array([[age, income, education_encoded, employed_encoded, location_encoded, gender_encoded]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Get probability scores for confidence calculation
        # predict_proba returns probabilities for each class [Not Eligible, Eligible]
        probabilities = model.predict_proba(input_features)[0]
        confidence_score = probabilities[prediction] * 100  # Convert to percentage
        
        # Convert numeric prediction to human-readable format
        result = "Eligible" if prediction == 1 else "Not Eligible"
        
        return result, confidence_score
    except Exception as e:
        # Return error message if prediction fails
        return "Error", 0.0

# Optional: Function to get model feature importance
def get_feature_importance():
    """
    Returns the importance of each feature in the random forest.
    Useful for understanding which factors most influence eligibility.
    """
    try:
        feature_names = ['Age', 'Income', 'Education', 'Employment Status', 'Location', 'Gender']
        importances = model.feature_importances_
        
        return dict(zip(feature_names, importances))
    except Exception as e:
        # Return empty dict if error occurs
        return {}

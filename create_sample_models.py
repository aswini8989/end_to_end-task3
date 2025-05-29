"""
This script creates sample model and scaler files for testing purposes.
Run this if you don't have actual model.pkl and scaler.pkl files.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

def create_sample_models():
    """Create sample ML model and scaler for demonstration"""
    
    # Generate sample regression data
    X, y = make_regression(n_samples=1000, n_features=6, noise=0.1, random_state=42)
    
    # Scale the target to represent price-like values (e.g., house prices)
    y = (y - y.min()) / (y.max() - y.min()) * 500000 + 100000  # Range: $100k - $600k
    
    # Create and train a simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create and fit a scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save the model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Sample model and scaler created successfully!")
    print("Files created:")
    print("- model.pkl (RandomForestRegressor)")
    print("- scaler.pkl (StandardScaler)")
    print("\nModel expects 6 features as input.")
    
    # Test the model
    sample_features = np.random.randn(1, 6)
    scaled_features = scaler.transform(sample_features)
    prediction = model.predict(scaled_features)
    
    print(f"\nSample prediction: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    create_sample_models()

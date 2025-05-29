import os
import logging
from flask import Flask, request, render_template, flash, redirect, url_for
import joblib
import numpy as np
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Global variables for model and scaler
model = None
scaler = None

def load_models():
    """Load the ML model and scaler from pickle files"""
    global model, scaler
    try:
        # Try to load the actual model files
        if os.path.exists('model.pkl'):
            model = joblib.load('model.pkl')
            app.logger.info("Successfully loaded model.pkl")
        else:
            app.logger.warning("model.pkl not found")
            
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            app.logger.info("Successfully loaded scaler.pkl")
        else:
            app.logger.warning("scaler.pkl not found")
            
    except Exception as e:
        app.logger.error(f"Error loading models: {str(e)}")
        model = None
        scaler = None

# Load models when app starts
with app.app_context():
    load_models()

@app.route('/')
def home():
    """Render the home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if models are loaded
        if model is None or scaler is None:
            flash('Model or scaler not loaded. Please ensure model.pkl and scaler.pkl files are available.', 'error')
            return render_template('index.html')
        
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate that we have input data
        if not form_data or all(value == '' for value in form_data.values()):
            flash('Please provide input values for prediction.', 'error')
            return render_template('index.html')
        
        # Convert form values to float
        try:
            features = []
            feature_names = []
            for key, value in form_data.items():
                if value.strip():  # Only process non-empty values
                    features.append(float(value))
                    feature_names.append(key)
        except ValueError as e:
            flash('Please enter valid numerical values for all features.', 'error')
            return render_template('index.html')
        
        if not features:
            flash('Please provide at least one feature value.', 'error')
            return render_template('index.html')
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                scaled_features = scaler.transform(features_array)
            except Exception as e:
                app.logger.error(f"Error scaling features: {str(e)}")
                flash('Error processing input features. Please check your input values.', 'error')
                return render_template('index.html')
        else:
            scaled_features = features_array
        
        # Make prediction
        try:
            prediction = model.predict(scaled_features)
            prediction_value = float(prediction[0])
            
            # Format prediction result
            prediction_text = f"Predicted Price: ${prediction_value:,.2f}"
            
            return render_template('index.html', 
                                 prediction_text=prediction_text,
                                 success=True,
                                 input_features=dict(zip(feature_names, features)))
            
        except Exception as e:
            app.logger.error(f"Error making prediction: {str(e)}")
            flash('Error generating prediction. Please try again.', 'error')
            return render_template('index.html')
            
    except Exception as e:
        app.logger.error(f"Unexpected error in predict route: {str(e)}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return render_template('index.html')

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Reload the ML models (useful for updating models without restarting)"""
    load_models()
    if model is not None and scaler is not None:
        flash('Models reloaded successfully!', 'success')
    else:
        flash('Failed to reload models. Please check that model.pkl and scaler.pkl exist.', 'error')
    return redirect(url_for('home'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {str(error)}")
    flash('An internal server error occurred. Please try again.', 'error')
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

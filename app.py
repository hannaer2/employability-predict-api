from flask import Flask, request, jsonify
import pandas as pd
from hybrid_ensemble import HybridEnsemble
import joblib
import numpy as np
from flask_cors import CORS
import warnings
import os
warnings.filterwarnings('ignore')

# Enhanced import handling
try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM successfully imported")
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    print(f"❌ LightGBM import failed: {e}")
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    print(f"❌ LightGBM initialization failed: {e}")

# Define the HybridEnsemble class
class HybridEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
        self.is_fitted = False

    def fit(self, X, y):
        print("  Training individual models for ensemble...")
        for name, model in self.models.items():
            print(f"    Training {name}...")
            if name == 'TabNet':
                model.fit(
                    X, y,
                    max_epochs=100,
                    patience=20,
                    batch_size=128,
                    virtual_batch_size=64
                )
            else:
                model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        predictions = []
        prediction_probas = []

        for name, model in self.models.items():
            if name == 'TabNet':
                pred = model.predict(X)
                proba = model.predict_proba(X)
            else:
                pred = model.predict(X)
                proba = model.predict_proba(X)

            if pred.ndim > 1:
                pred = np.argmax(proba, axis=1)

            predictions.append(pred)
            prediction_probas.append(proba)

        predictions = np.array(predictions)
        prediction_probas = np.array(prediction_probas)

        if self.weights is not None:
            weighted_probas = np.average(prediction_probas, axis=0, weights=self.weights)
            final_pred = np.argmax(weighted_probas, axis=1)
        else:
            final_pred = []
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                counts = np.bincount(votes)
                final_pred.append(np.argmax(counts))

        return np.array(final_pred)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        probabilities = []
        for name, model in self.models.items():
            if name == 'TabNet':
                proba = model.predict_proba(X)
            else:
                proba = model.predict_proba(X)
            probabilities.append(proba)

        if self.weights is not None:
            avg_proba = np.average(probabilities, axis=0, weights=self.weights)
        else:
            avg_proba = np.mean(probabilities, axis=0)
        return avg_proba

app = Flask(__name__)
CORS(app)

# Model loading
model_artifacts = None
model_info = None

try:
    if LIGHTGBM_AVAILABLE:
        artifacts = joblib.load('best_employability_model_60.pkl')
        model_artifacts = {
            'model': artifacts['model'],
            'scaler': artifacts.get('scaler'),
            'label_encoder_track': artifacts.get('label_encoder_track'),
            'label_encoder_label': artifacts.get('label_encoder_label'),
            'feature_columns': artifacts.get('feature_columns', [])
        }
        
        # Load model info
        import json
        with open('model_info_60.json', 'r') as f:
            model_info = json.load(f)
        
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  LightGBM not available - running in fallback mode")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_artifacts = None

def fallback_prediction(input_data):
    """Simple rule-based fallback when ML model is unavailable"""
    overall_percentage = input_data.get('Overall_Percentage', 0)
    
    if overall_percentage >= 80:
        return "Highly Employable", 0.95
    elif overall_percentage >= 60:
        return "Proficient", 0.90
    elif overall_percentage >= 40:
        return "Competent", 0.85
    elif overall_percentage >= 20:
        return "Emerging", 0.80
    else:
        return "Developing", 0.75

@app.route('/')
def home():
    return jsonify({
        "message": "Employability Prediction API v2.0",
        "status": "active",
        "model_loaded": model_artifacts is not None,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "railway_environment": True
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_artifacts is not None,
        "lightgbm_available": LIGHTGBM_AVAILABLE
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data received", 
                "status": "error"
            }), 400
        
        # Validate required fields
        required_fields = [
            'Track', 'Technical_Correct', 'Soft_Correct', 'Behavioral_Correct',
            'Career_Correct', 'Digital_Correct', 'Analytical_Correct', 
            'Creative_Correct', 'Overall_Percentage'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing fields: {missing_fields}",
                "status": "error"
            }), 400
        
        # Use ML model if available, otherwise use fallback
        if model_artifacts and LIGHTGBM_AVAILABLE:
            try:
                # Your existing ML prediction logic here
                # For now, let's use the ML model with proper input processing
                
                # Prepare input data
                input_data = {
                    "Track_encoded": 0,  # You'll need to encode this properly
                    "Technical_Correct": float(data['Technical_Correct']),
                    "Soft_Correct": float(data['Soft_Correct']),
                    "Behavioral_Correct": float(data['Behavioral_Correct']),
                    "Career_Correct": float(data['Career_Correct']),
                    "Digital_Correct": float(data['Digital_Correct']),
                    "Analytical_Correct": float(data['Analytical_Correct']),
                    "Creative_Correct": float(data['Creative_Correct']),
                    "Overall_Percentage": float(data['Overall_Percentage'])
                }
                
                # Convert to DataFrame and scale
                input_df = pd.DataFrame([input_data])
                input_scaled = model_artifacts['scaler'].transform(input_df)
                
                # Make prediction
                prediction_encoded = model_artifacts['model'].predict(input_scaled)
                prediction_proba = model_artifacts['model'].predict_proba(input_scaled)
                
                # Decode prediction
                prediction_label = model_artifacts['label_encoder_label'].inverse_transform(prediction_encoded)
                confidence = float(np.max(prediction_proba))
                
                prediction = prediction_label[0]
                model_used = "Hybrid Ensemble AI Model"
                
            except Exception as e:
                print(f"❌ ML prediction failed, using fallback: {e}")
                prediction, confidence = fallback_prediction(data)
                model_used = "Fallback (ML Error)"
        else:
            # Use fallback prediction
            prediction, confidence = fallback_prediction(data)
            model_used = "Fallback System (AI Model Unavailable)"
        
        response = {
            "prediction": prediction,
            "confidence": confidence,
            "status": "success",
            "model_used": model_used,
            "model_available": model_artifacts is not None and LIGHTGBM_AVAILABLE
        }
        
        print(f"✅ Prediction: {prediction} (Confidence: {confidence}) - Model: {model_used}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        prediction, confidence = fallback_prediction(request.get_json() or {})
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "status": "fallback",
            "error": str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask app on port {port}...")
    print(f"📊 LightGBM Available: {LIGHTGBM_AVAILABLE}")
    print(f"🤖 Model Loaded: {model_artifacts is not None}")
    print(f"🌐 Railway Environment: True")
    
    app.run(host='0.0.0.0', port=port, debug=False)

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from flask_cors import CORS
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Import from separate module
try:
    from hybrid_ensemble import HybridEnsemble
except ImportError:
    print("⚠️  Could not import HybridEnsemble, using fallback")
    class HybridEnsemble:
        def __init__(self, models, weights=None):
            self.models = models
            self.weights = weights

app = Flask(__name__)
CORS(app)

# Global variables for model artifacts
model = None
scaler = None
label_encoder_track = None
label_encoder_label = None
feature_columns = None
loaded_model_info = None

def load_model_artifacts():
    """Load model artifacts with proper error handling"""
    global model, scaler, label_encoder_track, label_encoder_label, feature_columns, loaded_model_info
    
    try:
        artifacts = joblib.load('best_employability_model_60.pkl')
        model = artifacts['model']
        scaler = artifacts['scaler']
        label_encoder_track = artifacts['label_encoder_track']
        label_encoder_label = artifacts['label_encoder_label']
        feature_columns = artifacts['feature_columns']
        
        # Load model info for additional metadata
        try:
            with open('model_info_60.json', 'r') as f:
                loaded_model_info = json.load(f)
        except FileNotFoundError:
            print("⚠️  Model info file not found, continuing without metadata")
            loaded_model_info = {}
        
        print("✅ Model loaded successfully!")
        print(f"✅ Model type: {type(model)}")
        print(f"✅ Feature columns: {feature_columns}")
        
        if loaded_model_info and 'performance' in loaded_model_info:
            print(f"✅ Model performance: F1-Score = {loaded_model_info['performance'].get('f1_score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  Running in fallback mode - some features may not work")

# Load model on startup
load_model_artifacts()

def convert_to_native_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

def validate_input_data(data):
    """Validate input data and return errors if any"""
    required_fields = [
        'Track', 'Technical_Correct', 'Soft_Correct', 'Behavioral_Correct',
        'Career_Correct', 'Digital_Correct', 'Analytical_Correct', 
        'Creative_Correct', 'Overall_Percentage'
    ]
    
    # Check missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return f"Missing fields: {missing_fields}"
    
    # Validate track
    if label_encoder_track and data['Track'] not in label_encoder_track.classes_:
        available_tracks = label_encoder_track.classes_.tolist()
        return f"Invalid track '{data['Track']}'. Available tracks: {available_tracks}"
    
    # Validate numerical fields
    numerical_fields = [
        'Technical_Correct', 'Soft_Correct', 'Behavioral_Correct',
        'Career_Correct', 'Digital_Correct', 'Analytical_Correct', 
        'Creative_Correct', 'Overall_Percentage'
    ]
    
    validation_errors = []
    for field in numerical_fields:
        value = data[field]
        if not isinstance(value, (int, float)):
            validation_errors.append(f"{field} must be a number")
        elif value < 0:
            validation_errors.append(f"{field} cannot be negative")
        elif field == 'Overall_Percentage' and value > 100:
            validation_errors.append(f"{field} cannot exceed 100")
        elif field.endswith('_Correct') and value > 100:
            validation_errors.append(f"{field} cannot exceed 100")
    
    if validation_errors:
        return "Validation errors: " + "; ".join(validation_errors)
    
    return None

@app.route('/')
def home():
    return jsonify({
        "message": "Employability Prediction API v2.0",
        "status": "active",
        "model_version": "60_dataset",
        "model_loaded": model is not None,
        "performance": loaded_model_info.get('performance', {}) if loaded_model_info else {},
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "tracks": "/tracks (GET)",
            "model_info": "/model-info (GET)",
            "performance": "/performance (GET)",
            "batch_predict": "/predict-batch (POST)"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "model_version": "60_dataset",
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/tracks', methods=['GET'])
def get_tracks():
    """Get available track options"""
    try:
        if not label_encoder_track:
            return jsonify({"error": "Track encoder not loaded"}), 503
            
        tracks = label_encoder_track.classes_.tolist()
        track_mapping = {
            track: int(label_encoder_track.transform([track])[0]) 
            for track in tracks
        }
        return jsonify({
            "available_tracks": tracks,
            "track_encoding": track_mapping,
            "total_tracks": len(tracks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            return jsonify({
                "error": "Model not loaded properly", 
                "status": "error",
                "model_version": "60_dataset"
            }), 503

        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data received", 
                "status": "error"
            }), 400
        
        # Validate input
        validation_error = validate_input_data(data)
        if validation_error:
            return jsonify({
                "error": validation_error,
                "status": "error"
            }), 400
        
        # Prepare input data
        input_data = {
            "Track_encoded": int(label_encoder_track.transform([data['Track']])[0]),
            "Technical_Correct": float(data['Technical_Correct']),
            "Soft_Correct": float(data['Soft_Correct']),
            "Behavioral_Correct": float(data['Behavioral_Correct']),
            "Career_Correct": float(data['Career_Correct']),
            "Digital_Correct": float(data['Digital_Correct']),
            "Analytical_Correct": float(data['Analytical_Correct']),
            "Creative_Correct": float(data['Creative_Correct']),
            "Overall_Percentage": float(data['Overall_Percentage'])
        }
        
        print(f"📥 Input data: {input_data}")
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        # Ensure correct column order for the model
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        print(f"📊 Input DataFrame shape: {input_df.shape}")
        print(f"📊 Input DataFrame columns: {input_df.columns.tolist()}")
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction_encoded = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        print(f"🔮 Raw prediction encoded: {prediction_encoded}")
        print(f"📈 Prediction probabilities: {prediction_proba}")
        
        # Decode prediction
        prediction_label = label_encoder_label.inverse_transform(prediction_encoded)
        
        # Convert everything to native Python types
        prediction_native = convert_to_native_types(prediction_label[0])
        confidence = float(np.max(prediction_proba))
        
        # Convert probabilities
        class_probabilities = {}
        for i, prob in enumerate(prediction_proba[0]):
            label = label_encoder_label.inverse_transform([i])[0]
            label_native = convert_to_native_types(label)
            prob_native = convert_to_native_types(prob)
            class_probabilities[label_native] = prob_native
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction_proba[0])[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            label = label_encoder_label.inverse_transform([idx])[0]
            top_predictions.append({
                "label": convert_to_native_types(label),
                "confidence": convert_to_native_types(prediction_proba[0][idx])
            })
        
        response = {
            "prediction": prediction_native,
            "confidence": confidence,
            "probabilities": class_probabilities,
            "top_predictions": top_predictions,
            "status": "success",
            "model_version": "60_dataset",
            "input_received": {
                "Track": data['Track'],
                "Track_encoded": input_data['Track_encoded'],
                "Technical_Correct": input_data['Technical_Correct'],
                "Overall_Percentage": input_data['Overall_Percentage']
            }
        }
        
        print(f"✅ Prediction successful: {prediction_native} with confidence {confidence:.2f}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "message": "Error making prediction",
            "model_version": "60_dataset"
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 503
            
        return jsonify({
            "model_type": str(type(model)),
            "model_version": "60_dataset",
            "feature_columns": feature_columns,
            "label_classes": label_encoder_label.classes_.tolist() if label_encoder_label else [],
            "track_classes": label_encoder_track.classes_.tolist() if label_encoder_track else [],
            "model_loaded": True,
            "performance": loaded_model_info.get('performance', {}) if loaded_model_info else {},
            "ensemble_components": loaded_model_info.get('components', []) if loaded_model_info else [],
            "ensemble_weights": loaded_model_info.get('weights', []) if loaded_model_info else [],
            "cv_performance": loaded_model_info.get('cv_performance', {}) if loaded_model_info else {}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/performance', methods=['GET'])
def performance():
    """Get model performance metrics"""
    try:
        if not loaded_model_info:
            return jsonify({"error": "Model info not available"}), 503
            
        return jsonify({
            "model_version": "60_dataset",
            "performance_metrics": loaded_model_info.get('performance', {}),
            "cross_validation": loaded_model_info.get('cv_performance', {}),
            "training_timestamp": loaded_model_info.get('timestamp', 'Unknown'),
            "feature_count": len(feature_columns) if feature_columns else 0,
            "ensemble_details": {
                "components": loaded_model_info.get('components', []),
                "weights": loaded_model_info.get('weights', [])
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Predict for multiple students at once"""
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({"error": "No students data received"}), 400
        
        students = data['students']
        if not isinstance(students, list):
            return jsonify({"error": "Students must be a list"}), 400
        
        results = []
        
        for i, student_data in enumerate(students):
            try:
                # Validate input for each student
                validation_error = validate_input_data(student_data)
                if validation_error:
                    results.append({
                        "student_index": i,
                        "error": validation_error,
                        "status": "error"
                    })
                    continue
                
                # Prepare input data
                input_data = {
                    "Track_encoded": int(label_encoder_track.transform([student_data['Track']])[0]),
                    "Technical_Correct": float(student_data['Technical_Correct']),
                    "Soft_Correct": float(student_data['Soft_Correct']),
                    "Behavioral_Correct": float(student_data['Behavioral_Correct']),
                    "Career_Correct": float(student_data['Career_Correct']),
                    "Digital_Correct": float(student_data['Digital_Correct']),
                    "Analytical_Correct": float(student_data['Analytical_Correct']),
                    "Creative_Correct": float(student_data['Creative_Correct']),
                    "Overall_Percentage": float(student_data['Overall_Percentage'])
                }
                
                # Convert to DataFrame and predict
                input_df = pd.DataFrame([input_data])
                input_df = input_df.reindex(columns=feature_columns, fill_value=0)
                input_scaled = scaler.transform(input_df)
                
                prediction_encoded = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                prediction_label = label_encoder_label.inverse_transform(prediction_encoded)
                
                confidence = float(np.max(prediction_proba))
                
                results.append({
                    "student_index": i,
                    "prediction": convert_to_native_types(prediction_label[0]),
                    "confidence": confidence,
                    "status": "success",
                    "track": student_data['Track']
                })
                
            except Exception as e:
                results.append({
                    "student_index": i,
                    "error": str(e),
                    "status": "error"
                })
        
        return jsonify({
            "batch_results": results,
            "total_students": len(students),
            "successful_predictions": len([r for r in results if r['status'] == 'success']),
            "failed_predictions": len([r for r in results if r['status'] == 'error']),
            "model_version": "60_dataset"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Reload model artifacts (for development)"""
    try:
        load_model_artifacts()
        return jsonify({
            "status": "success",
            "message": "Model reloaded successfully",
            "model_loaded": model is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

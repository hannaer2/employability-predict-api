from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Define the HybridEnsemble class (MUST BE THE SAME AS IN TRAINING)
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

# Load model artifacts - UPDATED TO NEW MODEL
try:
    artifacts = joblib.load('best_employability_model_60.pkl')  # CHANGED FILENAME
    model = artifacts['model']
    scaler = artifacts['scaler']
    label_encoder_track = artifacts['label_encoder_track']
    label_encoder_label = artifacts['label_encoder_label']
    feature_columns = artifacts['feature_columns']
    
    # Load model info for additional metadata
    import json
    with open('model_info_60.json', 'r') as f:
        model_info = json.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"✅ Model type: {type(model)}")
    print(f"✅ Feature columns: {feature_columns}")
    print(f"✅ Model performance: F1-Score = {model_info['performance']['f1_score']:.4f}")
    print(f"✅ Ensemble components: {model_info['components']}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_info = None

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

@app.route('/')
def home():
    return jsonify({
        "message": "Employability Prediction API v2.0",
        "status": "active",
        "model_version": "60_dataset",
        "performance": model_info['performance'] if model_info else None,
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "tracks": "/tracks (GET)",
            "model_info": "/model-info (GET)",
            "performance": "/performance (GET)"
        }
    })

@app.route('/health')
def health():
    if model:
        return jsonify({
            "status": "healthy", 
            "model_loaded": True,
            "model_version": "60_dataset"
        })
    else:
        return jsonify({
            "status": "unhealthy", 
            "model_loaded": False,
            "model_version": "60_dataset"
        }), 500

@app.route('/tracks', methods=['GET'])
def get_tracks():
    """Get available track options"""
    try:
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
                "error": "Model not loaded", 
                "status": "error",
                "model_version": "60_dataset"
            }), 500

        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data received", 
                "status": "error"
            }), 400
        
        # Validate input
        required_fields = [
            'Track', 'Technical_Correct', 'Soft_Correct', 'Behavioral_Correct',
            'Career_Correct', 'Digital_Correct', 'Analytical_Correct', 
            'Creative_Correct', 'Overall_Percentage'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing fields: {missing_fields}",
                "required_fields": required_fields,
                "status": "error"
            }), 400
        
        # Validate track
        available_tracks = label_encoder_track.classes_.tolist()
        if data['Track'] not in available_tracks:
            return jsonify({
                "error": f"Invalid track. Available tracks: {available_tracks}",
                "status": "error"
            }), 400
        
        # Validate numerical ranges
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
        
        if validation_errors:
            return jsonify({
                "error": "Validation errors",
                "details": validation_errors,
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
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns]
        
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
            return jsonify({"error": "Model not loaded"}), 500
            
        return jsonify({
            "model_type": str(type(model)),
            "model_version": "60_dataset",
            "feature_columns": feature_columns,
            "label_classes": label_encoder_label.classes_.tolist(),
            "track_classes": label_encoder_track.classes_.tolist(),
            "model_loaded": True,
            "performance": model_info['performance'] if model_info else None,
            "ensemble_components": model_info['components'] if model_info else None,
            "ensemble_weights": model_info['weights'] if model_info else None,
            "cv_performance": model_info.get('cv_performance', {}) if model_info else {}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/performance', methods=['GET'])
def performance():
    """Get model performance metrics"""
    try:
        if not model_info:
            return jsonify({"error": "Model info not available"}), 500
            
        return jsonify({
            "model_version": "60_dataset",
            "performance_metrics": model_info['performance'],
            "cross_validation": model_info.get('cv_performance', {}),
            "training_timestamp": model_info.get('timestamp', 'Unknown'),
            "feature_count": len(feature_columns),
            "ensemble_details": {
                "components": model_info['components'],
                "weights": model_info['weights']
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Predict for multiple students at once"""
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({"error": "No students data received"}), 400
        
        students = data['students']
        if not isinstance(students, list):
            return jsonify({"error": "Students must be a list"}), 400
        
        results = []
        
        for i, student_data in enumerate(students):
            try:
                # Validate required fields for each student
                required_fields = [
                    'Track', 'Technical_Correct', 'Soft_Correct', 'Behavioral_Correct',
                    'Career_Correct', 'Digital_Correct', 'Analytical_Correct', 
                    'Creative_Correct', 'Overall_Percentage'
                ]
                
                missing_fields = [field for field in required_fields if field not in student_data]
                if missing_fields:
                    results.append({
                        "student_index": i,
                        "error": f"Missing fields: {missing_fields}",
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
                input_df = input_df[feature_columns]
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
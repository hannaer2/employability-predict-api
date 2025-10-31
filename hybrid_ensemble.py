# hybrid_ensemble.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class HybridEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
        self.is_fitted = False
        self.classes_ = None

    def fit(self, X, y):
        print("  Training individual models for ensemble...")
        self.classes_ = np.unique(y)
        
        for name, model in self.models.items():
            print(f"    Training {name}...")
            try:
                if hasattr(model, 'fit'):
                    model.fit(X, y)
                else:
                    print(f"    Warning: {name} doesn't have fit method")
            except Exception as e:
                print(f"    Error training {name}: {e}")
                
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        predictions = []
        prediction_probas = []

        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    prediction_probas.append(proba)
                    pred = model.predict(X)
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)
                    # Create dummy probabilities if predict_proba not available
                    proba = self._dummy_proba(pred, len(self.classes_))
                    prediction_probas.append(proba)
                else:
                    raise ValueError(f"Model {name} doesn't have predict method")
                
                predictions.append(pred)
                
            except Exception as e:
                print(f"    Error predicting with {name}: {e}")
                continue

        if not prediction_probas:
            raise ValueError("No models produced predictions")

        predictions = np.array(predictions)
        prediction_probas = np.array(prediction_probas)

        if self.weights is not None and len(self.weights) == len(prediction_probas):
            weighted_probas = np.average(prediction_probas, axis=0, weights=self.weights)
            final_pred = np.argmax(weighted_probas, axis=1)
        else:
            # Use majority voting
            final_pred = []
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                if len(votes) > 0:
                    counts = np.bincount(votes, minlength=len(self.classes_))
                    final_pred.append(np.argmax(counts))
                else:
                    # Fallback: use first model's prediction
                    final_pred.append(predictions[0, i] if len(predictions) > 0 else 0)
                    
        return np.array(final_pred)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        probabilities = []
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities.append(proba)
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)
                    proba = self._dummy_proba(pred, len(self.classes_))
                    probabilities.append(proba)
            except Exception as e:
                print(f"    Error getting probabilities from {name}: {e}")
                continue

        if not probabilities:
            raise ValueError("No models produced probabilities")

        if self.weights is not None and len(self.weights) == len(probabilities):
            avg_proba = np.average(probabilities, axis=0, weights=self.weights)
        else:
            avg_proba = np.mean(probabilities, axis=0)
            
        return avg_proba

    def _dummy_proba(self, predictions, n_classes):
        """Create dummy probabilities when predict_proba is not available"""
        proba = np.zeros((len(predictions), n_classes))
        for i, pred in enumerate(predictions):
            proba[i, pred] = 1.0
        return proba

    def get_params(self, deep=True):
        return {'models': self.models, 'weights': self.weights}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

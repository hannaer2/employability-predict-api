# hybrid_ensemble.py
import numpy as np

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
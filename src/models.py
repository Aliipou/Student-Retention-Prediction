"""
Machine Learning Models for Student Retention Prediction
Implements baseline and advanced models with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StudentRetentionModel:
    """Train and manage machine learning models for student retention prediction."""

    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize model.

        Args:
            model_type: Type of model ('logistic', 'random_forest', 'xgboost', 'lightgbm')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default hyperparameters for each model type.

        Returns:
            Dictionary of default parameters
        """
        params = {
            'logistic': {
                'max_iter': 1000,
                'random_state': self.random_state,
                'class_weight': 'balanced'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'random_state': self.random_state,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbose': -1
            }
        }
        return params.get(self.model_type, {})

    def get_param_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for hyperparameter tuning.

        Returns:
            Dictionary of parameter ranges for GridSearchCV
        """
        grids = {
            'logistic': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 4, 8]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        }
        return grids.get(self.model_type, {})

    def create_model(self, params: Optional[Dict[str, Any]] = None):
        """
        Create model instance based on model type.

        Args:
            params: Model parameters (uses defaults if None)
        """
        if params is None:
            params = self.get_default_params()

        models = {
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'xgboost': xgb.XGBClassifier,
            'lightgbm': lgb.LGBMClassifier
        }

        model_class = models.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = model_class(**params)
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Train model on training data.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)
            params: Model parameters
        """
        self.create_model(params)

        print(f"\n=== Training {self.model_type} model ===")

        # For gradient boosting models, use validation set for early stopping
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            eval_set = [(X_val, y_val)]

            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:  # lightgbm
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )
        else:
            self.model.fit(X_train, y_train)

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5, scoring='roc_auc', n_jobs=-1
        )

        print(f"Training complete!")
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 3,
        scoring: str = 'roc_auc',
        n_jobs: int = -1
    ):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs

        Returns:
            Best parameters found
        """
        print(f"\n=== Tuning hyperparameters for {self.model_type} ===")

        base_model = self.create_model()
        param_grid = self.get_param_grid()

        if not param_grid:
            print("No parameter grid defined for this model type. Using default parameters.")
            return self.get_default_params()

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print(f"\nBest parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return self.best_params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature array

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature array

        Returns:
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from model.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, path: str):
        """
        Save trained model to disk.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params
        }

        joblib.dump(state, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load trained model from disk.

        Args:
            path: Path to load model from
        """
        state = joblib.load(path)
        self.model = state['model']
        self.model_type = state['model_type']
        self.best_params = state.get('best_params')
        print(f"Model loaded from {path}")


class ModelEnsemble:
    """Ensemble of multiple models for improved predictions."""

    def __init__(self, model_types: list = None):
        """
        Initialize ensemble.

        Args:
            model_types: List of model types to include in ensemble
        """
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'lightgbm']

        self.models = [StudentRetentionModel(model_type=mt) for mt in model_types]
        self.weights = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train all models in ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print("\n=== Training Ensemble ===")

        for model in self.models:
            model.train(X_train, y_train, X_val, y_val)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted average of all models.

        Args:
            X: Feature array

        Returns:
            Averaged predicted probabilities
        """
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)

        # Simple average (can be weighted)
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using ensemble.

        Args:
            X: Feature array
            threshold: Classification threshold

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


def main():
    """Train and evaluate models."""
    # This would normally load preprocessed data
    # For now, just demonstrate model creation
    print("=== Model Module ===")
    print("Available models: logistic, random_forest, xgboost, lightgbm")

    for model_type in ['logistic', 'random_forest', 'xgboost', 'lightgbm']:
        model = StudentRetentionModel(model_type=model_type)
        print(f"\n{model_type} default params:")
        print(model.get_default_params())


if __name__ == "__main__":
    main()

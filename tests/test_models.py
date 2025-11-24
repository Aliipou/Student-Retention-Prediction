"""
Unit tests for models module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models import StudentRetentionModel, ModelEnsemble
from data_generator import StudentDataGenerator
from preprocessing import DataPreprocessor


class TestStudentRetentionModel(unittest.TestCase):
    """Test cases for StudentRetentionModel class."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate small dataset for testing
        generator = StudentDataGenerator(n_samples=200, random_state=42)
        df = generator.generate()

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(df, fit=True)

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            preprocessor.split_data(X, y, test_size=0.2, val_size=0.1, random_state=42)

        self.feature_names = preprocessor.feature_columns

    def test_model_initialization(self):
        """Test model initialization."""
        for model_type in ['logistic', 'random_forest', 'xgboost', 'lightgbm']:
            model = StudentRetentionModel(model_type=model_type)
            self.assertEqual(model.model_type, model_type)
            self.assertIsNone(model.model)

    def test_get_default_params(self):
        """Test default parameter retrieval."""
        for model_type in ['logistic', 'random_forest', 'xgboost', 'lightgbm']:
            model = StudentRetentionModel(model_type=model_type)
            params = model.get_default_params()
            self.assertIsInstance(params, dict)
            self.assertGreater(len(params), 0)

    def test_create_model(self):
        """Test model creation."""
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            model = StudentRetentionModel(model_type=model_type)
            model.create_model()
            self.assertIsNotNone(model.model)

    def test_train_random_forest(self):
        """Test training random forest model."""
        model = StudentRetentionModel(model_type='random_forest')
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        self.assertIsNotNone(model.model)
        self.assertTrue(hasattr(model.model, 'predict'))

    def test_train_xgboost(self):
        """Test training XGBoost model."""
        model = StudentRetentionModel(model_type='xgboost')
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        self.assertIsNotNone(model.model)

    def test_train_lightgbm(self):
        """Test training LightGBM model."""
        model = StudentRetentionModel(model_type='lightgbm')
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        self.assertIsNotNone(model.model)

    def test_predict(self):
        """Test model predictions."""
        model = StudentRetentionModel(model_type='random_forest')
        model.train(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_predict_proba(self):
        """Test probability predictions."""
        model = StudentRetentionModel(model_type='random_forest')
        model.train(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_test)

        self.assertEqual(proba.shape[0], len(self.X_test))
        self.assertEqual(proba.shape[1], 2)
        self.assertTrue((proba >= 0).all())
        self.assertTrue((proba <= 1).all())
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(self.X_test)))

    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = StudentRetentionModel(model_type='random_forest')
        model.train(self.X_train, self.y_train)

        importance_df = model.get_feature_importance(self.feature_names)

        self.assertEqual(len(importance_df), len(self.feature_names))
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)

    def test_predict_without_training_raises_error(self):
        """Test that prediction without training raises error."""
        model = StudentRetentionModel(model_type='random_forest')

        with self.assertRaises(ValueError):
            model.predict(self.X_test)

    def test_model_performance(self):
        """Test that model achieves reasonable performance."""
        model = StudentRetentionModel(model_type='random_forest')
        model.train(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        accuracy = (predictions == self.y_test).mean()

        # Should achieve better than random performance (slightly lower threshold for small datasets)
        self.assertGreater(accuracy, 0.5)


class TestModelEnsemble(unittest.TestCase):
    """Test cases for ModelEnsemble class."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate small dataset for testing
        generator = StudentDataGenerator(n_samples=200, random_state=42)
        df = generator.generate()

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(df, fit=True)

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            preprocessor.split_data(X, y, test_size=0.2, val_size=0.1, random_state=42)

    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        ensemble = ModelEnsemble(model_types=['random_forest', 'xgboost'])
        self.assertEqual(len(ensemble.models), 2)

    def test_ensemble_train(self):
        """Test ensemble training."""
        ensemble = ModelEnsemble(model_types=['random_forest'])
        ensemble.train(self.X_train, self.y_train, self.X_val, self.y_val)

        for model in ensemble.models:
            self.assertIsNotNone(model.model)

    def test_ensemble_predict_proba(self):
        """Test ensemble probability predictions."""
        ensemble = ModelEnsemble(model_types=['random_forest'])
        ensemble.train(self.X_train, self.y_train)

        proba = ensemble.predict_proba(self.X_test)

        self.assertEqual(proba.shape[0], len(self.X_test))
        self.assertEqual(proba.shape[1], 2)

    def test_ensemble_predict(self):
        """Test ensemble predictions."""
        ensemble = ModelEnsemble(model_types=['random_forest'])
        ensemble.train(self.X_train, self.y_train)

        predictions = ensemble.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(set(predictions).issubset({0, 1}))


if __name__ == '__main__':
    unittest.main()

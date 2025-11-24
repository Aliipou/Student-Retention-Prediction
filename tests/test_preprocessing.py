"""
Unit tests for preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocessing import DataPreprocessor
from data_generator import StudentDataGenerator


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()

        # Generate small test dataset
        generator = StudentDataGenerator(n_samples=100, random_state=42)
        self.df = generator.generate()

    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor.scaler)
        self.assertIsInstance(self.preprocessor.label_encoders, dict)
        self.assertIsNone(self.preprocessor.feature_columns)

    def test_create_features(self):
        """Test feature engineering."""
        df_original_cols = set(self.df.columns)
        df_engineered = self.preprocessor.create_features(self.df)

        # Check new features are created
        self.assertGreater(len(df_engineered.columns), len(df_original_cols))

        # Check specific engineered features exist
        expected_new_features = [
            'engagement_score', 'academic_risk_score', 'low_attendance',
            'overloaded', 'underloaded', 'financial_stress'
        ]

        for feature in expected_new_features:
            self.assertIn(feature, df_engineered.columns)

    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Introduce missing values
        df_with_missing = self.df.copy()
        df_with_missing.loc[0:5, 'gpa'] = np.nan
        df_with_missing.loc[10:15, 'major'] = np.nan

        df_filled = self.preprocessor.handle_missing_values(df_with_missing)

        # Check no missing values remain
        self.assertEqual(df_filled.isnull().sum().sum(), 0)

    def test_encode_categorical(self):
        """Test categorical encoding."""
        df_encoded = self.preprocessor.encode_categorical(self.df, fit=True)

        # Check categorical columns are numeric after encoding
        for col in self.preprocessor.categorical_columns:
            if col in df_encoded.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(df_encoded[col]))

    def test_prepare_features(self):
        """Test complete preprocessing pipeline."""
        X, y = self.preprocessor.prepare_features(self.df, fit=True)

        # Check outputs
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        # Check shapes
        self.assertEqual(X.shape[0], len(self.df))
        self.assertEqual(len(y), len(self.df))

        # Check feature columns are stored
        self.assertIsNotNone(self.preprocessor.feature_columns)

    def test_split_data(self):
        """Test data splitting."""
        X, y = self.preprocessor.prepare_features(self.df, fit=True)

        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            X, y, test_size=0.2, val_size=0.1, random_state=42
        )

        # Check shapes
        total_samples = len(X)
        expected_test = int(total_samples * 0.2)
        expected_val = int(total_samples * 0.8 * 0.1)  # 10% of remaining 80%

        self.assertAlmostEqual(len(X_test), expected_test, delta=2)
        self.assertAlmostEqual(len(X_val), expected_val, delta=2)

        # Check no data leakage
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), total_samples)

    def test_scale_features(self):
        """Test feature scaling."""
        df_prep = self.preprocessor.create_features(self.df)
        df_prep = self.preprocessor.handle_missing_values(df_prep)
        df_prep = self.preprocessor.encode_categorical(df_prep, fit=True)

        # Remove non-feature columns
        df_prep = df_prep.drop(['student_id', 'dropout_risk'], axis=1, errors='ignore')

        X_scaled = self.preprocessor.scale_features(df_prep, fit=True)

        # Check scaling (mean ~0, std ~1)
        self.assertTrue(np.abs(X_scaled.mean()) < 0.5)
        self.assertTrue(np.abs(X_scaled.std() - 1.0) < 0.5)

    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent across calls."""
        X1, y1 = self.preprocessor.prepare_features(self.df, fit=True)
        X2, y2 = self.preprocessor.prepare_features(self.df, fit=False)

        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_engineered_features_valid(self):
        """Test that engineered features have valid values."""
        df_eng = self.preprocessor.create_features(self.df)

        # Test engagement score is in valid range
        self.assertTrue((df_eng['engagement_score'] >= 0).all())
        self.assertTrue((df_eng['engagement_score'] <= 100).all())

        # Test binary features are binary
        binary_features = ['low_attendance', 'overloaded', 'underloaded',
                          'financial_stress', 'long_commute', 'non_traditional_age']

        for feature in binary_features:
            self.assertTrue(set(df_eng[feature].unique()).issubset({0, 1}))


if __name__ == '__main__':
    unittest.main()

"""
Preprocessing Module for Student Retention Prediction
Handles data cleaning, feature engineering, scaling, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import joblib
import os


class DataPreprocessor:
    """Preprocess student data for machine learning."""

    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = ['gender', 'major']
        self.numeric_columns = [
            'credits_last_sem', 'failed_courses', 'moodle_activity_score',
            'library_visits', 'login_times_last_week', 'attendance_rate',
            'gpa', 'gpa_variance', 'age', 'scholarship_amount',
            'part_time_job', 'distance_from_campus_km'
        ]

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV or Parquet file.

        Args:
            file_path: Path to data file

        Returns:
            Loaded DataFrame
        """
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        print(f"Data loaded: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        # Engagement score (composite metric)
        df['engagement_score'] = (
            df['moodle_activity_score'] / 100 * 0.4 +
            np.clip(df['library_visits'] / 20, 0, 1) * 0.3 +
            np.clip(df['login_times_last_week'] / 30, 0, 1) * 0.3
        ) * 100

        # Academic risk score
        df['academic_risk_score'] = (
            df['failed_courses'] * 20 +
            (4.0 - df['gpa']) * 10 +
            df['gpa_variance'] * 15
        )

        # Attendance category
        df['low_attendance'] = (df['attendance_rate'] < 0.7).astype(int)

        # Heavy workload indicator
        df['overloaded'] = (df['credits_last_sem'] > 18).astype(int)
        df['underloaded'] = (df['credits_last_sem'] < 12).astype(int)

        # Financial stress indicator
        df['financial_stress'] = (
            (df['scholarship_amount'] < 2000) &
            (df['part_time_job'] == 1)
        ).astype(int)

        # Commute stress
        df['long_commute'] = (df['distance_from_campus_km'] > 25).astype(int)

        # Non-traditional student
        df['non_traditional_age'] = (df['age'] > 24).astype(int)

        # GPA to attendance ratio
        df['gpa_attendance_ratio'] = df['gpa'] / (df['attendance_rate'] + 0.01)

        # Activity consistency
        df['activity_consistency'] = df['moodle_activity_score'] * df['attendance_rate']

        print(f"Feature engineering complete. New shape: {df.shape}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        # Fill numeric columns with median
        for col in self.numeric_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Fill categorical columns with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.

        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)

        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features using StandardScaler.

        Args:
            X: Feature DataFrame
            fit: Whether to fit scaler (True for training, False for inference)

        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline: create features, encode, scale.

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            Tuple of (X_scaled, y)
        """
        # Remove student_id if present
        if 'student_id' in df.columns:
            df = df.drop('student_id', axis=1)

        # Extract target variable
        if 'dropout_risk' in df.columns:
            y = df['dropout_risk'].values
            df = df.drop('dropout_risk', axis=1)
        else:
            y = None

        # Feature engineering
        df = self.create_features(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical variables
        df = self.encode_categorical(df, fit=fit)

        # Store feature columns on first fit
        if fit:
            self.feature_columns = df.columns.tolist()

        # Scale features
        X_scaled = self.scale_features(df, fit=fit)

        return X_scaled, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )

        print(f"Data split complete:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        print(f"  Dropout rates - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save(self, path: str = 'models/preprocessor.joblib'):
        """
        Save preprocessor state (scalers, encoders, feature columns).

        Args:
            path: Path to save preprocessor
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns
        }
        joblib.dump(state, path)
        print(f"Preprocessor saved to {path}")

    def load(self, path: str = 'models/preprocessor.joblib'):
        """
        Load preprocessor state.

        Args:
            path: Path to load preprocessor from
        """
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.label_encoders = state['label_encoders']
        self.feature_columns = state['feature_columns']
        self.categorical_columns = state['categorical_columns']
        self.numeric_columns = state['numeric_columns']
        print(f"Preprocessor loaded from {path}")


def main():
    """Test preprocessing pipeline."""
    preprocessor = DataPreprocessor()

    # Load data
    df = preprocessor.load_data('data/student_data.csv')

    print("\n=== Preprocessing Data ===")

    # Prepare features
    X, y = preprocessor.prepare_features(df, fit=True)

    print(f"\nProcessed features shape: {X.shape}")
    print(f"Feature columns: {preprocessor.feature_columns}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    # Save preprocessor
    preprocessor.save()

    print("\n=== Preprocessing Complete ===")


if __name__ == "__main__":
    main()

"""
Unit tests for data_generator module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_generator import StudentDataGenerator


class TestStudentDataGenerator(unittest.TestCase):
    """Test cases for StudentDataGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = StudentDataGenerator(n_samples=1000, random_state=42)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.n_samples, 1000)
        self.assertEqual(self.generator.random_state, 42)

    def test_generate_returns_dataframe(self):
        """Test that generate returns a DataFrame."""
        df = self.generator.generate()
        self.assertIsInstance(df, pd.DataFrame)

    def test_generate_correct_shape(self):
        """Test that generated data has correct number of rows."""
        df = self.generator.generate()
        self.assertEqual(len(df), 1000)

    def test_generate_has_required_columns(self):
        """Test that generated data has all required columns."""
        df = self.generator.generate()

        required_columns = [
            'student_id', 'credits_last_sem', 'failed_courses',
            'moodle_activity_score', 'library_visits', 'login_times_last_week',
            'attendance_rate', 'gpa', 'gpa_variance', 'age', 'gender', 'major',
            'scholarship_amount', 'part_time_job', 'distance_from_campus_km',
            'dropout_risk'
        ]

        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_no_missing_values(self):
        """Test that generated data has no missing values."""
        df = self.generator.generate()
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_dropout_risk_binary(self):
        """Test that dropout_risk is binary (0 or 1)."""
        df = self.generator.generate()
        unique_values = df['dropout_risk'].unique()
        self.assertTrue(set(unique_values).issubset({0, 1}))

    def test_gpa_valid_range(self):
        """Test that GPA is in valid range [0, 4]."""
        df = self.generator.generate()
        self.assertTrue((df['gpa'] >= 0).all())
        self.assertTrue((df['gpa'] <= 4).all())

    def test_attendance_rate_valid_range(self):
        """Test that attendance_rate is in valid range [0, 1]."""
        df = self.generator.generate()
        self.assertTrue((df['attendance_rate'] >= 0).all())
        self.assertTrue((df['attendance_rate'] <= 1).all())

    def test_moodle_activity_score_valid_range(self):
        """Test that moodle_activity_score is in valid range [0, 100]."""
        df = self.generator.generate()
        self.assertTrue((df['moodle_activity_score'] >= 0).all())
        self.assertTrue((df['moodle_activity_score'] <= 100).all())

    def test_failed_courses_non_negative(self):
        """Test that failed_courses is non-negative."""
        df = self.generator.generate()
        self.assertTrue((df['failed_courses'] >= 0).all())

    def test_age_reasonable_range(self):
        """Test that age is in reasonable range."""
        df = self.generator.generate()
        self.assertTrue((df['age'] >= 18).all())
        self.assertTrue((df['age'] < 50).all())

    def test_categorical_variables(self):
        """Test that categorical variables have expected values."""
        df = self.generator.generate()

        # Test gender
        valid_genders = {'Male', 'Female', 'Other'}
        self.assertTrue(set(df['gender'].unique()).issubset(valid_genders))

        # Test major
        expected_majors = ['Computer Science', 'Engineering', 'Business', 'Arts',
                          'Science', 'Medicine', 'Education', 'Social Sciences']
        self.assertTrue(set(df['major'].unique()).issubset(expected_majors))

    def test_dropout_probability_calculation(self):
        """Test that dropout probability calculation produces valid probabilities."""
        df = self.generator.generate()
        probs = self.generator._calculate_dropout_probability(df)

        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())

    def test_reproducibility(self):
        """Test that generator produces same results with same seed."""
        gen1 = StudentDataGenerator(n_samples=100, random_state=42)
        gen2 = StudentDataGenerator(n_samples=100, random_state=42)

        df1 = gen1.generate()
        df2 = gen2.generate()

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_n_samples(self):
        """Test generator with different sample sizes."""
        for n in [100, 500, 2000]:
            gen = StudentDataGenerator(n_samples=n, random_state=42)
            df = gen.generate()
            self.assertEqual(len(df), n)


if __name__ == '__main__':
    unittest.main()
